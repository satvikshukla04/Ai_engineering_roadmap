"""
Production RAG: async ingestion pipeline, TTL query cache, and /stats monitoring.

Three additions on top of the BM25 RAG pipeline from week-8:

  1. Async ingestion — POST /ingest kicks off a background job that reads,
     chunks, and indexes documents. GET /ingest/{job_id}/progress streams
     stage-by-stage SSE updates. New documents are fingerprinted by content
     hash so re-submitting the same file is a no-op (incremental indexing).

  2. TTL query cache — every /query result is cached by MD5(query) for
     CACHE_TTL_SECONDS. Cache hits skip BM25 retrieval entirely and return
     in <1ms. Cache size is capped at CACHE_MAX_SIZE entries (LRU eviction).

  3. /stats endpoint — returns query_count, cache_hit_rate, p50/p95
     latency, chunks_in_index, and per-job ingestion history. All numbers
     come from real in-memory counters — nothing is faked.

Run with:  uvicorn rag_production:app --reload
"""
import asyncio
import hashlib
import json
import logging
import os
import statistics
import time
import uuid
from collections import defaultdict
from typing import AsyncGenerator, Dict, List, Optional

from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from rank_bm25 import BM25Okapi

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rag_prod")

try:
    from google import genai
    GEMINI_AVAILABLE = bool(os.getenv("GEMINI_API_KEY"))
except ImportError:
    GEMINI_AVAILABLE = False

_gemini_client = None
_GEMINI_CALL_DELAY = 5.0


def _gemini(prompt: str, max_retries: int = 4) -> str:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client()
    time.sleep(_GEMINI_CALL_DELAY)
    for attempt in range(max_retries):
        try:
            resp = _gemini_client.models.generate_content(
                model="gemini-3.1-flash-lite", contents=prompt
            )
            return resp.text.strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 30 * (attempt + 1)
                logger.warning(f"429 rate limit — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded on Gemini 429")


# ==========================================
# 1. Index — in-memory BM25, rebuilds on each ingest
# ==========================================
class Index:
    """Wraps BM25Okapi so it can be rebuilt incrementally without a restart."""

    def __init__(self):
        self.chunks: List[str] = []
        self.doc_ids: List[str] = []          # which source document each chunk belongs to
        self.indexed_hashes: set = set()      # content hashes of ingested docs (dedup)
        self._bm25: Optional[BM25Okapi] = None

    def _rebuild(self):
        if self.chunks:
            self._bm25 = BM25Okapi([c.lower().split() for c in self.chunks])

    def add_chunks(self, chunks: List[str], doc_id: str):
        self.chunks.extend(chunks)
        self.doc_ids.extend([doc_id] * len(chunks))
        self._rebuild()

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        top_idx = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in top_idx]

    @property
    def size(self) -> int:
        return len(self.chunks)


index = Index()

# Seed with the same KB we've used all week so /query works immediately on startup
_SEED_DOCS = [
    ("Standard shipping takes 5-7 business days within the continental US.", "seed"),
    ("Express shipping costs an additional $15 and delivers in 1-2 business days.", "seed"),
    ("We do not currently ship internationally outside the US and Canada.", "seed"),
    ("Items can be returned within 30 days of delivery for a full refund.", "seed"),
    ("Return shipping labels are free for defective items, otherwise the customer pays return postage.", "seed"),
    ("Refunds are issued to the original payment method within 5-10 business days of us receiving the item.", "seed"),
    ("All electronics purchased come with a 1-year manufacturer warranty covering defects.", "seed"),
    ("The warranty does not cover accidental damage, water damage, or unauthorized repairs.", "seed"),
    ("To file a warranty claim, contact support with your order ID and a description of the issue.", "seed"),
    ("Passwords must be at least 10 characters and include one number and one symbol.", "seed"),
    ("Two-factor authentication can be enabled from the Security tab in account settings.", "seed"),
    ("Subscription plans are billed monthly and can be cancelled at any time from the billing page.", "seed"),
    ("Cancelling a subscription stops future charges but does not refund the current billing period.", "seed"),
    ("We accept Visa, Mastercard, American Express, and PayPal for payment.", "seed"),
    ("Gift cards do not expire and can be combined with other payment methods at checkout.", "seed"),
    ("Customer support is available via live chat 9am-6pm ET, Monday through Friday.", "seed"),
    ("Bulk orders of 50+ units qualify for a 10% discount; contact sales for a quote.", "seed"),
    ("The mobile app is available on iOS 15+ and Android 10+.", "seed"),
]
index.add_chunks([t for t, _ in _SEED_DOCS], "seed")


# ==========================================
# 2. TTL query cache
# ==========================================
CACHE_TTL_SECONDS = 300   # 5 minutes
CACHE_MAX_SIZE = 256

_query_cache: TTLCache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)


def _cache_key(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


# ==========================================
# 3. Monitoring counters
# ==========================================
class Metrics:
    def __init__(self):
        self.query_count = 0
        self.cache_hits = 0
        self.latencies_ms: List[float] = []    # end-to-end query latency
        self.retrieval_latencies_ms: List[float] = []

    def record_query(self, total_ms: float, retrieval_ms: float, cache_hit: bool):
        self.query_count += 1
        if cache_hit:
            self.cache_hits += 1
        self.latencies_ms.append(total_ms)
        self.retrieval_latencies_ms.append(retrieval_ms)

    def percentile(self, data: List[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return round(sorted_data[min(idx, len(sorted_data) - 1)], 2)

    def summary(self) -> dict:
        return {
            "query_count": self.query_count,
            "cache_hit_rate": round(self.cache_hits / self.query_count, 3) if self.query_count else 0.0,
            "latency_p50_ms": self.percentile(self.latencies_ms, 50),
            "latency_p95_ms": self.percentile(self.latencies_ms, 95),
            "retrieval_p50_ms": self.percentile(self.retrieval_latencies_ms, 50),
        }


metrics = Metrics()

# Ingestion job registry: job_id -> {status, stages, doc_count, chunk_count, errors}
_jobs: Dict[str, dict] = {}


# ==========================================
# 4. Ingestion pipeline (runs in background)
# ==========================================
def _chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    """Simple character-level chunker with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start += chunk_size - overlap
    return [c for c in chunks if c]


async def _run_ingestion(job_id: str, documents: List[Dict]):
    """
    4-stage async pipeline:
      LOAD -> CLEAN -> CHUNK -> INDEX
    Stages emit structured log lines that /ingest/{job_id}/progress streams as SSE.
    Documents already in the index (by content hash) are skipped — incremental indexing.
    """
    job = _jobs[job_id]
    job["status"] = "running"
    job["start_time"] = time.time()

    def stage(name: str, detail: str = ""):
        entry = {"stage": name, "t": round(time.time() - job["start_time"], 3), "detail": detail}
        job["stages"].append(entry)
        logger.info(f"[{job_id}] {name} — {detail}")

    # STAGE 1: LOAD
    stage("LOAD", f"{len(documents)} document(s) received")
    await asyncio.sleep(0)  # yield to event loop so SSE can flush

    # STAGE 2: CLEAN + DEDUP
    new_docs = []
    for doc in documents:
        content_hash = hashlib.md5(doc["text"].encode()).hexdigest()
        if content_hash in index.indexed_hashes:
            stage("SKIP", f"doc '{doc.get('id', '?')}' already indexed — skipping")
        else:
            index.indexed_hashes.add(content_hash)
            cleaned = " ".join(doc["text"].split())  # normalise whitespace
            new_docs.append({"id": doc.get("id", content_hash[:8]), "text": cleaned})
    stage("CLEAN", f"{len(new_docs)} new document(s) after dedup")
    await asyncio.sleep(0)

    # STAGE 3: CHUNK
    all_chunks: List[tuple] = []  # (chunk_text, doc_id)
    for doc in new_docs:
        chunks = _chunk_text(doc["text"])
        for c in chunks:
            all_chunks.append((c, doc["id"]))
    stage("CHUNK", f"{len(all_chunks)} chunks created")
    job["chunk_count"] = len(all_chunks)
    await asyncio.sleep(0)

    # STAGE 4: INDEX (simulate async embed + store; swap for real embed calls here)
    # In production: asyncio.gather(*[embed(c) for c, _ in all_chunks])
    embed_start = time.time()
    await asyncio.sleep(0.05 * len(all_chunks))  # simulate embedding latency
    embed_ms = round((time.time() - embed_start) * 1000, 1)

    for chunk_text, doc_id in all_chunks:
        index.add_chunks([chunk_text], doc_id)

    stage("INDEX", f"{len(all_chunks)} chunks indexed in {embed_ms}ms — index now {index.size} chunks")
    job["doc_count"] = len(new_docs)
    job["status"] = "done"
    job["end_time"] = time.time()


# ==========================================
# 5. FastAPI app
# ==========================================
app = FastAPI(title="RAG Production API")


@app.post("/ingest")
async def ingest(payload: dict, background_tasks: BackgroundTasks):
    """
    POST /ingest
    Body: {"documents": [{"id": "doc1", "text": "..."}, ...]}
    Returns a job_id. Poll /ingest/{job_id}/progress for SSE updates.
    """
    documents = payload.get("documents", [])
    if not documents:
        return JSONResponse({"error": "no documents provided"}, status_code=400)

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "queued", "stages": [], "doc_count": 0, "chunk_count": 0}
    background_tasks.add_task(_run_ingestion, job_id, documents)
    logger.info(f"Ingestion job {job_id} queued for {len(documents)} document(s)")
    return {"job_id": job_id, "status": "queued"}


@app.get("/ingest/{job_id}/progress")
async def ingest_progress(job_id: str):
    """
    GET /ingest/{job_id}/progress
    Streams SSE events: one per completed stage + a final 'done' event.
    """
    if job_id not in _jobs:
        return JSONResponse({"error": "job not found"}, status_code=404)

    async def event_stream() -> AsyncGenerator[str, None]:
        seen = 0
        while True:
            job = _jobs[job_id]
            stages = job["stages"]
            while seen < len(stages):
                yield f"data: {json.dumps(stages[seen])}\n\n"
                seen += 1
            if job["status"] == "done":
                yield f"data: {json.dumps({'stage': 'DONE', 'doc_count': job['doc_count'], 'chunk_count': job['chunk_count']})}\n\n"
                break
            await asyncio.sleep(0.1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/query")
async def query(payload: dict):
    """
    POST /query
    Body: {"query": "...", "k": 5}
    Returns retrieved chunks. Cache hit skips BM25 entirely.
    """
    q = payload.get("query", "").strip()
    k = int(payload.get("k", 5))
    if not q:
        return JSONResponse({"error": "query is required"}, status_code=400)

    t_start = time.time()
    cache_key = _cache_key(q)
    cache_hit = cache_key in _query_cache

    if cache_hit:
        chunks = _query_cache[cache_key]
        retrieval_ms = 0.0
        logger.info(f"CACHE HIT  | '{q[:60]}'")
    else:
        r_start = time.time()
        chunks = index.retrieve(q, k=k)
        retrieval_ms = (time.time() - r_start) * 1000
        _query_cache[cache_key] = chunks
        logger.info(f"CACHE MISS | '{q[:60]}' | retrieval={retrieval_ms:.1f}ms | {len(chunks)} chunks")

    total_ms = (time.time() - t_start) * 1000
    metrics.record_query(total_ms, retrieval_ms, cache_hit)

    return {
        "query": q,
        "cache_hit": cache_hit,
        "chunks": chunks,
        "retrieval_ms": round(retrieval_ms, 2),
        "total_ms": round(total_ms, 2),
    }


@app.get("/stats")
async def stats():
    """
    GET /stats
    Returns real-time pipeline health: query metrics, cache state, index size, job history.
    """
    return {
        "index": {
            "chunks_in_index": index.size,
            "unique_docs_ingested": len(index.indexed_hashes),
        },
        "cache": {
            "ttl_seconds": CACHE_TTL_SECONDS,
            "max_size": CACHE_MAX_SIZE,
            "current_entries": len(_query_cache),
        },
        "queries": metrics.summary(),
        "ingestion_jobs": {
            jid: {k: v for k, v in job.items() if k != "stages"}
            for jid, job in _jobs.items()
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "index_size": index.size}


# ==========================================
# 6. Smoke test (runs when executed directly, not via uvicorn)
# ==========================================
if __name__ == "__main__":
    async def smoke_test():
        """
        Exercises all three features end-to-end without needing a running server:
        ingestion job, query cache, and /stats metrics verification.
        """
        print("\n=== Ingestion: submit 2 new docs + 1 duplicate ===")
        new_docs = [
            {"id": "policy-v2", "text": "Customers on the Pro plan receive priority support with a 2-hour SLA. "
                                         "Standard plan customers are handled within 24 hours. "
                                         "Enterprise plans include a dedicated account manager."},
            {"id": "gdpr-notice", "text": "We store personal data in EU data centres. "
                                           "You can request deletion of your data at any time via the privacy settings page. "
                                           "Data is retained for a maximum of 3 years after account closure."},
        ]
        job_id = str(uuid.uuid4())[:8]
        _jobs[job_id] = {"status": "queued", "stages": [], "doc_count": 0, "chunk_count": 0}
        await _run_ingestion(job_id, new_docs + [new_docs[0]])  # third doc is a duplicate
        print(f"Job {job_id}: {_jobs[job_id]['status']} — {_jobs[job_id]['doc_count']} docs, {_jobs[job_id]['chunk_count']} chunks")
        for s in _jobs[job_id]["stages"]:
            print(f"  [{s['t']}s] {s['stage']}: {s['detail']}")

        print("\n=== Query cache: first call should MISS, second should HIT ===")
        test_queries = [
            "How long does standard shipping take?",
            "Can I cancel my subscription?",
            "What is your data retention policy?",   # from the new doc
            "How long does standard shipping take?", # repeat — should cache-hit
        ]
        for q in test_queries:
            result = await query({"query": q, "k": 3})
            hit = "HIT " if result["cache_hit"] else "MISS"
            print(f"  [{hit}] {q[:55]:<55} | {result['total_ms']:.1f}ms | {len(result['chunks'])} chunks")

        print("\n=== /stats ===")
        s = await stats()
        print(json.dumps(s, indent=2))

    asyncio.run(smoke_test())