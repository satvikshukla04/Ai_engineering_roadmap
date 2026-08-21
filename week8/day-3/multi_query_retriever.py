"""
Day 38 — Multi-query retrieval: generate 3 Gemini-rephrased variants of each
query, retrieve the top-5 chunks for each, deduplicate, and return a merged
top-10. Tested on 5 queries to document the recall improvement over
single-query retrieval.

Why: a single BM25/embedding query can miss relevant chunks due to vocabulary
mismatch (e.g. "cancel subscription" vs "stop recurring billing"). Generating
3 surface-level paraphrases of the same intent and merging their result sets
dramatically improves recall at the cost of N extra LLM calls.

Falls back to rule-based variant generation if GEMINI_API_KEY is not set so
the whole pipeline still runs end-to-end without a key.
"""
import hashlib
import json
import logging
import os
import time
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("multi_query")

try:
    from google import genai
    GEMINI_AVAILABLE = bool(os.getenv("GEMINI_API_KEY"))
except ImportError:
    GEMINI_AVAILABLE = False

if not GEMINI_AVAILABLE:
    logger.warning("No GEMINI_API_KEY — using rule-based query variants as fallback.")

# Free tier: 15 RPM. One variant-generation call per query = safe at any cadence.
# If you chain this with generation/judging, keep _GEMINI_CALL_DELAY = 5.0.
_GEMINI_CALL_DELAY = 2.0
_gemini_client = None


def _gemini(prompt: str, max_retries: int = 4) -> str:
    """Gemini call with per-call sleep and exponential-backoff on 429."""
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
# 1. Same knowledge base as rag_eval.py — swap for your real vector store
# ==========================================
KNOWLEDGE_BASE = [
    "Standard shipping takes 5-7 business days within the continental US.",
    "Express shipping costs an additional $15 and delivers in 1-2 business days.",
    "We do not currently ship internationally outside the US and Canada.",
    "Items can be returned within 30 days of delivery for a full refund.",
    "Return shipping labels are free for defective items, otherwise the customer pays return postage.",
    "Refunds are issued to the original payment method within 5-10 business days of us receiving the item.",
    "All electronics purchased come with a 1-year manufacturer warranty covering defects.",
    "The warranty does not cover accidental damage, water damage, or unauthorized repairs.",
    "To file a warranty claim, contact support with your order ID and a description of the issue.",
    "You can create an account using an email address or by signing in with Google.",
    "Passwords must be at least 10 characters and include one number and one symbol.",
    "Two-factor authentication can be enabled from the Security tab in account settings.",
    "Subscription plans are billed monthly and can be cancelled at any time from the billing page.",
    "Cancelling a subscription stops future charges but does not refund the current billing period.",
    "We accept Visa, Mastercard, American Express, and PayPal for payment.",
    "Gift cards do not expire and can be combined with other payment methods at checkout.",
    "Order status can be tracked in real time from the 'My Orders' page after logging in.",
    "Customer support is available via live chat 9am-6pm ET, Monday through Friday.",
    "Bulk orders of 50+ units qualify for a 10% discount; contact sales for a quote.",
    "The mobile app is available on iOS 15+ and Android 10+.",
]
_tokenized_kb = [doc.lower().split() for doc in KNOWLEDGE_BASE]
_bm25 = BM25Okapi(_tokenized_kb)


def _chunk_id(text: str) -> str:
    """Stable hash for deduplication — same content = same id regardless of retrieval order."""
    return hashlib.md5(text.encode()).hexdigest()[:8]


def bm25_retrieve(query: str, k: int = 5) -> List[str]:
    """Return top-k chunks for a single query."""
    scores = _bm25.get_scores(query.lower().split())
    top_idx = sorted(range(len(KNOWLEDGE_BASE)), key=lambda i: scores[i], reverse=True)[:k]
    return [KNOWLEDGE_BASE[i] for i in top_idx]


# ==========================================
# 2. Query variant generation — 3 Gemini-rephrased variants per query
# ==========================================
def generate_variants_llm(query: str, n: int = 3) -> List[str]:
    """Ask Gemini to rephrase the query N times, targeting vocabulary diversity."""
    prompt = (
        f"Rephrase the following question {n} different ways, using different vocabulary "
        "each time. Return only the rephrased questions, one per line, no numbering.\n\n"
        f"Question: {query}"
    )
    raw = _gemini(prompt)
    variants = [v.strip() for v in raw.splitlines() if v.strip()][:n]
    logger.info(f"  variants: {variants}")
    return variants


def generate_variants_fallback(query: str, n: int = 3) -> List[str]:
    """
    Rule-based paraphrasing when no LLM is available.
    Covers the most common vocabulary-mismatch patterns:
      - swap question words (how/what/tell me about)
      - add/remove 'policy' / 'details about'
    Good enough to demonstrate deduplication and recall-merge logic.
    """
    prefixes = ["Tell me about", "What is the policy on", "Explain"]
    suffixes = ["?", " in detail?", " — what should I know?"]
    variants = []
    # Strip leading question word to get the topic
    topic = query.lstrip("WhHowDoICanWillAre ").rstrip("?").strip().lower()
    for i in range(n):
        v = f"{prefixes[i % len(prefixes)]} {topic}{suffixes[i % len(suffixes)]}"
        variants.append(v)
    return variants[:n]


def generate_variants(query: str, n: int = 3) -> List[str]:
    return generate_variants_llm(query, n) if GEMINI_AVAILABLE else generate_variants_fallback(query, n)


# ==========================================
# 3. Multi-query retrieval: retrieve top-5 per variant, dedupe, merge top-10
# ==========================================
def single_query_retrieve(query: str, k: int = 5) -> List[str]:
    return bm25_retrieve(query, k=k)


def multi_query_retrieve(query: str, n_variants: int = 3, k_per_variant: int = 5, top_n: int = 10) -> Dict:
    """
    1. Generate n_variants rephrased versions of the query.
    2. Retrieve top k_per_variant chunks for each variant + the original.
    3. Deduplicate by content hash, preserving first-seen order.
    4. Return merged top_n chunks + metadata for recall comparison.
    """
    all_queries = [query] + generate_variants(query, n=n_variants)
    seen_ids = set()
    merged: List[str] = []
    per_query_results: Dict[str, List[str]] = {}

    for q in all_queries:
        chunks = bm25_retrieve(q, k=k_per_variant)
        per_query_results[q] = chunks
        for chunk in chunks:
            cid = _chunk_id(chunk)
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(chunk)

    return {
        "original_query": query,
        "variants": all_queries[1:],
        "per_query_results": per_query_results,
        "merged_chunks": merged[:top_n],
        "total_unique": len(merged),
    }


# ==========================================
# 4. Recall comparison: single-query vs. multi-query, across 5 test queries
# ==========================================
TEST_QUERIES = [
    "How do I stop my recurring subscription charges?",     # target: chunks 12,13 — "cancel", "stop future charges"
    "My item arrived broken — what are my options?",        # target: chunks 6,7,8 — warranty, defective return label
    "Is there a faster delivery option?",                   # target: chunks 0,1 — express shipping
    "How do I secure my account better?",                   # target: chunks 10,11 — password, 2FA
    "Can I use a gift voucher at checkout?",                # target: chunk 15 — gift cards
]

# Ground-truth relevant chunks (by index in KNOWLEDGE_BASE) — used to compute recall.
# A chunk is "relevant" if a human would expect it to help answer that specific question.
RELEVANT_CHUNKS: List[List[int]] = [
    [12, 13],       # subscription cancel + no-refund-current-period
    [6, 7, 8, 4],   # warranty coverage, exclusions, claim process, defective return label
    [0, 1],         # standard + express shipping
    [10, 11],       # password requirements, 2FA
    [15],           # gift cards
]


def recall_at_k(retrieved: List[str], relevant_indices: List[int]) -> float:
    """Fraction of relevant chunks that appear in the retrieved set."""
    relevant_texts = {KNOWLEDGE_BASE[i] for i in relevant_indices}
    retrieved_set = set(retrieved)
    hits = relevant_texts & retrieved_set
    return round(len(hits) / len(relevant_texts), 2)


def run_comparison():
    """Runs all 5 test queries through both single-query and multi-query
    retrieval, and writes the recall comparison (documenting the recall
    improvement) to multi_query_results.json."""
    results = []
    for query, relevant_idx in zip(TEST_QUERIES, RELEVANT_CHUNKS):
        logger.info(f"Query: {query}")

        single = single_query_retrieve(query, k=5)
        single_recall = recall_at_k(single, relevant_idx)

        multi = multi_query_retrieve(query, n_variants=3, k_per_variant=5, top_n=10)
        multi_recall = recall_at_k(multi["merged_chunks"], relevant_idx)

        improvement = round(multi_recall - single_recall, 2)
        logger.info(f"  single recall={single_recall}  multi recall={multi_recall}  delta={improvement:+.2f}")

        results.append({
            "query": query,
            "variants": multi["variants"],
            "single_query": {"chunks": single, "recall": single_recall},
            "multi_query": {
                "merged_chunks": multi["merged_chunks"],
                "total_unique": multi["total_unique"],
                "recall": multi_recall,
            },
            "recall_improvement": improvement,
        })

    mean_single = round(sum(r["single_query"]["recall"] for r in results) / len(results), 2)
    mean_multi = round(sum(r["multi_query"]["recall"] for r in results) / len(results), 2)

    report = {
        "mode": "gemini" if GEMINI_AVAILABLE else "rule_based_fallback",
        "mean_single_recall": mean_single,
        "mean_multi_recall": mean_multi,
        "mean_improvement": round(mean_multi - mean_single, 2),
        "results": results,
    }

    out_path = "multi_query_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Mean single-query recall : {mean_single}")
    print(f"Mean multi-query recall  : {mean_multi}")
    print(f"Mean improvement         : {mean_multi - mean_single:+.2f}")
    print(f"Full results             : {out_path}")
    return report


if __name__ == "__main__":
    run_comparison()