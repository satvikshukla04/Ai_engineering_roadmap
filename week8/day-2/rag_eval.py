"""
Day 37 — RAG evaluation harness: 15-question eval set, scored on faithfulness
and answer relevancy via LLM-as-judge, with mean scores and per-question detail
written out for failure analysis.

Pipeline under test: BM25 retrieval over a mock knowledge base -> Gemini generation.
Judge: Gemini extracts claims / reverse-questions and scores them.

If GEMINI_API_KEY is not set, a deterministic local fallback stands in for the
LLM calls (TF-IDF based) so the whole pipeline still runs end-to-end. Swap in
your real vector store in `retrieve()` and this becomes a real CI eval.
"""
import json
import logging
import os
import re
import time
from typing import List, Tuple

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rag_eval")

try:
    from google import genai
    from google.genai import errors as genai_errors
    GEMINI_AVAILABLE = bool(os.getenv("GEMINI_API_KEY"))
except ImportError:
    GEMINI_AVAILABLE = False

if not GEMINI_AVAILABLE:
    logger.warning("No GEMINI_API_KEY found - using local keyword judge as a stand-in for LLM-as-judge.")

# Free tier: 15 RPM. We make 3 calls/question -> ~5s between calls keeps us at ~12 RPM.
_GEMINI_CALL_DELAY = 5.0
_gemini_client = None

def _gemini(model: str, prompt: str, max_retries: int = 4) -> str:
    """Single Gemini call with rate-limit backoff and mandatory inter-call sleep."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client()
    time.sleep(_GEMINI_CALL_DELAY)
    for attempt in range(max_retries):
        try:
            resp = _gemini_client.models.generate_content(model=model, contents=prompt)
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
# 1. Mock knowledge base + BM25 retriever (swap for your real vector store)
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


def retrieve(query: str, k: int = 3) -> Tuple[List[str], List[float]]:
    scores = _bm25.get_scores(query.lower().split())
    top_idx = sorted(range(len(KNOWLEDGE_BASE)), key=lambda i: scores[i], reverse=True)[:k]
    return [KNOWLEDGE_BASE[i] for i in top_idx], [round(scores[i], 3) for i in top_idx]


# ==========================================
# 2. Generation - the RAG pipeline being evaluated
# ==========================================
def generate_answer(question: str, context: List[str], context_scores: List[float]) -> str:
    if GEMINI_AVAILABLE:
        prompt = (
            "Answer the question using ONLY the context below. Be concise.\n\n"
            f"Context:\n{chr(10).join(context)}\n\nQuestion: {question}"
        )
        return _gemini("gemini-3.5-flash-lite", prompt)

    # --- Local fallback generator ---
    # When the top BM25 score is weak, the context is a poor match for the
    # question. A real LLM asked to answer anyway will often reach beyond the
    # provided context and add plausible-but-unsupported detail - this is the
    # single most common real-world faithfulness failure, so we simulate it.
    if not context or max(context_scores) < 3.0:
        base = context[0] if context else "There is no directly relevant policy on file."
        return f"{base} In general this also depends on your account tier and region, which may affect the exact terms."
    return context[0]


# ==========================================
# 3. LLM-as-judge: faithfulness + answer relevancy
# ==========================================
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "i", "you",
    "my", "your", "to", "for", "of", "in", "on", "at", "and", "or", "it", "this",
    "that", "can", "will", "be", "with", "if", "how", "what", "when", "get", "have",
    "has", "not", "also", "back", "up", "from",
}


def _split_claims(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _keywords(text: str) -> set:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def judge_faithfulness(answer: str, context: List[str]) -> Tuple[float, str]:
    """supported_claims / total_claims. Each claim is checked against the context."""
    claims = _split_claims(answer)
    if not claims:
        return 0.0, "no claims extracted"
    context_text = " ".join(context)

    if GEMINI_AVAILABLE:
        prompt = (
            "Context:\n" + context_text + "\n\nClaims:\n" +
            "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims)) +
            "\n\nFor each claim, reply YES if it is directly supported by the context, else NO. "
            "One YES/NO per line, in order, nothing else."
        )
        raw = _gemini("gemini-3.5-flash-lite", prompt)
        verdicts = [v.strip().upper() for v in raw.splitlines() if v.strip()]
        supported = sum(1 for v in verdicts if v.startswith("YES"))
        return round(supported / len(claims), 2), f"{supported}/{len(claims)} claims supported"

    # --- Local fallback judge: keyword-coverage per claim vs. context ---
    # A claim is "supported" if most of its content words also appear in the
    # retrieved context - a crude but workable stand-in for LLM entailment.
    context_kw = _keywords(context_text)
    supported = 0
    unsupported_claims = []
    for c in claims:
        claim_kw = _keywords(c)
        if not claim_kw:
            supported += 1
            continue
        coverage = len(claim_kw & context_kw) / len(claim_kw)
        if coverage >= 0.6:
            supported += 1
        else:
            unsupported_claims.append(c)
    reason = f"{supported}/{len(claims)} claims supported" + (
        f"; unsupported: {unsupported_claims}" if unsupported_claims else ""
    )
    return round(supported / len(claims), 2), reason


def judge_answer_relevancy(question: str, answer: str) -> float:
    """Reference-free: how well does the answer address the actual question?"""
    if GEMINI_AVAILABLE:
        prompt = (
            f"Answer: {answer}\n\nGenerate 3 questions this answer could be responding to, "
            "one per line, nothing else."
        )
        raw = _gemini("gemini-3.5-flash-lite", prompt)
        generated_qs = [q.strip() for q in raw.splitlines() if q.strip()][:3]
        q_kw = _keywords(question)
        coverages = [len(q_kw & _keywords(gq)) / len(q_kw) for gq in generated_qs] if q_kw else [0.0]
        return round(sum(coverages) / len(coverages), 2)

    # --- Local fallback: does the answer cover the question's key terms? ---
    q_kw = _keywords(question)
    a_kw = _keywords(answer)
    if not q_kw:
        return 0.0
    return round(len(q_kw & a_kw) / len(q_kw), 2)


# ==========================================
# 4. Run the eval set
# ==========================================
def run_eval(dataset_path: str = "rag_eval_dataset.json", out_path: str = "rag_eval_results.json"):
    """Runs the full 15-question eval set through the pipeline, scores each
    answer on faithfulness and answer relevancy, and writes mean scores plus
    per-question detail to out_path for downstream failure analysis."""
    with open(dataset_path) as f:
        questions = json.load(f)

    results = []
    for item in questions:
        q = item["question"]
        context, scores = retrieve(q, k=3)
        answer = generate_answer(q, context, scores)
        faithfulness, faith_reason = judge_faithfulness(answer, context)
        relevancy = judge_answer_relevancy(q, answer)

        logger.info(f"[{item['id']}] faithfulness={faithfulness} relevancy={relevancy}")
        results.append({
            "id": item["id"],
            "question": q,
            "retrieved_context": context,
            "top_retrieval_score": max(scores) if scores else 0,
            "answer": answer,
            "faithfulness": faithfulness,
            "faithfulness_reason": faith_reason,
            "answer_relevancy": relevancy,
        })

    summary = {
        "timestamp": time.time(),
        "judge_mode": "gemini" if GEMINI_AVAILABLE else "local_tfidf_fallback",
        "n_questions": len(results),
        "mean_faithfulness": round(sum(r["faithfulness"] for r in results) / len(results), 2),
        "mean_answer_relevancy": round(sum(r["answer_relevancy"] for r in results) / len(results), 2),
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {out_path} | mean_faithfulness={summary['mean_faithfulness']} "
                f"mean_answer_relevancy={summary['mean_answer_relevancy']}")
    return summary


if __name__ == "__main__":
    run_eval()