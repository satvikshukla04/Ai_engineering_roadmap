from dataclasses import dataclass

from app.services.rag import cosine_similarity, embed_text


@dataclass
class EvalCase:
    query: str
    expected_keywords: list[str]
    relevant_document_title: str


@dataclass
class EvalResult:
    query: str
    retrieval_hit: bool
    keyword_overlap: float
    score: float


def score_case(case: EvalCase, retrieved_titles: list[str], answer: str) -> EvalResult:
    retrieval_hit = case.relevant_document_title in retrieved_titles
    answer_lower = answer.lower()
    hits = sum(1 for kw in case.expected_keywords if kw.lower() in answer_lower)
    keyword_overlap = hits / len(case.expected_keywords) if case.expected_keywords else 0.0
    score = 0.5 * float(retrieval_hit) + 0.5 * keyword_overlap
    return EvalResult(
        query=case.query, retrieval_hit=retrieval_hit, keyword_overlap=keyword_overlap, score=score
    )


def aggregate(results: list[EvalResult]) -> float:
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)


def embedding_self_similarity_sanity_check(text: str) -> float:
    """Retrieval sanity check: a chunk's embedding must be maximally similar to itself."""
    vec = embed_text(text)
    return cosine_similarity(vec, vec)
