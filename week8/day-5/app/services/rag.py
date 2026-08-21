import hashlib
import re
from collections.abc import AsyncIterator

import numpy as np

from app.config import get_settings

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def embed_text(text: str) -> list[float]:
    """Deterministic local embedding (hashing bag-of-words). No network/API key required.
    Swap for a real provider by setting LLM_PROVIDER and implementing embed_text_remote."""
    settings = get_settings()
    dim = settings.embedding_dim
    vector = np.zeros(dim, dtype=np.float64)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for token in tokens:
        h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vector[idx] += sign
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def retrieve(query_embedding: list[float], candidates: list, top_k: int) -> list[tuple]:
    """candidates: list of Chunk ORM objects. Returns top_k (chunk, score) sorted by similarity desc."""
    scored = [(chunk, cosine_similarity(query_embedding, chunk.embedding)) for chunk in candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def generate_answer(query: str, retrieved: list[tuple]) -> str:
    """Mock LLM: extractive synthesis over top chunks. Deterministic, offline-safe.
    Swap for a real provider (OpenAI/Gemini) by branching on settings.llm_provider."""
    if not retrieved:
        return "I don't have enough information in the knowledge base to answer that."
    parts = [chunk.text.strip().split(". ")[0] for chunk, _ in retrieved]
    return f"Based on the retrieved context: {' '.join(parts)}."


async def stream_answer(answer: str) -> AsyncIterator[str]:
    for word in answer.split(" "):
        yield word + " "
