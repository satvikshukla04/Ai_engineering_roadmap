from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found")

MODEL_NAME = "gemini-embedding-001"
CACHE_FILE = Path("embedding_cache.json")

_client = genai.Client(api_key=API_KEY)
_cache: dict = {}


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def _embed_batch(texts: List[str], task_type: str, cache: dict) -> List[np.ndarray]:
    """Embed a batch of texts using Gemini, using/populating the cache by text."""

    uncached = [text for text in texts if text not in cache]

    if uncached:
        response = _client.models.embed_content(
            model=MODEL_NAME,
            contents=uncached,
            config=types.EmbedContentConfig(task_type=task_type),
        )

        # Bug fix: cache must be keyed by the original text, not the
        # response embedding object (which was the original bug —
        # `self.cache[emb] = ...` — that made every cache lookup on a
        # subsequent run miss or KeyError).
        for text, emb in zip(uncached, response.embeddings):
            vector = _normalize(np.array(emb.values, dtype=np.float32))
            cache[text] = vector.tolist()

        _save_cache(cache)

    return [np.array(cache[text], dtype=np.float32) for text in texts]


class SemanticSearch:
    """In-memory semantic search over a fixed corpus using Gemini embeddings."""

    def __init__(self) -> None:
        self.documents: List[str] = []
        self.embeddings: np.ndarray | None = None
        self.cache = _load_cache()

    def index(self, documents: List[str]) -> None:
        self.documents = documents
        vectors = _embed_batch(documents, "RETRIEVAL_DOCUMENT", self.cache)
        self.embeddings = np.vstack(vectors)

    def query(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.embeddings is None:
            raise RuntimeError("No documents indexed.")

        query_vector = _embed_batch([query], "RETRIEVAL_QUERY", self.cache)[0]
        similarities = self.embeddings @ query_vector
        indices = np.argsort(similarities)[::-1][:top_k]

        return [(self.documents[i], float(similarities[i])) for i in indices]


# Module-level engine backing the standalone function below, so
# `semantic_search(query, top_k=5)` can be called directly per the
# task spec without the caller having to manage a class instance.
_engine = SemanticSearch()


def semantic_search(query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """Return the top-k most similar indexed documents to `query`.

    Requires `_engine.index(...)` to have been called first (done in
    `__main__` below for the demo corpus).
    """
    return _engine.query(query, top_k=top_k)


if __name__ == "__main__":

    corpus = [
        "Python is a popular programming language.",
        "Machine learning is a branch of artificial intelligence.",
        "Neural networks are inspired by the human brain.",
        "Paris is the capital of France.",
        "The stock market fluctuates daily.",
        "Football is played worldwide.",
        "Deep learning uses many neural network layers.",
        "Cats are common household pets.",
        "Dogs are loyal companions.",
        "Quantum computing uses qubits.",
        "Cloud computing enables scalable infrastructure.",
        "The Amazon rainforest has rich biodiversity.",
        "Solar energy is renewable.",
        "Cybersecurity protects computer systems.",
        "Databases store structured information.",
        "APIs enable software communication.",
        "Electric vehicles reduce emissions.",
        "Blockchain powers cryptocurrencies.",
        "Natural language processing understands text.",
        "Large language models generate human-like responses.",
        "Docker packages applications into containers.",
        "Kubernetes orchestrates containerized applications.",
        "Git tracks source code changes.",
        "Linux is widely used on servers.",
        "Birds can migrate thousands of kilometers.",
        "Mount Everest is the tallest mountain.",
        "Water boils at 100 degrees Celsius.",
        "Photosynthesis converts sunlight into energy.",
        "Vaccines help prevent infectious diseases.",
        "Space telescopes observe distant galaxies.",
        "Computer vision analyzes images.",
        "Recommendation systems personalize content.",
        "Reinforcement learning learns through rewards.",
        "Data science combines statistics and programming.",
        "Graphs model relationships between entities.",
        "Robotics integrates hardware and AI.",
        "Smartphones contain powerful processors.",
        "Music can influence emotions.",
        "Chess requires strategic thinking.",
        "Baking bread involves yeast fermentation.",
        "Coffee contains caffeine.",
        "Ocean currents affect climate.",
        "Satellites provide GPS navigation.",
        "Memory management improves software performance.",
        "Compilers translate source code into machine code.",
        "Operating systems manage hardware resources.",
        "Distributed systems improve scalability.",
        "Encryption secures communication.",
        "Search engines index web pages.",
        "Artificial intelligence is transforming healthcare.",
    ]

    _engine.index(corpus)

    test_queries = [
        "AI for medicine",
        "How do neural networks learn?",
        "Programming language",
        "Renewable power",
        "Container orchestration",
        "Protecting networks",
        "Space exploration",
        "Pet animals",
        "Climate change",
        "Software version control",
    ]

    for query in test_queries:
        print("=" * 70)
        print(f"Query: {query}\n")

        for document, score in semantic_search(query, top_k=5):
            print(f"{score:.4f}  {document}")