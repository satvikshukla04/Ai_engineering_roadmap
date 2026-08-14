"""Day 35 — Generation layer: query + retrieved chunks -> prompt -> Gemini -> parsed citations.

Retrieves relevant chunks, constructs a prompt, calls Gemini, and parses the
citations out of the response. The no-result path returns a clear "not
found" message instead of calling the model. Tested on 10 queries below.
"""
from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = "gemini-3.1-flash-lite"
SIM_THRESHOLD = 0.5

SYSTEM_PROMPT = (
    "Answer based only on the provided context. If the context doesn't contain "
    "the answer, say so. After each fact, add a citation in the format "
    "[Source: {source}, page {page}]. Only cite sources from the provided context."
)

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Missing API key: set GEMINI_API_KEY in your .env file.")

client = genai.Client(api_key=api_key)


def retrieve(query: str, corpus: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """Filters retrieved chunks by similarity threshold and returns the top_k.

    Replace with a real vector store `.search()` call (see week-7/day-3) —
    this assumes chunks already carry a similarity `score` from that step.
    """
    chunks = [c for c in corpus if c["score"] >= SIM_THRESHOLD]
    chunks.sort(key=lambda c: -c["score"])
    return chunks[:top_k]


def build_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    """Delimits chunks with '---' and places the best chunk first and last
    to avoid the 'lost in the middle' problem."""
    ordered = sorted(chunks, key=lambda c: -c["score"])
    if len(ordered) > 1:
        ordered = [ordered[0]] + ordered[1:] + [ordered[0]]
    context = "\n---\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}" for c in ordered
    )
    return f"Context:\n{context}\n\nQuestion: {query}"


def parse_citations(text: str) -> list[tuple[str, str]]:
    return re.findall(r"\[Source:\s*(.*?),\s*page\s*(\d+)\]", text)


def verify_citations(citations: list[tuple[str, str]], chunks: list[dict[str, Any]]) -> bool:
    valid = {(c["source"], str(c["page"])) for c in chunks}
    return all((src, page) in valid for src, page in citations)


def generate(query: str, corpus: list[dict[str, Any]]) -> str:
    """Retrieve -> build prompt -> call Gemini -> verify citations.

    No-result path: if nothing clears the similarity threshold, skip the
    model call entirely and return a clear "not found" response.
    """
    chunks = retrieve(query, corpus)
    if not chunks:
        return "I could not find relevant information about this in the available documents."

    prompt = build_prompt(query, chunks)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    answer = response.text or ""

    citations = parse_citations(answer)
    if citations and not verify_citations(citations, chunks):
        answer += "\n\n[Warning: response cited a source not in retrieved context]"
    return answer


if __name__ == "__main__":
    corpus = [
        {"source": "handbook.pdf", "page": 3, "score": 0.82, "text": "Employees get 20 PTO days per year."},
        {"source": "handbook.pdf", "page": 5, "score": 0.71, "text": "Remote work requires manager approval."},
        {"source": "handbook.pdf", "page": 8, "score": 0.65, "text": "New hires complete onboarding within 2 weeks."},
        {"source": "benefits.pdf", "page": 1, "score": 0.60, "text": "Health insurance starts on day one of employment."},
        {"source": "benefits.pdf", "page": 2, "score": 0.55, "text": "401k matching is up to 4% of salary."},
        {"source": "handbook.pdf", "page": 12, "score": 0.30, "text": "Office parking is available on a first-come basis."},
    ]

    # 10 test queries, including two with no relevant context to exercise
    # the "not found" path.
    test_queries = [
        "How many PTO days do employees get?",
        "Do I need approval for remote work?",
        "How long is onboarding?",
        "When does health insurance start?",
        "What is the 401k match?",
        "What is the company's stock price?",   # no relevant context -> not-found path
        "Who is the CEO?",                       # no relevant context -> not-found path
        "Can I work remotely without asking my manager?",
        "Is parking free?",
        "What benefits do new hires get on day one?",
    ]

    for q in test_queries:
        print(f"Q: {q}\nA: {generate(q, corpus)}\n")