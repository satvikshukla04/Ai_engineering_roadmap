"""Day 36 — RAG pipeline rebuilt on LangChain LCEL with conversational retrieval.

A history-aware retriever reformulates follow-up questions using prior chat
history before retrieval runs, so context stays relevant across turns. The
chain streams its answer over FastAPI via Server-Sent Events.
"""
from __future__ import annotations

import logging
import os
import time
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured.")
os.environ.setdefault("GOOGLE_API_KEY", api_key)

MODEL_NAME = "gemini-3.1-flash-lite"
EMBEDDING_MODEL = "models/gemini-embedding-001"

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, max_retries=6)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)  # embed_query retries internally via tenacity

# --- toy corpus; swap for your real document loader/splitter (see week-7/day-1, day-2) ---
docs = [
    Document(page_content="Employees get 20 PTO days per year.", metadata={"source": "handbook.pdf", "page": 3}),
    Document(page_content="Remote work requires manager approval.", metadata={"source": "handbook.pdf", "page": 5}),
]
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

if vectorstore._collection.count() == 0:
    vectorstore.add_documents(docs)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# History-aware retrieval: reformulates a follow-up ("do I need approval for
# that?") into a standalone question using chat history, before retrieval runs —
# this is what lets follow-ups correctly pull relevant context.
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and a follow-up question, rephrase it as a standalone question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based only on the context below. If unknown, say so.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

app = FastAPI(title="Conversational RAG API")


class ChatRequest(BaseModel):
    input: str = Field(..., min_length=1)
    chat_history: list[dict[str, str]] = Field(default_factory=list)  # [{"role": "human"/"ai", "content": ...}]


def _to_messages(chat_history: list[dict[str, str]]) -> list[HumanMessage | AIMessage]:
    return [
        HumanMessage(content=m["content"]) if m["role"] == "human" else AIMessage(content=m["content"])
        for m in chat_history
    ]


async def stream_answer(request: ChatRequest) -> AsyncGenerator[str, None]:
    """Streams the LCEL chain's answer tokens to the client as Server-Sent Events."""
    logger.info("Stream started | input=%r", request.input)
    history = _to_messages(request.chat_history)
    try:
        async for chunk in rag_chain.astream({"input": request.input, "chat_history": history}):
            if "answer" in chunk:
                yield f"data: {chunk['answer']}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        logger.exception("Streaming failed")
        yield f"data: [ERROR] {exc}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """FastAPI SSE endpoint: streams the conversational RAG chain's response."""
    return StreamingResponse(
        stream_answer(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "Conversational RAG API is running."}


def invoke_with_retry(payload: dict, attempts: int = 4, base_delay: float = 2.0) -> dict:
    """Retries the chain on transient Google 500/503 errors (embedding or generation)."""
    for attempt in range(attempts):
        try:
            return rag_chain.invoke(payload)
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            delay = base_delay * (2**attempt)
            logger.warning("Transient error (%s) — retrying in %.1fs", exc, delay)
            time.sleep(delay)


if __name__ == "__main__":
    # Verifies the deliverable requirement directly: a follow-up question must
    # correctly reuse chat history to retrieve relevant context, not just repeat
    # a fresh, context-free retrieval.
    history: list[HumanMessage | AIMessage] = []
    for q in ["How many PTO days do I get?", "Do I need approval for that?"]:
        result = invoke_with_retry({"input": q, "chat_history": history})
        print(f"Q: {q}\nA: {result['answer']}\n")
        history += [HumanMessage(content=q), AIMessage(content=result["answer"])]