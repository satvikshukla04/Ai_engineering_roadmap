"""FastAPI SSE endpoint that streams Gemini responses token by token.

Daily task spec:
    POST /chat/stream
    body: {"message": str, "session_id": str}
    -> streams Gemini response token by token via SSE, sends [DONE] on completion

Verify with:
    curl -N -X POST http://localhost:8000/chat/stream \
        -H "Content-Type: application/json" \
        -d '{"message": "hello", "session_id": "abc123"}'
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from google import genai
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured.")

MODEL_NAME = "gemini-2.5-flash"

client = genai.Client(api_key=API_KEY)
app = FastAPI(title="Streaming Gemini Chat API")


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


def _sse(data: dict[str, str]) -> str:
    """Format a single SSE data frame."""
    return f"data: {json.dumps(data)}\n\n"


async def stream_response(message: str, session_id: str) -> AsyncGenerator[str, None]:
    """Stream Gemini output as Server-Sent Events, one token per frame."""
    start = time.perf_counter()
    chunk_count = 0
    logger.info("Stream started | session=%s", session_id)

    try:
        stream = await client.aio.models.generate_content_stream(
            model=MODEL_NAME,
            contents=message,
        )
        async for chunk in stream:
            # chunk.text raises ValueError on non-text (e.g. metadata-only) chunks
            try:
                text = chunk.text or ""
            except ValueError:
                text = ""

            if not text:
                continue

            chunk_count += 1
            yield _sse({"token": text})

    except Exception as exc:  # noqa: BLE001 - surfaced to the client as an SSE frame
        logger.exception("Streaming failed | session=%s", session_id)
        yield _sse({"error": str(exc)})

    finally:
        duration = time.perf_counter() - start
        logger.info(
            "Stream ended | session=%s | duration=%.2fs | chunks=%d",
            session_id,
            duration,
            chunk_count,
        )
        # Sentinel is always sent, even after an error, so clients can stop reading.
        yield "data: [DONE]\n\n"


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream a Gemini response over SSE for the given message/session."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    return StreamingResponse(
        stream_response(request.message, request.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "Streaming API is running."}