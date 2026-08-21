import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.database import get_session
from app.db.models import Chunk, Document, User
from app.schemas import ChatRequest
from app.security import get_current_user
from app.services.rag import embed_text, generate_answer, retrieve

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def chat(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    settings = get_settings()

    result = await session.execute(
        select(Chunk).join(Document).where(Document.owner_id == current_user.id)
    )
    candidates = list(result.scalars().all())

    query_embedding = embed_text(payload.query)
    retrieved = retrieve(query_embedding, candidates, top_k=settings.top_k)
    answer = generate_answer(payload.query, retrieved)

    titles: dict[int, str] = {}
    for chunk, _ in retrieved:
        if chunk.document_id not in titles:
            doc = await session.get(Document, chunk.document_id)
            titles[chunk.document_id] = doc.title if doc else "unknown"

    citations = [
        {
            "document_id": chunk.document_id,
            "document_title": titles[chunk.document_id],
            "chunk_index": chunk.chunk_index,
            "score": round(score, 4),
        }
        for chunk, score in retrieved
    ]

    async def event_stream():
        for word in answer.split(" "):
            yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
        yield f"event: citations\ndata: {json.dumps({'citations': citations})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
