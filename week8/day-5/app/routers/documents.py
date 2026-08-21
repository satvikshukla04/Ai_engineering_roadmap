from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_session
from app.db.models import Chunk, Document, User
from app.schemas import DocumentCreate, DocumentOut
from app.security import get_current_user
from app.services.rag import chunk_text, embed_text

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def create_document(
    payload: DocumentCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> DocumentOut:
    document = Document(title=payload.title, owner_id=current_user.id)
    session.add(document)
    await session.flush()

    for idx, chunk in enumerate(chunk_text(payload.content)):
        session.add(
            Chunk(document_id=document.id, chunk_index=idx, text=chunk, embedding=embed_text(chunk))
        )
    await session.commit()
    await session.refresh(document, attribute_names=["chunks"])
    return DocumentOut(id=document.id, title=document.title, chunk_count=len(document.chunks))


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[DocumentOut]:
    result = await session.execute(select(Document).where(Document.owner_id == current_user.id))
    documents = result.scalars().unique().all()
    out = []
    for doc in documents:
        await session.refresh(doc, attribute_names=["chunks"])
        out.append(DocumentOut(id=doc.id, title=doc.title, chunk_count=len(doc.chunks)))
    return out


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> DocumentOut:
    document = await session.get(Document, document_id, options=[])
    if document is None or document.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    await session.refresh(document, attribute_names=["chunks"])
    return DocumentOut(id=document.id, title=document.title, chunk_count=len(document.chunks))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    document = await session.get(Document, document_id)
    if document is None or document.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    await session.delete(document)
    await session.commit()
