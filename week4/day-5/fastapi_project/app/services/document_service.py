"""Business logic for the documents resource.

Keeping this logic out of the router means the router stays a thin HTTP
layer, and this logic is trivially unit-testable without spinning up FastAPI.
"""
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Document
from app.schemas.document import DocumentCreate, DocumentUpdate


class DocumentNotFoundError(Exception):
    """Raised when a document with the given id does not exist."""

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(f"Document '{document_id}' not found")


class DocumentService:
    """Encapsulates all CRUD operations for documents."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, payload: DocumentCreate) -> Document:
        document = Document(title=payload.title, content=payload.content)
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def get(self, document_id: str) -> Document:
        document = self.db.get(Document, document_id)
        if document is None:
            raise DocumentNotFoundError(document_id)
        return document

    def list(self, skip: int = 0, limit: int = 100) -> tuple[list[Document], int]:
        total = self.db.scalar(select(func.count()).select_from(Document)) or 0
        items = list(
            self.db.scalars(select(Document).offset(skip).limit(limit).order_by(Document.created_at))
        )
        return items, total

    def update(self, document_id: str, payload: DocumentUpdate) -> Document:
        document = self.get(document_id)
        update_data = payload.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(document, field, value)
        self.db.commit()
        self.db.refresh(document)
        return document

    def delete(self, document_id: str) -> None:
        document = self.get(document_id)
        self.db.delete(document)
        self.db.commit()
