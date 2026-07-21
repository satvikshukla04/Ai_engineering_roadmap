"""Pydantic schemas for the documents resource."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(default="", max_length=100_000)


class DocumentCreate(DocumentBase):
    """Payload for creating a document."""


class DocumentUpdate(BaseModel):
    """Payload for partially updating a document."""

    title: str | None = Field(default=None, min_length=1, max_length=255)
    content: str | None = Field(default=None, max_length=100_000)


class DocumentResponse(DocumentBase):
    """Document as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""

    items: list[DocumentResponse]
    total: int
