"""Database tables and the request/response shapes built on top of them.

I'm using SQLModel because it gives me one class that's both the DB table
*and* the Pydantic schema, which keeps a small project like this from
having two parallel sets of classes that drift out of sync.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import List

from pydantic import BaseModel
from sqlmodel import Field, Relationship, SQLModel


def utcnow() -> datetime:
    """Small helper so every timestamp in the app is UTC and consistent."""
    return datetime.now(UTC)


class Role(str, Enum):
    """Who sent a given message."""

    user = "user"
    assistant = "assistant"


# --------------------------------------------------------------------------
# DB tables
# --------------------------------------------------------------------------


class User(SQLModel, table=True):
    """A registered user. Passwords are stored hashed, never in plain text."""

    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    hashed_password: str
    created_at: datetime = Field(default_factory=utcnow)

    sessions: List["ChatSession"] = Relationship(back_populates="user")


class ChatSession(SQLModel, table=True):
    """One chat conversation, tied to a user and a chosen system prompt."""

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    title: str = "New chat"
    system_prompt_id: str = "default"
    created_at: datetime = Field(default_factory=utcnow)

    user: User = Relationship(back_populates="sessions")
    messages: List["Message"] = Relationship(
        back_populates="session",
        sa_relationship_kwargs={"order_by": "Message.created_at"},
    )


class Message(SQLModel, table=True):
    """A single turn in a chat session (either the user's or the model's)."""

    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="chatsession.id", index=True)
    role: Role
    content: str
    created_at: datetime = Field(default_factory=utcnow)

    session: ChatSession = Relationship(back_populates="messages")


# --------------------------------------------------------------------------
# Request / response schemas (plain Pydantic, not DB tables)
# --------------------------------------------------------------------------


class UserCreate(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class SessionCreate(BaseModel):
    title: str = "New chat"
    system_prompt_id: str = "default"


class SessionRead(BaseModel):
    id: int
    title: str
    system_prompt_id: str
    created_at: datetime


class MessageCreate(BaseModel):
    content: str


class MessageRead(BaseModel):
    id: int
    role: Role
    content: str
    created_at: datetime


class SystemPromptRead(BaseModel):
    id: str
    name: str
    prompt: str
