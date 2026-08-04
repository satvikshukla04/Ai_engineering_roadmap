"""The FastAPI app.

Endpoints:
  POST /auth/register              create a user
  POST /auth/token                 log in, get a JWT (OAuth2 password flow)
  GET  /system-prompts              list selectable system prompts
  POST /sessions                    create a chat session (pick a system prompt)
  GET  /sessions                    list your sessions
  GET  /sessions/{id}/history        full message history for a session
  POST /sessions/{id}/messages       send a message, get the reply streamed back (SSE)

Everything under /sessions is auth-protected: you can only see/use your own
sessions.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select
from starlette.responses import StreamingResponse

from genai_chat_api.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    hash_password,
)
from genai_chat_api.db import get_session, init_db
from genai_chat_api.llm import get_provider
from genai_chat_api.models import (
    ChatSession,
    Message,
    MessageCreate,
    MessageRead,
    Role,
    SessionCreate,
    SessionRead,
    SystemPromptRead,
    Token,
    User,
    UserCreate,
)
from genai_chat_api.system_prompts import get_system_prompt, list_system_prompts


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(title="genai_chat_api", lifespan=lifespan)

SessionDep = Annotated[Session, Depends(get_session)]
CurrentUserDep = Annotated[User, Depends(get_current_user)]


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


@app.post("/auth/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, session: SessionDep) -> Token:
    existing = session.exec(select(User).where(User.username == user_in.username)).first()
    if existing is not None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Username already taken")

    user = User(username=user_in.username, hashed_password=hash_password(user_in.password))
    session.add(user)
    session.commit()
    session.refresh(user)

    return Token(access_token=create_access_token(user.username))


@app.post("/auth/token", response_model=Token)
def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: SessionDep,
) -> Token:
    user = authenticate_user(session, form_data.username, form_data.password)
    if user is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Token(access_token=create_access_token(user.username))


# --------------------------------------------------------------------------
# System prompts
# --------------------------------------------------------------------------


@app.get("/system-prompts", response_model=list[SystemPromptRead])
def get_system_prompts() -> list[SystemPromptRead]:
    return list_system_prompts()


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------


def _get_owned_session(session_id: int, user: User, db: Session) -> ChatSession:
    """Load a chat session, 404-ing if it doesn't exist or isn't the user's.

    Returning 404 (not 403) for "exists but isn't yours" avoids leaking
    which session ids exist to other users.
    """
    chat_session = db.get(ChatSession, session_id)
    if chat_session is None or chat_session.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return chat_session


@app.post("/sessions", response_model=SessionRead, status_code=status.HTTP_201_CREATED)
def create_session(
    session_in: SessionCreate, user: CurrentUserDep, db: SessionDep
) -> ChatSession:
    chat_session = ChatSession(
        user_id=user.id,
        title=session_in.title,
        system_prompt_id=session_in.system_prompt_id,
    )
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)
    return chat_session


@app.get("/sessions", response_model=list[SessionRead])
def list_sessions(user: CurrentUserDep, db: SessionDep) -> list[ChatSession]:
    return list(db.exec(select(ChatSession).where(ChatSession.user_id == user.id)).all())


@app.get("/sessions/{session_id}/history", response_model=list[MessageRead])
def get_history(session_id: int, user: CurrentUserDep, db: SessionDep) -> list[Message]:
    chat_session = _get_owned_session(session_id, user, db)
    return list(chat_session.messages)


# --------------------------------------------------------------------------
# Streaming chat
# --------------------------------------------------------------------------


def _sse_event(event: str, data: dict[str, str]) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: int,
    message_in: MessageCreate,
    user: CurrentUserDep,
    db: SessionDep,
) -> StreamingResponse:
    chat_session = _get_owned_session(session_id, user, db)

    # 1. Persist the user's message immediately.
    user_message = Message(session_id=chat_session.id, role=Role.user, content=message_in.content)
    db.add(user_message)
    db.commit()

    # 2. Build the conversation history to send to the model.
    history = [{"role": m.role.value, "content": m.content} for m in chat_session.messages]
    system_prompt = get_system_prompt(chat_session.system_prompt_id)
    provider = get_provider()

    async def event_stream() -> AsyncIterator[str]:
        full_reply = ""
        try:
            async for chunk in provider.stream_reply(system_prompt, history):
                full_reply += chunk
                yield _sse_event("chunk", {"content": chunk})
        finally:
            # Persist whatever we streamed, even on a mid-stream failure,
            # so history stays consistent with what the client saw.
            if full_reply:
                assistant_message = Message(
                    session_id=chat_session.id, role=Role.assistant, content=full_reply
                )
                db.add(assistant_message)
                db.commit()
        yield _sse_event("done", {"content": full_reply})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
