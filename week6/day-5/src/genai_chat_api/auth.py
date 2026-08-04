"""Auth: password hashing + JWT bearer tokens.

Kept intentionally simple:
- passwords are hashed with bcrypt (never store plaintext)
- login returns a signed JWT
- protected routes depend on get_current_user, which decodes the JWT and
  loads the matching user from the DB

SECRET_KEY should come from an environment variable in real deployments.
It falls back to a fixed dev value here so the project runs out of the box,
but that fallback is NOT safe for production use.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Annotated

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlmodel import Session, select

from genai_chat_api.db import get_session
from genai_chat_api.models import User

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-secret-do-not-use-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# Note: went with the `bcrypt` package directly instead of passlib's
# CryptContext. passlib's bcrypt backend-detection code has a known
# incompatibility with recent bcrypt releases (it probes with an
# oversized dummy password and throws), so calling bcrypt ourselves
# sidesteps that entirely. bcrypt itself still caps inputs at 72 bytes,
# which is plenty for a password.


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(username: str) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    return str(jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM))


def authenticate_user(session: Session, username: str, password: str) -> User | None:
    user = session.exec(select(User).where(User.username == username)).first()
    if user is None or not verify_password(password, user.hashed_password):
        return None
    return user


def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: Annotated[Session, Depends(get_session)],
) -> User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_error
    except JWTError as exc:
        raise credentials_error from exc

    user = session.exec(select(User).where(User.username == username)).first()
    if user is None:
        raise credentials_error
    return user
