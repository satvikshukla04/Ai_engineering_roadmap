from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

import sys
import os

# 1. Get the absolute path of the current file's directory (week-3/day-5)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up two levels to reach your root folder (qualtech-tasks)
project_root = os.path.dirname(os.path.dirname(current_dir))

# 3. Build the path to the app folder where 'db' lives
target_db_path = os.path.join(project_root, "week-4", "day-1", "app")

# 4. Inject it into Python's system path
sys.path.append(target_db_path)

# 5. Now Python can magically find the 'db' folder directly!
from db.database import get_db
from db.repository import UserRepository

app = FastAPI()

SECRET_KEY = "your_super_secret_key_change_this"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


class UserRegister(BaseModel):
    username: str
    password: str


def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Updated dependency to fetch user from DB asynchronously
async def get_current_user(
    token: str = Depends(oauth2_scheme), 
    db: AsyncSession = Depends(get_db)
):
    username = verify_token(token)
    repo = UserRepository(db)
    user = await repo.get_by_username(username)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@app.post("/auth/register")
async def register(user: UserRegister, db: AsyncSession = Depends(get_db)):
    repo = UserRepository(db)
    
    existing_user = await repo.get_by_username(user.username)
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="User already exists"
        )

    await repo.create(
        username=user.username, 
        hashed_password=hash_password(user.password)
    )

    return {"message": "User registered successfully"}


@app.post("/auth/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: AsyncSession = Depends(get_db)
):
    repo = UserRepository(db)
    user = await repo.get_by_username(form_data.username)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )

    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )

    token = create_access_token({"sub": user.username})

    return {
        "access_token": token,
        "token_type": "bearer"
    }


@app.get("/auth/me")
async def read_me(current_user=Depends(get_current_user)):
    return {
        "username": current_user.username,
        "id": current_user.id,
        "created_at": current_user.created_at
    }