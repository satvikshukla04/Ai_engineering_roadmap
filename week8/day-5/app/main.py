from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db.database import init_db
from app.routers import auth, chat, documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="RAG Assistant", version="1.0.0", lifespan=lifespan)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
