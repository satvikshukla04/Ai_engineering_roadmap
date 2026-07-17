from fastapi import FastAPI

from app.routers.documents import router as document_router

app = FastAPI(
    title="Document API",
    version="1.0.0",
)

app.include_router(document_router)


@app.get("/")
def root():
    return {
        "message": "FastAPI Intro Project"
    }