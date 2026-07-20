import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------
# Mock Processor Service
# -----------------------------
class Processor:
    def __init__(self):
        self.initialized = False

    async def initialize(self):
        logger.info("Initializing processor...")
        await asyncio.sleep(1)
        self.initialized = True
        logger.info("Processor ready.")

    async def shutdown(self):
        logger.info("Shutting down processor...")
        await asyncio.sleep(1)
        logger.info("Processor closed.")

    def chunk_document(self, text: str):
        size = 40
        return [text[i:i + size] for i in range(0, len(text), size)]


# -----------------------------
# Lifespan
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    processor = Processor()
    await processor.initialize()
    app.state.processor = processor
    yield
    await processor.shutdown()


app = FastAPI(lifespan=lifespan)


# -----------------------------
# Request Model
# -----------------------------
class Document(BaseModel):
    text: str


# -----------------------------
# Background Task
# -----------------------------
async def process_document(job_id: str, text: str, processor: Processor):
    logger.info("Job %s started", job_id)
    await asyncio.sleep(2)
    chunks = processor.chunk_document(text)
    logger.info("Job %s complete (%d chunks)", job_id, len(chunks))


# -----------------------------
# Route
# -----------------------------
@app.post("/process")
async def process(
    document: Document,
    request: Request,
):
    job_id = str(uuid.uuid4())
    processor = request.app.state.processor
    asyncio.create_task(
        process_document(job_id, document.text, processor)
    )
    return {
        "message": "Processing started",
        "job_id": job_id,
    }