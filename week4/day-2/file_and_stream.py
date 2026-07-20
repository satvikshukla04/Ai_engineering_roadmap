import asyncio
import json
import os
from pathlib import Path

import aiofiles
import magic
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Check reported content type
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Read first bytes to detect actual type
    first_bytes = await file.read(2048)

    mime = magic.from_buffer(first_bytes, mime=True)

    if mime != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid PDF file.")

    await file.seek(0)

    file_size = 0
    save_path = UPLOAD_DIR / file.filename

    async with aiofiles.open(save_path, "wb") as out_file:
        while chunk := await file.read(1024 * 1024):
            file_size += len(chunk)

            if file_size > MAX_FILE_SIZE:
                await out_file.close()
                os.remove(save_path)
                raise HTTPException(
                    status_code=400,
                    detail="File exceeds 10 MB limit."
                )

            await out_file.write(chunk)

    return {
        "filename": file.filename,
        "content_type": mime,
        "size_bytes": file_size,
        "saved_to": str(save_path),
    }


async def fake_gemini_stream():
    tokens = [
        "Hello",
        ", ",
        "this ",
        "is ",
        "a ",
        "streamed ",
        "response ",
        "from ",
        "Gemini!"
    ]

    for token in tokens:
        await asyncio.sleep(0.4)
        yield token


async def event_generator():
    async for token in fake_gemini_stream():
        yield f"data: {json.dumps({'token': token})}\n\n"

    yield "data: [DONE]\n\n"


@app.get("/stream")
async def stream():
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )