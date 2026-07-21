from fastapi import APIRouter, Request

from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter(prefix="/auth", tags=["Authentication"])

limiter = Limiter(key_func=get_remote_address)


@router.post("/login")
@limiter.limit("10/minute")
async def login(request: Request):

    return {
        "message": "Login successful",
        "request_id": request.state.request_id,
    }