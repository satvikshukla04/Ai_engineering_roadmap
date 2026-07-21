import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request, call_next):

        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()

        response = await call_next(request)

        duration = (time.perf_counter() - start) * 1000

        response.headers["X-Request-ID"] = request_id

        client_ip = request.client.host if request.client else "Unknown"

        logging.info(
            "[%s] %s %s | %s | %.2f ms | IP=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration,
            client_ip,
        )

        return response