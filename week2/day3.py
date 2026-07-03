
import asyncio, json, logging, random, sys, time
from datetime import datetime, timezone
 
# ---------- Exception hierarchy ----------
class FetchError(Exception):
    """Base for all fetch-related errors."""
    def __init__(self, msg, url=None):
        super().__init__(msg)
        self.url = url
 
class FetchTimeoutError(FetchError):
    """Raised when a request exceeds the allowed timeout."""
 
class FetchConnectionError(FetchError):
    """Raised on simulated connection/network failure."""
 
class FetchHTTPError(FetchError):
    """Raised when the server returns a bad status code."""
    def __init__(self, msg, url=None, status_code=None):
        super().__init__(msg, url)
        self.status_code = status_code
 
class FetchValidationError(FetchError):
    """Raised when a response fails schema/shape checks."""
 
# ---------- Logging setup ----------
class JSONFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k in ("url", "latency", "status", "attempt", "error_type"):
            if hasattr(record, k):
                payload[k] = getattr(record, k)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)
 
class ColorFormatter(logging.Formatter):
    COLORS = {"DEBUG": "\033[36m", "INFO": "\033[32m",
              "WARNING": "\033[33m", "ERROR": "\033[31m", "CRITICAL": "\033[41m"}
    RESET = "\033[0m"
 
    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        base = f"{color}[{record.levelname:<8}]{self.RESET} {record.getMessage()}"
        extras = " ".join(f"{k}={getattr(record, k)}" for k in
                           ("url", "latency", "attempt") if hasattr(record, k))
        return f"{base} {extras}".rstrip()
 
def setup_logging(logfile="fetch_log.json") -> logging.Logger:
    logger = logging.getLogger("async_fetch")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
 
    fh = logging.FileHandler(logfile)
    fh.setFormatter(JSONFormatter())
    fh.setLevel(logging.DEBUG)
 
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(ColorFormatter())
    ch.setLevel(logging.INFO)
 
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
 
log = setup_logging()
 
# ---------- Fetch logic ----------
URLS = [f"https://jsonplaceholder.typicode.com/posts/{i}" for i in range(1, 6)]
TIMEOUT = 5
FAIL_RATE = 0.15
 
async def fetch(url: str) -> dict:
    delay = random.uniform(0.3, 1.2)
    roll = random.random()
    if roll < FAIL_RATE / 2:
        raise FetchConnectionError("simulated connection reset", url=url)
    if roll < FAIL_RATE:
        delay = TIMEOUT + 1
    await asyncio.sleep(delay)
    if random.random() < 0.05:
        raise FetchHTTPError("server returned bad status", url=url, status_code=500)
    return {"url": url, "status": "ok", "latency": round(delay, 3)}
 
async def fetch_with_timeout(url: str) -> dict:
    try:
        async with asyncio.timeout(TIMEOUT):
            data = await fetch(url)
            if "latency" not in data:
                raise FetchValidationError("missing latency field", url=url)
            log.info("fetch succeeded", extra={"url": url, "latency": data["latency"]})
            return data
    except TimeoutError:
        err = FetchTimeoutError(f"request exceeded {TIMEOUT}s", url=url)
        log.error("fetch timed out", extra={"url": url}, exc_info=err)
        raise err
    except FetchConnectionError as e:
        log.warning("connection error", extra={"url": url, "error_type": "connection"})
        raise e
    except FetchHTTPError as e:
        log.error("http error", extra={"url": url, "status": e.status_code})
        raise e
 
async def concurrent_fetch() -> list:
    tasks = [fetch_with_timeout(u) for u in URLS]
    return await asyncio.gather(*tasks, return_exceptions=True)
 
def print_summary(results: list) -> None:
    success = sum(1 for r in results if isinstance(r, dict))
    failed = len(results) - success
    log.info(f"summary: {success}/{len(results)} ok, {failed}/{len(results)} failed")
 
async def main():
    log.info("starting async fetch run")
    start = time.perf_counter()
    results = await concurrent_fetch()
    elapsed = time.perf_counter() - start
    print_summary(results)
    log.info(f"total time {elapsed:.3f}s")
 
if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
 
