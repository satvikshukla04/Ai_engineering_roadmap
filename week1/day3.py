import asyncio
import time
import random


URLS = [
    "https://jsonplaceholder.typicode.com/posts/1",
    "https://jsonplaceholder.typicode.com/posts/2",
    "https://jsonplaceholder.typicode.com/posts/3",
    "https://jsonplaceholder.typicode.com/posts/4",
    "https://jsonplaceholder.typicode.com/posts/5",
]

TIMEOUT = 5       
FAIL_RATE = 0.15   



async def fetch(url: str) -> dict:
    """
    Simulate fetching one URL.
    - Random latency between 0.3s and 1.2s
    - Small chance of exceeding the timeout
    """
    delay = random.uniform(0.3, 1.2)

    if random.random() < FAIL_RATE:
        delay = TIMEOUT + 1

    await asyncio.sleep(delay)
    return {"url": url, "status": "ok", "latency": round(delay, 3)}


async def fetch_with_timeout(url: str) -> dict:
    """
    Wrap fetch() with a hard timeout.
    Prints whether the request succeeded or timed out.
    """
    try:
        async with asyncio.timeout(TIMEOUT):
            data = await fetch(url)
            print(f"  [ok]      {url}  ({data['latency']:.3f}s)")
            return data
    except TimeoutError:
        print(f"  [timeout] {url}")
        raise



async def sequential_fetch() -> list:
    """
    Fetch all URLs one at a time.
    Each request waits for the previous one to finish.
    Total time = sum of all individual request times.
    """
    results = []
    for url in URLS:
        try:
            result = await fetch_with_timeout(url)
            results.append(result)
        except TimeoutError:
            results.append({"url": url, "status": "timeout"})
    return results


async def concurrent_fetch() -> list:
    """
    Fetch all URLs at the same time using asyncio.gather.
    Total time ≈ slowest single request (not the sum).
    return_exceptions=True means one failure won't cancel the others.
    """
    tasks = [fetch_with_timeout(url) for url in URLS]
    return await asyncio.gather(*tasks, return_exceptions=True)



def print_summary(results: list) -> None:
    """Print how many requests succeeded vs timed out."""
    success = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "ok")
    failed  = len(results) - success
    print(f"  Succeeded: {success}/{len(results)}  |  Timed out: {failed}/{len(results)}")



async def main():
    print("=" * 50)
    print("  Async Fetch Demo — Sequential vs Concurrent")
    print("=" * 50)

    print("\nSequential fetch (one by one):")
    start = time.perf_counter()
    seq_results = await sequential_fetch()
    seq_time = time.perf_counter() - start
    print_summary(seq_results)
    print(f"  Total time : {seq_time:.3f}s\n")
  
    print("Concurrent fetch (all at once):")
    start = time.perf_counter()
    con_results = await concurrent_fetch()
    con_time = time.perf_counter() - start
    print_summary(con_results)
    print(f"  Total time : {con_time:.3f}s\n")


    print("=" * 50)
    print(f"  Sequential : {seq_time:.3f}s")
    print(f"  Concurrent : {con_time:.3f}s")
    print(f"  Speedup    : {seq_time / con_time:.2f}x faster")
    print("=" * 50)


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())