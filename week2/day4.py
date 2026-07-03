"""
Profile -> optimize -> benchmark demo.
Run directly for timing/memory comparison: python3 profile_demo.py
Run with kernprof for line-by-line profiling:
    kernprof -l -v profile_demo.py --profile-original
"""
import random, time, tracemalloc, sys
from functools import lru_cache
from collections import defaultdict

# ---------- ORIGINAL (slow) ----------
def fib_slow(n):
    """Naive recursive fibonacci - no caching."""
    if n < 2:
        return n
    return fib_slow(n - 1) + fib_slow(n - 2)

def analyze_original(records):
    """O(n^2) dup check + uncached recursive fib. Bottleneck target."""
    scores = [fib_slow(r["value"] % 22) for r in records]

    duplicates = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if records[i]["email"] == records[j]["email"]:
                duplicates.append(records[i]["email"])

    return {"total_score": sum(scores), "duplicates": duplicates}

# ---------- OPTIMIZED ----------
@lru_cache(maxsize=None)
def fib_fast(n):
    """Memoized fibonacci - each n computed once."""
    if n < 2:
        return n
    return fib_fast(n - 1) + fib_fast(n - 2)

def analyze_optimized(records):
    """
    Same output as original:
    - fib results cached (huge win, only 22 distinct n values)
    - dup check is O(n): count emails, then rebuild the exact
      C(k,2) pairwise duplicate list the nested loop would produce
    """
    scores = [fib_fast(r["value"] % 22) for r in records]

    counts = defaultdict(int)
    for r in records:
        counts[r["email"]] += 1

    duplicates = []
    for email, k in counts.items():
        duplicates.extend([email] * (k * (k - 1) // 2))

    return {"total_score": sum(scores), "duplicates": duplicates}

# ---------- profiling hook (only active under kernprof) ----------
try:
    profile          # injected by kernprof into builtins
except NameError:
    def profile(f):
        return f
analyze_original = profile(analyze_original)

# ---------- helpers ----------
def build_records():
    random.seed(1)
    emails = [f"user{i}@test.com" for i in range(200)]
    return [{"email": random.choice(emails), "value": random.randint(0, 40)}
            for _ in range(1500)]

def bench(fn, label):
    records = build_records()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(records)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{label:10s} time={t1 - t0:8.4f}s  peak_mem={peak / 1024:8.1f} KB  "
          f"score={result['total_score']}  dup_count={len(result['duplicates'])}")

if __name__ == "__main__":
    if "--profile-original" in sys.argv:
        analyze_original(build_records())   # for use under kernprof -l -v
    else:
        bench(analyze_original, "ORIGINAL")
        bench(analyze_optimized, "OPTIMIZED")