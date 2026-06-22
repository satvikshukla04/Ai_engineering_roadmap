import time
import random
import functools
import pytest


def timeit(func):
    """Measures and prints how long a function takes to run."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            print(f"[timeit] {func.__name__} took {time.perf_counter() - start:.4f}s")
    return wrapper


def retry(max_attempts=3, delay=1.0, exceptions=(IOError,)):
    """Retries a function if it raises a specified exception."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(f"[retry] {func.__name__} attempt {attempt}/{max_attempts} failed: {e}")
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator


@timeit
@retry(max_attempts=3, delay=0.1, exceptions=(IOError,))
def fetch_user(user_id: int):
    """Simulates a network call that randomly fails."""
    if random.random() < 0.6:
        raise IOError(f"Timeout fetching user {user_id}")
    return {"user_id": user_id, "name": "Alice"}


@timeit
@retry(max_attempts=3, delay=0.1, exceptions=(IOError,))
def fetch_orders(user_id: int):
    """Simulates a network call that randomly fails."""
    if random.random() < 0.6:
        raise IOError(f"Timeout fetching orders for user {user_id}")
    return {"user_id": user_id, "orders": [101, 102, 103]}



def test_retry_succeeds_eventually():
    """retry should return the value once the function stops failing."""
    attempts = []

    @retry(max_attempts=3, delay=0, exceptions=(IOError,))
    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise IOError("not yet")
        return "ok"

    assert flaky() == "ok"
    assert len(attempts) == 3


def test_retry_raises_after_max_attempts():
    """retry should re-raise after exhausting all attempts."""
    @retry(max_attempts=2, delay=0, exceptions=(ValueError,))
    def always_fails():
        raise ValueError("permanent error")

    with pytest.raises(ValueError, match="permanent error"):
        always_fails()


def test_timeit_prints_time_even_on_failure(capsys):
    """timeit should print timing even when an exception is raised."""
    @timeit
    @retry(max_attempts=2, delay=0, exceptions=(IOError,))
    def always_fails():
        raise IOError("fail")

    with pytest.raises(IOError):
        always_fails()

    assert "took" in capsys.readouterr().out


def test_metadata_is_preserved():
    """@functools.wraps should keep the original function name and docstring."""
    @timeit
    @retry(max_attempts=2, delay=0)
    def my_func():
        """My docstring."""
        pass

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "My docstring."



if __name__ == "__main__":
    random.seed(42)

    for label, fn, arg in [
        ("fetch_user(1)",   fetch_user,   1),
        ("fetch_orders(1)", fetch_orders, 1),
    ]:
        print(f"\n--- {label} ---")
        try:
            print("Result:", fn(arg))
        except IOError as e:
            print("Failed after all retries:", e)
