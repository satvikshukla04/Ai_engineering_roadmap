from contextlib import contextmanager
from pathlib import Path
import time


class Timer:
    def __enter__(self):
        self.start = time.time()
        print("Timer Started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Timer Stopped: {time.time() - self.start:.2f}s")


@contextmanager
def managed_file(filename: str):
    f = open(filename, "w")
    try:
        yield f
    finally:
        f.close()
        print("File Closed")


with managed_file("sample.txt") as f:
    f.write("Hello from Context Manager")
print(Path("sample.txt").read_text())

with Timer():
    time.sleep(1)


class DatabaseConnection:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self._store: dict = {}
        self.connected = False

    def insert(self, key, value):
        self._check(); self._store[key] = value
        print(f"  [DB] INSERT {key!r} = {value!r}")

    def fetch(self, key):
        self._check()
        print(f"  [DB] FETCH  {key!r} → {self._store.get(key)!r}")
        return self._store.get(key)

    def delete(self, key):
        self._check()
        print(f"  [DB] DELETE {key!r} (was {self._store.pop(key, None)!r})")

    def __enter__(self):
        self.connected = True
        print(f"[DB] Connected to '{self.db_name}'")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connected = False
        if exc_type:
            print(f"[DB] Exception: {exc_type.__name__}: {exc_val}")
        print(f"[DB] Disconnected from '{self.db_name}'")
        return False

    def _check(self):
        if not self.connected:
            raise RuntimeError("No active database connection.")


print("\n--- DB: normal session ---")
with DatabaseConnection("users_db") as db:
    db.insert("user:1", {"name": "Alice", "age": 30})
    db.fetch("user:1")
    db.delete("user:1")

print("\n--- DB: exception session ---")
try:
    with DatabaseConnection("orders_db") as db:
        db.insert("order:1", {"item": "Widget"})
        raise ValueError("Payment timeout")
except ValueError:
    print("[App] Error handled.\n")


class Validated:
    def __set_name__(self, owner, name):
        self.name = name
        self._attr = f"_validated_{name}"

    def __init__(self, *, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def __get__(self, obj, objtype=None):
        return self if obj is None else getattr(obj, self._attr, None)

    def __set__(self, obj, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"'{self.name}' must be numeric, got {type(value).__name__!r}")
        if not (self.min_val <= value <= self.max_val):
            raise ValueError(f"'{self.name}' must be in [{self.min_val}, {self.max_val}], got {value}")
        setattr(obj, self._attr, value)


class Sensor:
    temperature = Validated(min_val=-40.0, max_val=125.0)
    humidity    = Validated(min_val=0.0,   max_val=100.0)

    def __init__(self, temperature, humidity):
        self.temperature = temperature
        self.humidity    = humidity

    def __repr__(self):
        return f"Sensor(temp={self.temperature}°C, humidity={self.humidity}%)"


print("--- Validated: valid ---")
s = Sensor(22.5, 60.0)
print(s)

print("\n--- Validated: out-of-range ---")
try:
    s.humidity = 150.0
except ValueError as e:
    print(f"ValueError → {e}")

print("\n--- Validated: wrong type ---")
try:
    s.temperature = "hot"
except TypeError as e:
    print(f"TypeError  → {e}")

print("\nFinal:", s)
#changes done
