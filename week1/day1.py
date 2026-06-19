from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel


def greet(name: str) -> str:
    return f"Hello, {name}"


def add(a: int, b: int) -> int:
    return a + b


def process_input(value: str | int) -> str:
    return str(value)


T = TypeVar("T")


def first(items: list[T]) -> T:
    if not items:
        raise ValueError("List is empty")
    return items[0]



STUDENT_SCORES: dict[str, int] = {
    "deep1": 95,
    "deep2": 88,
    "deep3": 91,
}


def get_student_names() -> list[str]:
    return list(STUDENT_SCORES.keys())


def get_scores() -> dict[str, int]:
    return STUDENT_SCORES



USERS: dict[int, str] = {
    1: "deep1",
    2: "deep2",
    3: "deep3",
}


class User(BaseModel):
    id: int
    name: str
    email: str


def find_user(user_id: int) -> str | None:
    return USERS.get(user_id)



class Retriever(Protocol):
    def retrieve(self, query: str) -> list[str]: ...


class LocalRetriever:
    def retrieve(self, query: str) -> list[str]:
        return [f"Result for {query}"]


def search_docs(retriever: Retriever, query: str) -> list[str]:
    return retriever.retrieve(query)



if __name__ == "__main__":
    # Greetings & arithmetic
    print(greet("Satvik"))
    print(add(10, 20))

    # Student data
    print(get_student_names())
    print(get_scores())

    # User lookup
    print(find_user(1))
    print(find_user(99))

    # Generic input processing
    print(process_input("hello"))
    print(process_input(100))

    # Pydantic model  ← name changed to Satvik
    user = User(id=1, name="Satvik", email="satvik.shukla2004@gmail.com")
    print(user)

    # Generic first()
    print(first([1, 2, 3]))
    print(first(["a", "b", "c"]))

    # Protocol-based retrieval
    local = LocalRetriever()
    print(search_docs(local, "python"))