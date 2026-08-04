"""End-to-end test of the full flow: register -> login -> create session ->
send a streamed message -> check it landed in history.

Uses a temporary on-disk SQLite file per test run (via monkeypatching the
DATABASE_URL before the app's engine is created) so tests never touch a
developer's real genai_chat_api.db, and don't depend on each other.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    # Point at a fresh SQLite file before importing the app, so init_db()
    # creates tables in an isolated database for this test.
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    # Make sure no real API key sneaks in during tests -> forces MockProvider.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # Import here (after env vars are set) so db.py picks up the test URL.
    from genai_chat_api.main import app

    with TestClient(app) as test_client:
        yield test_client


def _unique_username() -> str:
    return f"intern_{uuid.uuid4().hex[:8]}"


def test_full_chat_flow(client: TestClient) -> None:
    username = _unique_username()
    password = "correct-horse-battery-staple"

    # 1. Register.
    register_resp = client.post(
        "/auth/register", json={"username": username, "password": password}
    )
    assert register_resp.status_code == 201
    assert "access_token" in register_resp.json()

    # 2. Log in via the OAuth2 password flow (form-encoded, not JSON).
    login_resp = client.post(
        "/auth/token", data={"username": username, "password": password}
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 3. System prompts are listed and selectable.
    prompts_resp = client.get("/system-prompts")
    assert prompts_resp.status_code == 200
    prompt_ids = [p["id"] for p in prompts_resp.json()]
    assert "coding_tutor" in prompt_ids

    # 4. Create a session with a chosen system prompt.
    session_resp = client.post(
        "/sessions",
        json={"title": "Test chat", "system_prompt_id": "coding_tutor"},
        headers=headers,
    )
    assert session_resp.status_code == 201
    session_id = session_resp.json()["id"]

    # 5. Send a message and read the streamed SSE response.
    with client.stream(
        "POST",
        f"/sessions/{session_id}/messages",
        json={"content": "What is a closure?"},
        headers=headers,
    ) as stream_resp:
        assert stream_resp.status_code == 200
        assert stream_resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(stream_resp.iter_text())

    assert "event: chunk" in body
    assert "event: done" in body

    # 6. History persisted both the user message and the assistant reply.
    history_resp = client.get(f"/sessions/{session_id}/history", headers=headers)
    assert history_resp.status_code == 200
    history = history_resp.json()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What is a closure?"
    assert history[1]["role"] == "assistant"
    assert "What is a closure?" in history[1]["content"]  # mock echoes it back


def test_cannot_access_another_users_session(client: TestClient) -> None:
    # User A creates a session.
    user_a = _unique_username()
    client.post("/auth/register", json={"username": user_a, "password": "pw"})
    token_a = client.post(
        "/auth/token", data={"username": user_a, "password": "pw"}
    ).json()["access_token"]
    session_id = client.post(
        "/sessions", json={"title": "A's chat"}, headers={"Authorization": f"Bearer {token_a}"}
    ).json()["id"]

    # User B tries to read it.
    user_b = _unique_username()
    client.post("/auth/register", json={"username": user_b, "password": "pw"})
    token_b = client.post(
        "/auth/token", data={"username": user_b, "password": "pw"}
    ).json()["access_token"]

    resp = client.get(
        f"/sessions/{session_id}/history", headers={"Authorization": f"Bearer {token_b}"}
    )
    assert resp.status_code == 404


def test_protected_route_requires_auth(client: TestClient) -> None:
    resp = client.post("/sessions", json={"title": "no auth"})
    assert resp.status_code == 401
