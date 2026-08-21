import pytest


@pytest.mark.asyncio
async def test_chat_streams_answer_and_citations(client, auth_headers):
    await client.post(
        "/documents",
        json={"title": "Python Basics", "content": "Python is a dynamically typed language. " * 10},
        headers=auth_headers,
    )

    async with client.stream(
        "POST", "/chat", json={"query": "What is Python?"}, headers=auth_headers
    ) as resp:
        assert resp.status_code == 200
        body = ""
        async for chunk in resp.aiter_text():
            body += chunk

    assert "event: token" in body
    assert "event: citations" in body
    assert "event: done" in body


@pytest.mark.asyncio
async def test_chat_requires_auth(client):
    resp = await client.post("/chat", json={"query": "hello"})
    assert resp.status_code == 401
