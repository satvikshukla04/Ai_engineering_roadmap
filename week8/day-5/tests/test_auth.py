import pytest


@pytest.mark.asyncio
async def test_register_and_login(client):
    resp = await client.post("/auth/register", json={"username": "bob", "password": "password123"})
    assert resp.status_code == 201

    resp = await client.post("/auth/login", data={"username": "bob", "password": "password123"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


@pytest.mark.asyncio
async def test_duplicate_register_rejected(client):
    await client.post("/auth/register", json={"username": "bob", "password": "password123"})
    resp = await client.post("/auth/register", json={"username": "bob", "password": "password123"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    await client.post("/auth/register", json={"username": "bob", "password": "password123"})
    resp = await client.post("/auth/login", data={"username": "bob", "password": "wrong"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me_requires_token(client):
    resp = await client.get("/auth/me")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me_with_token(client, auth_headers):
    resp = await client.get("/auth/me", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["username"] == "alice"
