import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import db.database as database
import db.models  # noqa: F401  (import registers User on Base.metadata before create_all)
from db.database import Base

TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_app.db"


@pytest_asyncio.fixture(autouse=True)
async def use_test_db(monkeypatch):
    """Point the app's DB session at a throwaway SQLite file for each test."""
    test_engine = create_async_engine(TEST_DATABASE_URL)
    test_session_local = async_sessionmaker(
        bind=test_engine, class_=database.AsyncSession, expire_on_commit=False
    )

    monkeypatch.setattr(database, "engine", test_engine)
    monkeypatch.setattr(database, "AsyncSessionLocal", test_session_local)

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()


@pytest.fixture
def app():
    import main
    return main.app


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_register_creates_user(client):
    response = await client.post(
        "/auth/register", json={"username": "alice", "password": "secretpw"}
    )
    assert response.status_code == 200
    assert response.json() == {"message": "User registered successfully"}


@pytest.mark.asyncio
async def test_duplicate_register_rejected(client):
    await client.post("/auth/register", json={"username": "bob", "password": "secretpw"})
    response = await client.post(
        "/auth/register", json={"username": "bob", "password": "secretpw"}
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_login_and_me_roundtrip(client):
    await client.post("/auth/register", json={"username": "carol", "password": "secretpw"})

    login_response = await client.post(
        "/auth/login", data={"username": "carol", "password": "secretpw"}
    )
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]

    me_response = await client.get(
        "/auth/me", headers={"Authorization": f"Bearer {token}"}
    )
    assert me_response.status_code == 200
    body = me_response.json()
    assert body["username"] == "carol"
    assert "id" in body
    assert "created_at" in body


@pytest.mark.asyncio
async def test_login_wrong_password_rejected(client):
    await client.post("/auth/register", json={"username": "dave", "password": "secretpw"})
    response = await client.post(
        "/auth/login", data={"username": "dave", "password": "wrongpw"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_without_token_rejected(client):
    response = await client.get("/auth/me")
    assert response.status_code == 401
