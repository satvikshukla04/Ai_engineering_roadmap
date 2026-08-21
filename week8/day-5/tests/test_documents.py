import pytest


@pytest.mark.asyncio
async def test_create_and_list_documents(client, auth_headers):
    resp = await client.post(
        "/documents",
        json={"title": "Doc 1", "content": "The quick brown fox jumps over the lazy dog. " * 20},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["chunk_count"] > 0

    resp = await client.get("/documents", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 1


@pytest.mark.asyncio
async def test_get_and_delete_document(client, auth_headers):
    resp = await client.post(
        "/documents", json={"title": "Doc 1", "content": "Hello world"}, headers=auth_headers
    )
    doc_id = resp.json()["id"]

    resp = await client.get(f"/documents/{doc_id}", headers=auth_headers)
    assert resp.status_code == 200

    resp = await client.delete(f"/documents/{doc_id}", headers=auth_headers)
    assert resp.status_code == 204

    resp = await client.get(f"/documents/{doc_id}", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_documents_require_auth(client):
    resp = await client.get("/documents")
    assert resp.status_code == 401
