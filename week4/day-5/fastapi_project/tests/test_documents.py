from fastapi.testclient import TestClient


def test_create_document(client: TestClient) -> None:
    response = client.post("/documents", json={"title": "Notes", "content": "hello"})
    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Notes"
    assert body["content"] == "hello"
    assert "id" in body


def test_create_document_rejects_empty_title(client: TestClient) -> None:
    response = client.post("/documents", json={"title": "", "content": "x"})
    assert response.status_code == 422


def test_get_document(client: TestClient) -> None:
    created = client.post("/documents", json={"title": "A", "content": "B"}).json()
    response = client.get(f"/documents/{created['id']}")
    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


def test_get_missing_document_returns_404(client: TestClient) -> None:
    response = client.get("/documents/does-not-exist")
    assert response.status_code == 404


def test_list_documents(client: TestClient) -> None:
    client.post("/documents", json={"title": "One", "content": ""})
    client.post("/documents", json={"title": "Two", "content": ""})
    response = client.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2


def test_update_document(client: TestClient) -> None:
    created = client.post("/documents", json={"title": "Old", "content": "x"}).json()
    response = client.patch(f"/documents/{created['id']}", json={"title": "New"})
    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "New"
    assert body["content"] == "x"


def test_delete_document(client: TestClient) -> None:
    created = client.post("/documents", json={"title": "Temp", "content": ""}).json()
    delete_response = client.delete(f"/documents/{created['id']}")
    assert delete_response.status_code == 204

    get_response = client.get(f"/documents/{created['id']}")
    assert get_response.status_code == 404
