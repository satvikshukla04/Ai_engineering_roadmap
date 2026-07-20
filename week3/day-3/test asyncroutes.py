import time
 
from fastapi.testclient import TestClient
 
from asyncroutes import app
 
 
def test_process_returns_fast_and_returns_job_id():
    with TestClient(app) as client:
        start = time.perf_counter()
        response = client.post("/process", json={"text": "x" * 200})
        elapsed = time.perf_counter() - start
 
    assert response.status_code == 200
    body = response.json()
    assert "job_id" in body
    assert elapsed < 0.05
 
