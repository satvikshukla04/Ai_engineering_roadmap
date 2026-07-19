from typing import Annotated

from fastapi import FastAPI, Depends

from dependencies import (
    get_db,
    verify_api_key
)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI items API! Go to /items to see items, or /docs for documentation."}

DBDep = Annotated[dict, Depends(get_db)]
APIKeyDep = Annotated[str, Depends(verify_api_key)]


@app.get("/items")
def get_items(
    db: DBDep,
    _: APIKeyDep
):
    return db["items"]


@app.post("/items/{item_id}")
def create_item(
    item_id: int,
    value: str,
    db: DBDep,
    _: APIKeyDep
):
    db["items"][item_id] = value

    return {
        "message": "created",
        "item": value
    }


@app.put("/items/{item_id}")
def update_item(
    item_id: int,
    value: str,
    db: DBDep,
    _: APIKeyDep
):
    db["items"][item_id] = value

    return {
        "message": "updated",
        "item": value
    }


@app.delete("/items/{item_id}")
def delete_item(
    item_id: int,
    db: DBDep,
    _: APIKeyDep
):
    db["items"].pop(item_id, None)

    return {
        "message": "deleted"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)