from fastapi import Header, HTTPException, status

DB = {
    "items": {}
}


def get_db():
    """
    Yield dependency.
    Simulates opening and closing a DB session.
    """
    print("Opening DB connection")

    try:
        yield DB

    finally:
        print("Closing DB connection")


def verify_api_key(
    x_api_key: str | None = Header(default=None)
):
    """
    Checks X-API-Key header.
    """

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )

    return x_api_key