from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import status

from pydantic import (
    BaseModel,
    EmailStr,
    field_validator,
    model_validator,
    ConfigDict,
)

app = FastAPI()

# ----------------------------
# Error Schema
# ----------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: str | list[str]
    code: str


# ----------------------------
# Request Model
# ----------------------------

class UserRegistration(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    email: EmailStr
    password: str
    confirm_password: str
    age: int

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value):
        return value.lower()

    @field_validator("password")
    @classmethod
    def validate_password(cls, value):
        if len(value) < 8:
            raise ValueError(
                "Password must be at least 8 characters."
            )

        if not any(ch.isdigit() for ch in value):
            raise ValueError(
                "Password must contain at least one digit."
            )

        return value

    @field_validator("age")
    @classmethod
    def validate_age(cls, value):
        if value < 18:
            raise ValueError(
                "Age must be at least 18."
            )
        return value

    @model_validator(mode="after")
    def validate_password_match(self):
        if self.password != self.confirm_password:
            raise ValueError(
                "Passwords do not match."
            )

        return self


# ----------------------------
# Response Model
# ----------------------------

class UserResponse(BaseModel):
    email: EmailStr
    age: int
    nickname: str | None = None


# ----------------------------
# Routes
# ----------------------------

@app.post(
    "/register",
    response_model=UserResponse,
    response_model_exclude_none=True,
)
def register(user: UserRegistration):

    return UserResponse(
        email=user.email,
        age=user.age,
    )


@app.get("/users/{user_id}")
def get_user(user_id: int):

    if user_id != 1:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return {
        "id": 1,
        "name": "Alice",
    }


@app.get("/crash")
def crash():
    return 1 / 0


# ----------------------------
# Global Exception Handlers
# ----------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP Error",
            detail=exc.detail,
            code=f"HTTP_{exc.status_code}",
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    messages = [
        error["msg"]
        for error in exc.errors()
    ]

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation Error",
            detail=messages,
            code="VALIDATION_ERROR",
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(
    request: Request,
    exc: Exception,
):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="Something went wrong.",
            code="SERVER_ERROR",
        ).model_dump(),
    )