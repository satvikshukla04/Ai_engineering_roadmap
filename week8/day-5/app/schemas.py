from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8)


class UserOut(BaseModel):
    id: int
    username: str

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class DocumentCreate(BaseModel):
    title: str
    content: str


class DocumentOut(BaseModel):
    id: int
    title: str
    chunk_count: int

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)


class Citation(BaseModel):
    document_id: int
    document_title: str
    chunk_index: int
    score: float
