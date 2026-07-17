from fastapi import APIRouter, HTTPException, status

from app.schemas.document import (
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
)

from app.services.document_service import (
    get_all_documents,
    get_document,
    create_document,
    update_document,
    delete_document,
)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.get(
    "/",
    response_model=list[DocumentResponse]
)
def list_documents():
    return get_all_documents()


@router.get(
    "/{doc_id}",
    response_model=DocumentResponse
)
def read_document(doc_id: int):
    doc = get_document(doc_id)

    if not doc:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    return doc


@router.post(
    "/",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def add_document(document: DocumentCreate):
    return create_document(document)


@router.put(
    "/{doc_id}",
    response_model=DocumentResponse,
)
def edit_document(
    doc_id: int,
    document: DocumentUpdate,
):
    updated = update_document(
        doc_id,
        document,
    )

    if not updated:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    return updated


@router.delete(
    "/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def remove_document(doc_id: int):
    deleted = delete_document(doc_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )