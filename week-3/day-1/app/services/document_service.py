documents = []


def get_all_documents():
    return documents


def get_document(doc_id: int):
    for doc in documents:
        if doc["id"] == doc_id:
            return doc
    return None


def create_document(data):
    new_doc = {
        "id": len(documents) + 1,
        "title": data.title,
        "content": data.content,
    }

    documents.append(new_doc)
    return new_doc


def update_document(doc_id: int, data):
    for doc in documents:
        if doc["id"] == doc_id:
            doc["title"] = data.title
            doc["content"] = data.content
            return doc

    return None


def delete_document(doc_id: int):
    for doc in documents:
        if doc["id"] == doc_id:
            documents.remove(doc)
            return True

    return False