import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Setup Mock Data
documents = [
    "The cancellation policy states you can get a full refund if you cancel within 24 hours.",
    "To cancel your order, please contact support with your OrderID.",
    "OrderID AB123 has been dispatched and cannot be cancelled.",
    "Python is a versatile programming language used in AI.",
    "Rank_bm25 is a Python library for sparse retrieval.",
    "The return policy is strictly 30 days. No refunds after.",
    "Your OrderID CD456 is currently pending in our system.",
    "Machine learning models require dense vector embeddings."
]

# Test queries designed to test different strengths
queries = [
    "cancellation policy",              # Dense should win (semantic)
    "OrderID AB123",                    # Sparse should win (exact match)
    "refund for my order",              # Hybrid shines (semantic 'refund' + keyword 'order')
    "sparse retrieval Python library",  # Sparse should win
    "AI vector embeddings"              # Dense should win
]

# 2. Initialize Models
print("Loading Embedding Model...")
# Using a small, fast model for demonstration
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedder.encode(documents)

# Tokenize for BM25
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# 3. Define Retrieval Functions
def get_dense_results(query, k=10):
    query_embedding = embedder.encode([query])
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    # Get top k indices sorted descending
    top_indices = np.argsort(scores)[::-1][:k]
    return top_indices.tolist()

def get_bm25_results(query, k=10):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    # Get top k indices sorted descending
    top_indices = np.argsort(scores)[::-1][:k]
    return top_indices.tolist()

def rrf_fusion(dense_ranks, sparse_ranks, k_constant=60, top_n=10):
    """Fuses two ranked lists using Reciprocal Rank Fusion."""
    rrf_scores = {}
    
    # Process Dense list
    for rank, doc_idx in enumerate(dense_ranks):
        if doc_idx not in rrf_scores:
            rrf_scores[doc_idx] = 0.0
        rrf_scores[doc_idx] += 1.0 / (k_constant + (rank + 1))
        
    # Process Sparse list
    for rank, doc_idx in enumerate(sparse_ranks):
        if doc_idx not in rrf_scores:
            rrf_scores[doc_idx] = 0.0
        rrf_scores[doc_idx] += 1.0 / (k_constant + (rank + 1))
        
    # Sort by accumulated RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in sorted_docs][:top_n]

# 4. Evaluation Loop
print("\n--- Running Evaluation ---")
for q in queries:
    dense_top = get_dense_results(q, k=10)
    sparse_top = get_bm25_results(q, k=10)
    hybrid_top = rrf_fusion(dense_top, sparse_top, k_constant=60, top_n=10)
    
    print(f"\nQuery: '{q}'")
    print(f"Top Dense Doc: {dense_top[0]} -> {documents[dense_top[0]]}")
    print(f"Top Sparse Doc: {sparse_top[0]} -> {documents[sparse_top[0]]}")
    print(f"Top Hybrid Doc: {hybrid_top[0]} -> {documents[hybrid_top[0]]}")