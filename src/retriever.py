
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Use the same model as for document embeddings
_query_model = SentenceTransformer(DEFAULT_MODEL_NAME)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single user query into a vector (same space as document embeddings).
    """
    vec = _query_model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.astype("float32")


def cosine_similarity_matrix(query_vec: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    """
    Since both query_vec and doc_embeddings are normalized, cosine similarity
    is just the dot product.

    query_vec: shape (d,)
    doc_embeddings: shape (N, d)

    Returns:
        similarities: shape (N,)
    """
    if doc_embeddings.size == 0:
        return np.array([], dtype="float32")

    sims = doc_embeddings @ query_vec  # (N, d) @ (d,) -> (N,)
    return sims


def retrieve_top_k(
    query: str,
    doc_embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most similar chunks for a given query.

    Returns a list of dicts:
        {
            "score": float,
            "filename": str,
            "chunk_index": int,
            "text": str,
        }
    """
    if doc_embeddings.size == 0 or not metadata:
        return []

    query_vec = embed_query(query)
    sims = cosine_similarity_matrix(query_vec, doc_embeddings)

    k = min(k, len(sims))
    top_indices = np.argsort(-sims)[:k]

    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        m = metadata[idx]
        results.append(
            {
                "score": float(sims[idx]),
                "filename": m["filename"],
                "chunk_index": m["chunk_index"],
                "text": m["text"],
            }
        )

    return results
