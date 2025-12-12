
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


# Small, fast model. You can change if you want.
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load the model once globally for performance
_model = SentenceTransformer(DEFAULT_MODEL_NAME)


def get_local_embeddings(texts: List[str]) -> np.ndarray:
    """
    Compute embeddings for a list of texts using a local SentenceTransformer model.

    Returns:
        A NumPy array of shape (len(texts), embedding_dim)
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    vectors = _model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit vectors
    )
    return vectors.astype("float32")


def build_embeddings_from_chunks(
    chunks_by_file: Dict[str, List[str]],
) -> Tuple[np.ndarray, List[dict]]:
    """
    From a dict {filename: [chunk1, chunk2, ...]}, build:

    - embeddings matrix of shape (N_chunks, embedding_dim)
    - metadata list of length N_chunks, each item:
        {
            "filename": ...,
            "chunk_index": ...,
            "text": ...,
        }
    """
    all_texts: List[str] = []
    metadata: List[dict] = []

    for filename, chunks in chunks_by_file.items():
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            all_texts.append(chunk)
            metadata.append(
                {
                    "filename": filename,
                    "chunk_index": idx,
                    "text": chunk,
                }
            )

    if not all_texts:
        return np.zeros((0, 0), dtype="float32"), []

    embeddings = get_local_embeddings(all_texts)
    return embeddings, metadata
