
from typing import Dict, List


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Split a long text into chunks of size `chunk_size` with an overlap of `chunk_overlap`.

    Example:
      chunk_size = 800, chunk_overlap = 200
      -> chunk 1: characters 0 to 800
         chunk 2: characters 600 to 1400
         etc.
    """
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - chunk_overlap

    return chunks


def chunk_documents(
    texts_by_file: Dict[str, str],
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> Dict[str, List[str]]:
    """
    Apply `chunk_text` to each document in a dict.

    Parameters
    ----------
    texts_by_file : dict
        {filename: full_text}

    Returns
    -------
    dict
        {filename: [chunk1, chunk2, ...]}
    """
    chunks_by_file: Dict[str, List[str]] = {}

    for filename, text in texts_by_file.items():
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks_by_file[filename] = chunks

    return chunks_by_file
