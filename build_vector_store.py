
import os
import glob
import json
import numpy as np

from src.ingestion import extract_texts_from_files
from src.chunker import chunk_documents
from src.embeddings import build_embeddings_from_chunks


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    brochures_dir = os.path.join(base_dir, "data", "brochures")
    vector_store_dir = os.path.join(base_dir, "vector_store")

    if not os.path.isdir(brochures_dir):
        raise FileNotFoundError(
            f"Brochures folder not found: {brochures_dir}. "
            "Create it and add some PDF/TXT files."
        )

    pattern = os.path.join(brochures_dir, "*")
    file_paths = [p for p in glob.glob(pattern) if p.lower().endswith((".pdf", ".txt"))]

    if not file_paths:
        raise FileNotFoundError(
            f"No PDF/TXT files found in {brochures_dir}. "
            "Add some brochures before running this script."
        )

    print(f"Found {len(file_paths)} brochure file(s):")
    for p in file_paths:
        print(" -", os.path.basename(p))

    # 1) Open files
    file_objects = [open(p, "rb") for p in file_paths]

    # 2) Extract text
    print("\nExtracting text from documents...")
    texts_by_file = extract_texts_from_files(file_objects)
    print("Done text extraction.")

    # 3) Chunk documents
    print("Chunking documents...")
    chunks_by_file = chunk_documents(
        texts_by_file,
        chunk_size=800,
        chunk_overlap=200,
    )
    print("Done chunking.")

    # 4) Build embeddings
    print("Computing embeddings...")
    embeddings, metadata = build_embeddings_from_chunks(chunks_by_file)
    print("Done embeddings computation.")

    # 5) Clean filenames
    for m in metadata:
        if "filename" in m and m["filename"]:
            m["filename"] = os.path.basename(m["filename"])

    # 6) Save vector store
    os.makedirs(vector_store_dir, exist_ok=True)

    emb_path = os.path.join(vector_store_dir, "embeddings.npy")
    meta_path = os.path.join(vector_store_dir, "metadata.json")

    np.save(emb_path, embeddings)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nSaved embeddings to: {emb_path}")
    print(f"Saved metadata to:    {meta_path}")
    print("\nVector store built successfully.")


if __name__ == "__main__":
    main()
