
from typing import Dict
from io import BytesIO
import PyPDF2


def extract_text_from_pdf(file_bytes: BytesIO) -> str:
    """
    Extract text from a PDF file given as bytes.
    Works with files opened via open(..., 'rb') or similar.
    """
    reader = PyPDF2.PdfReader(file_bytes)
    texts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        texts.append(txt)
    return "\n".join(texts)


def extract_text_from_txt(file_bytes: BytesIO, encoding: str = "utf-8") -> str:
    """
    Extract text from a plain text (.txt) file.
    """
    content = file_bytes.read().decode(encoding, errors="ignore")
    return content


def extract_texts_from_files(files) -> Dict[str, str]:
    """
    Extract text from a list of file objects.

    'files' can be:
        - Python file objects from open('file.pdf', 'rb')
        - Streamlit UploadedFile
        - any object with .read() and .name attributes

    Returns a dictionary: {filename: extracted_text}
    """
    results: Dict[str, str] = {}

    for f in files:
        filename = getattr(f, "name", "unknown_file")
        suffix = filename.lower().split(".")[-1]

        if suffix == "pdf":
            text = extract_text_from_pdf(f)
        elif suffix == "txt":
            text = extract_text_from_txt(f)
        else:
            text = ""  # unsupported for now

        results[filename] = text

    return results
