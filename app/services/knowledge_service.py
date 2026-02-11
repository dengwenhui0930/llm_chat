import io

from docx import Document

# In-memory knowledge storage: filename -> list of chunk strings
_knowledge_store: dict[str, list[str]] = {}

ALLOWED_EXTENSIONS = {".txt", ".md", ".docx"}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def extract_docx_text(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def split_chunks(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return chunks


def store_chunks(filename: str, chunks: list[str]) -> None:
    _knowledge_store[filename] = chunks


def list_files() -> list[dict]:
    return [
        {"filename": fn, "chunks": len(chunks)}
        for fn, chunks in _knowledge_store.items()
    ]


def delete_file(filename: str) -> bool:
    if filename in _knowledge_store:
        del _knowledge_store[filename]
        return True
    return False
