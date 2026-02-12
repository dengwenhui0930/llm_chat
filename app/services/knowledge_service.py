import io

from docx import Document

# In-memory knowledge storage: filename -> list of chunk strings
_knowledge_store: dict[str, list[str]] = {}

ALLOWED_EXTENSIONS = {".txt", ".md", ".docx"}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Flag for retriever to know when to rebuild IDF cache
_idf_cache_dirty = True


def extract_docx_text(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def split_chunks(text: str) -> list[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def store_chunks(filename: str, chunks: list[str]) -> None:
    global _idf_cache_dirty
    _knowledge_store[filename] = chunks
    _idf_cache_dirty = True


def get_all_chunks() -> list[str]:
    all_chunks: list[str] = []
    for chunks in _knowledge_store.values():
        all_chunks.extend(chunks)
    return all_chunks


def list_files() -> list[dict]:
    return [
        {"filename": fn, "chunks": len(chunks)}
        for fn, chunks in _knowledge_store.items()
    ]


def delete_file(filename: str) -> bool:
    global _idf_cache_dirty
    if filename in _knowledge_store:
        del _knowledge_store[filename]
        _idf_cache_dirty = True
        return True
    return False


def is_idf_dirty() -> bool:
    return _idf_cache_dirty


def mark_idf_clean() -> None:
    global _idf_cache_dirty
    _idf_cache_dirty = False


def clear_all() -> None:
    """Clear all knowledge data and invalidate cache. Primarily for testing."""
    global _idf_cache_dirty
    _knowledge_store.clear()
    _idf_cache_dirty = True
