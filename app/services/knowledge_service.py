# In-memory knowledge storage: filename -> list of chunk strings
_knowledge_store: dict[str, list[str]] = {}

ALLOWED_EXTENSIONS = {".txt", ".md"}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


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
