import math

import jieba

from app.services.knowledge_service import _knowledge_store

TOP_K = 3


def _tokenize(text: str) -> list[str]:
    return [w for w in jieba.cut(text) if w.strip()]


def _build_idf(all_chunks: list[str]) -> dict[str, float]:
    """Calculate IDF for each term across all chunks."""
    n = len(all_chunks)
    df: dict[str, int] = {}
    for chunk in all_chunks:
        seen = set(_tokenize(chunk))
        for word in seen:
            df[word] = df.get(word, 0) + 1
    return {word: math.log((n + 1) / (count + 1)) + 1 for word, count in df.items()}


def _tfidf_score(query_tokens: list[str], chunk: str, idf: dict[str, float]) -> float:
    """Score a chunk against query tokens using TF-IDF."""
    chunk_tokens = _tokenize(chunk)
    if not chunk_tokens:
        return 0.0
    # Term frequency in chunk
    tf: dict[str, float] = {}
    for token in chunk_tokens:
        tf[token] = tf.get(token, 0) + 1
    chunk_len = len(chunk_tokens)
    score = 0.0
    for token in query_tokens:
        if token in tf:
            score += (tf[token] / chunk_len) * idf.get(token, 1.0)
    return score


def retrieve(query: str, top_k: int = TOP_K) -> list[str]:
    """Retrieve top-k most relevant chunks from all uploaded knowledge."""
    # Gather all chunks
    all_chunks: list[str] = []
    for chunks in _knowledge_store.values():
        all_chunks.extend(chunks)

    if not all_chunks:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    idf = _build_idf(all_chunks)

    scored = [(chunk, _tfidf_score(query_tokens, chunk, idf)) for chunk in all_chunks]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, score in scored[:top_k] if score > 0]
