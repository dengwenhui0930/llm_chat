import math

import jieba

from app.services.knowledge_service import get_all_chunks, is_idf_dirty, mark_idf_clean

TOP_K = 3

# IDF and tokenization cache — rebuilt only when knowledge store changes
_idf_cache: dict[str, float] | None = None
_tokens_cache: list[list[str]] | None = None


def _tokenize(text: str) -> list[str]:
    return [w for w in jieba.cut(text) if w.strip()]


def _ensure_cache(all_chunks: list[str]) -> tuple[dict[str, float], list[list[str]]]:
    global _idf_cache, _tokens_cache
    if _idf_cache is None or is_idf_dirty():
        _tokens_cache = [_tokenize(chunk) for chunk in all_chunks]
        n = len(all_chunks)
        df: dict[str, int] = {}
        for tokens in _tokens_cache:
            for word in set(tokens):
                df[word] = df.get(word, 0) + 1
        _idf_cache = {word: math.log((n + 1) / (count + 1)) + 1 for word, count in df.items()}
        mark_idf_clean()
    return _idf_cache, _tokens_cache


def _tfidf_score(query_tokens: list[str], chunk_tokens: list[str], idf: dict[str, float]) -> float:
    if not chunk_tokens:
        return 0.0
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
    all_chunks = get_all_chunks()
    if not all_chunks:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    idf, tokens_cache = _ensure_cache(all_chunks)

    scored = [
        (chunk, _tfidf_score(query_tokens, tokens_cache[i], idf))
        for i, chunk in enumerate(all_chunks)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, score in scored[:top_k] if score > 0]
