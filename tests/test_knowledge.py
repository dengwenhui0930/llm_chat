import pytest
from app.services.knowledge_service import split_chunks, _knowledge_store, store_chunks


class TestSplitChunks:
    def test_short_text_single_chunk(self):
        chunks = split_chunks("hello")
        assert chunks == ["hello"]

    def test_exact_chunk_size(self):
        text = "a" * 500
        chunks = split_chunks(text)
        # 0-500 then overlap starts at 450, producing a small tail chunk
        assert len(chunks) == 2
        assert chunks[0] == text
        assert len(chunks[1]) == 50

    def test_overlap(self):
        text = "a" * 600
        chunks = split_chunks(text)
        assert len(chunks) == 2
        # First chunk: 0-500, second chunk: 450-600
        assert len(chunks[0]) == 500
        assert len(chunks[1]) == 150
        # Overlap region should match
        assert chunks[0][450:] == chunks[1][:50]

    def test_multiple_chunks(self):
        text = "x" * 1400
        chunks = split_chunks(text)
        # 0-500, 450-950, 900-1400, 1350-1400
        assert len(chunks) == 4

    def test_empty_text(self):
        chunks = split_chunks("")
        assert chunks == []


class TestStoreChunks:
    def test_store_and_retrieve(self):
        _knowledge_store.clear()
        store_chunks("test.txt", ["chunk1", "chunk2"])
        assert _knowledge_store["test.txt"] == ["chunk1", "chunk2"]
