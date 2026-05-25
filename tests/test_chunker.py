from rag.chunker import chunk_text

def test_chunk_text():
    chunks = chunk_text("hello world", target=5, overlap=1)
    assert len(chunks) > 0
