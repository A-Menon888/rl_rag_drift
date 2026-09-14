from pathlib import Path

from src.data.documents import generate_document_queries, load_knowledge_base
from src.retrieval.retriever import Retriever


CORPUS = Path(__file__).parents[1] / "data" / "documentation"


def test_document_loader_preserves_chunk_metadata():
    chunks = load_knowledge_base(CORPUS, "kb_a")
    assert len(chunks) >= 1
    assert all(chunk.document_id and chunk.source and chunk.title for chunk in chunks)
    assert any("authentication" in chunk.text.lower() for chunk in chunks)


def test_document_queries_are_grounded_and_direct_path_is_unknown():
    chunks = load_knowledge_base(CORPUS, "kb_a")
    query = generate_document_queries(chunks)[0]
    assert query.answer(0) == chunks[0].text
    assert query.direct_answer == "UNKNOWN"


def test_retrieval_returns_document_chunk_metadata():
    chunks = load_knowledge_base(CORPUS, "kb_a")
    result = Retriever(chunks).search("How long do authentication access tokens last?", 1)[0]
    assert result.fact.source == "authentication.md"
    assert isinstance(result.score, float)


def test_knowledge_base_selection_is_explicit():
    kb_a = load_knowledge_base(CORPUS, "kb_a")
    kb_b = load_knowledge_base(CORPUS, "kb_b")
    assert kb_a and kb_b
    assert {chunk.source for chunk in kb_a} == {chunk.source for chunk in kb_b}
    assert {chunk.text for chunk in kb_a} != {chunk.text for chunk in kb_b}
