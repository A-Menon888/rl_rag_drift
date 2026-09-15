from pathlib import Path
import pytest

from src.data.documents import generate_document_queries, load_knowledge_base
from src.retrieval.retriever import Retriever

CORPUS = Path(__file__).parents[1] / "data" / "documentation"


def _make_queries():
    kb_a = load_knowledge_base(CORPUS, "kb_a")
    kb_b = load_knowledge_base(CORPUS, "kb_b")
    return generate_document_queries(kb_a, kb_b)


def test_document_loader_preserves_chunk_metadata():
    chunks = load_knowledge_base(CORPUS, "kb_a")
    assert len(chunks) >= 1
    assert all(chunk.document_id and chunk.source and chunk.title for chunk in chunks)
    assert any("authentication" in chunk.text.lower() for chunk in chunks)


def test_kb_a_queries_have_no_drift():
    queries_a, _ = _make_queries()
    assert all(not q.affected_by_drift for q in queries_a), \
        "KB-A queries should never be marked as drifted"
    assert all(q.memorized_answer == q.current_answer for q in queries_a), \
        "memorized_answer must equal current_answer for all KB-A queries"


def test_kb_b_drift_flag_matches_content():
    queries_a, queries_b = _make_queries()
    for qa, qb in zip(queries_a, queries_b):
        assert qa.query_id == qb.query_id, "Query IDs must be stable across KB-A/KB-B pairs"
        expected_drift = qa.memorized_answer != qb.current_answer
        assert qb.affected_by_drift == expected_drift, (
            f"Query {qb.query_id}: affected_by_drift={qb.affected_by_drift} "
            f"but content differs={expected_drift}"
        )


def test_kb_b_has_some_drifted_and_some_stable_queries():
    _, queries_b = _make_queries()
    drifted = [q for q in queries_b if q.affected_by_drift]
    stable  = [q for q in queries_b if not q.affected_by_drift]
    assert len(drifted) >= 1, "KB-B must have at least one drifted query"
    assert len(stable)  >= 1, "KB-B must have at least one stable (unchanged) query"


def test_query_phrasing_is_seeded_and_preserves_alignment():
    kb_a = load_knowledge_base(CORPUS, "kb_a")
    kb_b = load_knowledge_base(CORPUS, "kb_b")
    queries_a_1, queries_b_1 = generate_document_queries(kb_a, kb_b, seed=1)
    queries_a_2, queries_b_2 = generate_document_queries(kb_a, kb_b, seed=2)
    queries_a_1_repeat, queries_b_1_repeat = generate_document_queries(kb_a, kb_b, seed=1)

    assert [query.text for query in queries_a_1] == [query.text for query in queries_a_1_repeat]
    assert [query.text for query in queries_b_1] == [query.text for query in queries_b_1_repeat]
    assert [query.text for query in queries_a_1] != [query.text for query in queries_a_2]
    for first, second in zip(queries_a_1, queries_b_1):
        assert first.query_id == second.query_id
        assert first.memorized_answer == second.memorized_answer
        assert second.current_answer in {chunk.text for chunk in kb_b}


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


def test_chunk_count_mismatch_is_aligned_without_failing(tmp_path):
    """Inserted or deleted chunks must not abort A/B query generation."""
    (tmp_path / "kb_a").mkdir()
    (tmp_path / "kb_b").mkdir()
    (tmp_path / "kb_a" / "doc.md").write_text("# Title\n\nParagraph one.\n\nParagraph two.")
    (tmp_path / "kb_b" / "doc.md").write_text("# Title\n\nParagraph one.\n\nInserted paragraph.\n\nParagraph two.")
    ka = load_knowledge_base(tmp_path, "kb_a")
    kb = load_knowledge_base(tmp_path, "kb_b")
    queries_a, queries_b = generate_document_queries(ka, kb)
    assert len(queries_a) == len(ka)
    assert len(queries_b) == len(ka)
    assert [query.current_answer for query in queries_b] == [chunk.text for chunk in ka]
