from pathlib import Path
import re
import pytest

from src.data.documents import load_knowledge_base
from src.data.facts import (
    DRIFT_TYPES, Fact, FactValue, build_fact_queries, drift_type, fact_value, load_facts, memory_status,
    select_facts, split_facts,
)
from src.retrieval.retriever import Retriever

CORPUS = Path(__file__).parents[1] / "data" / "documentation"


def _snapshots():
    return tuple(load_knowledge_base(CORPUS, name) for name in ("kb_0", "kb_a", "kb_b"))


def _make_queries():
    return build_fact_queries(load_facts(CORPUS / "facts.yaml"), *_snapshots())


def test_document_loader_preserves_chunk_metadata():
    chunks = load_knowledge_base(CORPUS, "kb_a")
    assert chunks
    assert {"document_id", "source", "title", "chunk_id", "text"} <= set(chunks[0].__dataclass_fields__)
    assert all(chunk.text for chunk in chunks)


def test_knowledge_base_selection_is_explicit():
    kb_0, kb_a, kb_b = _snapshots()
    assert kb_0 and kb_a and kb_b
    with pytest.raises(ValueError):
        load_knowledge_base(CORPUS, "kb_c")


def test_corpus_has_no_formatting_artifacts():
    for path in CORPUS.glob("kb_*/*.md"):
        text = path.read_text(encoding="utf-8")
        assert "¶" not in text and "’" not in text, path


def test_every_fact_is_stated_at_most_once_on_its_page():
    kb_0, kb_a, kb_b = _snapshots()
    for fact in load_facts(CORPUS / "facts.yaml"):
        for chunks in (kb_0, kb_a, kb_b):
            fact_value(fact, chunks)  # raises if the owning page states the slot more than once


def test_front_matter_dates_are_parsed_and_stale_faq_is_older():
    _, kb_a, kb_b = _snapshots()
    assert all(chunk.updated for chunk in kb_a + kb_b)
    assert not any(chunk.text.startswith("---") or "updated:" in chunk.text for chunk in kb_a + kb_b)
    dates = {chunk.source: chunk.updated for chunk in kb_b}
    assert dates["faq.md"] < dates["payments.md"] and dates["faq.md"] < dates["users.md"]


def test_drift_type_is_computed_from_values():
    assert drift_type(FactValue("a", ()), FactValue("a", ())) == "unchanged"
    assert drift_type(FactValue("a", ()), FactValue("b", ())) == "modified"
    assert drift_type(FactValue("a", ()), FactValue("b", ("a",))) == "contradicted"
    assert drift_type(FactValue(None, ()), FactValue("b", ())) == "added"
    assert drift_type(FactValue("a", ()), FactValue(None, ())) == "removed"
    assert drift_type(FactValue(None, ()), FactValue(None, ())) == "absent"


def test_formatting_only_change_is_not_drift(tmp_path):
    fact = Fact("f", "doc.md", r"valid for (\d+ hours)", ("How long?",))
    for name, body in (("kb_a", "Keys are valid for 24 hours."), ("kb_b", "Keys are  valid for 24 hours!")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "doc.md").write_text(f"# Doc\n\n{body}", encoding="utf-8")
    before, after = (fact_value(fact, load_knowledge_base(tmp_path, name)) for name in ("kb_a", "kb_b"))
    assert drift_type(before, after) == "unchanged"


def test_memory_status():
    assert memory_status("24 hours", "24 hours") == "correct"
    assert memory_status("24 hours", "48 hours") == "stale"
    assert memory_status(None, "48 hours") == "unknown"
    assert memory_status(None, None) == "unknown"
    assert memory_status("24 hours", None) == "stale"


def test_kb_b_covers_every_drift_type_and_queries_changed_facts():
    _, queries_b = _make_queries()
    assert {query.drift_type for query in queries_b} == set(DRIFT_TYPES)
    assert all(query.affected_by_drift == (query.drift_type not in {"unchanged", "absent"}) for query in queries_b)
    assert all((query.gold_answer is None) == (query.drift_type in {"removed", "absent"}) for query in queries_b)


def test_kb_a_does_not_leak_kb_b_facts():
    queries_a, queries_b = _make_queries()
    added = {query.fact_id for query in queries_b if query.drift_type == "added"}
    assert added and not added & {query.fact_id for query in queries_a}
    assert all(query.drift_type == "baseline" and not query.affected_by_drift for query in queries_a)
    absent = {query.fact_id for query in queries_b if query.drift_type == "absent"}
    assert {query.fact_id for query in queries_a if query.gold_answer is None} == absent


def test_kb_a_memory_is_non_trivial():
    """Closed-book answering must sometimes fail on KB-A, so retrieval is sometimes needed."""
    queries_a, _ = _make_queries()
    statuses = {query.memory_status for query in queries_a}
    assert statuses == {"correct", "stale", "unknown"}


def test_memory_answer_is_shared_across_kbs():
    queries_a, queries_b = _make_queries()
    by_id = {query.query_id: query for query in queries_b}
    assert all(by_id[query.query_id].memory_answer == query.memory_answer for query in queries_a)


def test_questions_do_not_contain_answer_values():
    kb_0, kb_a, kb_b = _snapshots()
    for fact in load_facts(CORPUS / "facts.yaml"):
        values = set()
        for chunks in (kb_0, kb_a, kb_b):
            found = fact_value(fact, chunks)
            values |= {found.value, *found.conflicting} - {None}
        for question in fact.questions:
            assert not any(re.search(rf"\b{re.escape(v)}\b", question.lower()) for v in values), fact.fact_id


def test_retrieval_returns_document_chunk_metadata():
    chunks = load_knowledge_base(CORPUS, "kb_a")
    result = Retriever(chunks).search("How long does an idempotency key remain valid?", 1)[0]
    assert result.fact.source in {"payments.md", "faq.md"}
    assert isinstance(result.score, float)


def test_fact_split_is_disjoint_stratified_and_keeps_paraphrases_together():
    queries_a, queries_b = _make_queries()
    train, test = split_facts(queries_b, test_fraction=0.4, seed=0)
    assert train and test and not train & test
    assert train | test == {query.fact_id for query in queries_b}
    for split in (train, test):
        assert {q.drift_type for q in select_facts(queries_b, split)} == set(DRIFT_TYPES)
    # Every question of a fact lands in the same split (no paraphrase leakage).
    for fact_id in train | test:
        in_test = {q.fact_id in test for q in queries_b if q.fact_id == fact_id}
        assert len(in_test) == 1
    # The KB-A test questions are a subset of the KB-B test questions (same held-out facts).
    assert ({q.query_id for q in select_facts(queries_a, test)}
            <= {q.query_id for q in select_facts(queries_b, test)})


def test_fact_split_is_seeded():
    _, queries_b = _make_queries()
    assert split_facts(queries_b, 0.4, 0) == split_facts(queries_b, 0.4, 0)
    assert any(split_facts(queries_b, 0.4, 0)[1] != split_facts(queries_b, 0.4, seed)[1] for seed in range(1, 5))
