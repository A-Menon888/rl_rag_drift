"""Dataset invariants: chunking, correlated drift events, kb_0 memory, fact trajectories."""
import csv
from collections import Counter
from pathlib import Path
import re

import pytest

from src.data.audit import COLUMNS, fact_trajectories, validate
from src.data.documents import load_documents, load_knowledge_base
from src.data.facts import (MEMORY_STATUSES, build_fact_queries, describe_queries, drift_type, extract_values,
                            fact_value, load_drift_events, load_facts, load_split_groups, split_facts)
from src.data.memory import NOT_IN_KB_A, MemoryAssignment, assign_memory, build, build_memory_pages, load_memory_spec
from src.environment.rl_rag_env import ANSWER, RLRAGEnv
from src.generation.mock import MockAnswerGenerator

CORPUS = Path(__file__).parents[1] / "data" / "documentation"
FACTS = load_facts(CORPUS / "facts.yaml")
SPEC = load_memory_spec(CORPUS / "memory.yaml")


@pytest.fixture(scope="module")
def snapshots():
    return tuple(load_knowledge_base(CORPUS, name) for name in ("kb_0", "kb_a", "kb_b"))


@pytest.fixture(scope="module")
def assignment(snapshots):
    _, kb_a, kb_b = snapshots
    return assign_memory(FACTS, kb_a, kb_b, SPEC.seed)


# ── Chunking ────────────────────────────────────────────────────────────────

def test_lead_in_stays_with_the_block_it_introduces_but_not_across_headings(tmp_path):
    (tmp_path / "doc.md").write_text(
        "# Doc\n\n## A\n\nCreate a user with:\n\n`POST /v1/users`\n\nTrailing lead-in:\n\n## B\n\nNext section.",
        encoding="utf-8")
    texts = [chunk.text for chunk in load_documents(tmp_path)]
    assert texts == ["Create a user with: `POST /v1/users`", "Trailing lead-in:", "Next section."]


def test_no_retrievable_chunk_is_a_bare_lead_in(snapshots):
    for chunks in snapshots[1:]:
        assert not [chunk.text for chunk in chunks if chunk.text.endswith(":")]


def test_every_documented_fact_is_stated_in_exactly_one_chunk_of_its_page(snapshots):
    """No fact is favoured by redundant answer chunks on its owning page."""
    for chunks in snapshots[1:]:
        for fact in FACTS:
            stating = [c for c in chunks if c.source == fact.source and extract_values(fact.pattern, c.text)]
            assert len(stating) == (fact_value(fact, chunks).value is not None), fact.fact_id


# ── Correlated drift events ─────────────────────────────────────────────────

def _members(event):
    return [fact for fact in FACTS if fact.drift_event == event]


def _values(snapshots, fact):
    _, kb_a, kb_b = snapshots
    return fact_value(fact, kb_a), fact_value(fact, kb_b)


def test_declared_drift_events_have_several_members_and_default_to_the_fact():
    declared = load_drift_events(CORPUS / "facts.yaml")
    assert declared and all(len(_members(event)) >= 2 for event in declared)
    assert all(fact.event == (fact.drift_event or fact.fact_id) for fact in FACTS)


def test_api_migration_event_is_exactly_the_version_path_substitutions(snapshots):
    unversioned = lambda value: re.sub(r"/v\d+", "", value)
    migrated = set()
    for fact in FACTS:
        a, b = _values(snapshots, fact)
        if None not in (a.value, b.value) and a.value != b.value and unversioned(a.value) == unversioned(b.value):
            assert "/v2" in b.value
            migrated.add(fact.fact_id)
    assert migrated == {fact.fact_id for fact in _members("api_v2_migration")}


def test_other_events_are_one_quantity_or_one_added_section(snapshots):
    trajectories = {tuple(v.value for v in _values(snapshots, f)) for f in _members("user_list_max_size")}
    assert len(trajectories) == 1 and None not in next(iter(trajectories))
    assert all(drift_type(*_values(snapshots, f)) == "added" for f in _members("webhook_api_added"))


def test_correlated_and_near_duplicate_facts_never_straddle_the_split(snapshots):
    _, queries_b = build_fact_queries(FACTS, *snapshots)
    linked = [[f.fact_id for f in FACTS if f.event == event] for event in {f.event for f in FACTS}]
    linked += list(load_split_groups(CORPUS / "facts.yaml").values())
    for seed in range(5):
        train, test = split_facts(queries_b, 0.4, seed)
        assert train | test == {f.fact_id for f in FACTS} and not train & test
        for members in linked:
            assert set(members) <= train or set(members) <= test, (seed, members)
        units = {f.unit for f in FACTS}
        assert all({f.fact_id for f in FACTS if f.unit == u} <= train or
                   {f.fact_id for f in FACTS if f.unit == u} <= test for u in units)


def test_cross_page_restatements_agree_within_each_snapshot(snapshots):
    """The two fixed inconsistencies: api.md must agree with payments.md in KB-A and KB-B."""
    by_id = {f.fact_id: f for f in FACTS}
    for chunks in snapshots[1:]:
        value = lambda fid: fact_value(by_id[fid], chunks).value
        assert value("create_status") == value("payments.create_status")
        (verb,) = [m for c in chunks if c.source == "api.md"
                   for m in re.findall(r"payment creation endpoint (support|require)s an", c.text)]
        assert verb == value("payments.idempotency_requirement")


def test_describe_queries_counts_drift_events_separately_from_facts(snapshots):
    _, queries_b = build_fact_queries(FACTS, *snapshots)
    summary = describe_queries(queries_b)
    assert summary["drift_events"]["modified"] < summary["facts"]["modified"]
    assert sum(summary["facts"].values()) == len(FACTS)


# ── kb_0 memory ─────────────────────────────────────────────────────────────

def test_every_fact_has_an_explicit_auditable_memory_status(snapshots, assignment):
    kb_0, kb_a, _ = snapshots
    assert set(assignment) == {fact.fact_id for fact in FACTS}
    for fact in FACTS:
        status, reason = assignment[fact.fact_id].status, assignment[fact.fact_id].reason
        assert status in MEMORY_STATUSES and reason
        memory, gold = fact_value(fact, kb_0).value, fact_value(fact, kb_a).value
        assert {"correct": memory == gold and memory is not None,
                "stale": memory is not None and memory != gold,
                "unknown": memory is None}[status], fact.fact_id
        if gold is None:
            assert status == "unknown" and reason == NOT_IN_KB_A


def test_memory_assignment_is_reproducible_and_matches_checked_in_kb_0(snapshots, tmp_path):
    _, kb_a, kb_b = snapshots
    assert assign_memory(FACTS, kb_a, kb_b, SPEC.seed) == assign_memory(FACTS, kb_a, kb_b, SPEC.seed)
    assert any(assign_memory(FACTS, kb_a, kb_b, seed) != assign_memory(FACTS, kb_a, kb_b, SPEC.seed)
               for seed in range(1, 4))
    build(CORPUS, CORPUS / "facts.yaml", CORPUS / "memory.yaml", check=True)  # raises if kb_0 drifted from the rule
    build(CORPUS, CORPUS / "facts.yaml", CORPUS / "memory.yaml", out_dir=tmp_path)
    assert {p.name: p.read_text(encoding="utf-8") for p in tmp_path.glob("*.md")} == \
           {p.name: p.read_text(encoding="utf-8") for p in (CORPUS / "kb_0").glob("*.md")}


def test_memory_statuses_are_balanced_within_each_drift_type(assignment):
    units = {}
    for fact in FACTS:
        if assignment[fact.fact_id].reason != NOT_IN_KB_A:
            units[fact.event] = (assignment[fact.fact_id].reason, assignment[fact.fact_id].status)
    for stratum in {reason for reason, _ in units.values()}:
        counts = Counter(status for reason, status in units.values() if reason == stratum)
        assert max(counts[s] for s in MEMORY_STATUSES) - min(counts[s] for s in MEMORY_STATUSES) <= 1, stratum


def test_correlated_facts_share_one_memory_status(assignment):
    for event in load_drift_events(CORPUS / "facts.yaml"):
        assert len({assignment[fact.fact_id].status for fact in _members(event)}) == 1, event


def test_memory_never_holds_kb_b_only_values(snapshots):
    kb_0, kb_a, kb_b = snapshots
    for fact in FACTS:
        memory, a, b = (fact_value(fact, kb).value for kb in (kb_0, kb_a, kb_b))
        assert memory is None or memory == a or memory != b, fact.fact_id
        if a is None:
            assert memory is None, fact.fact_id  # nothing KB-A lacks (e.g. KB-B additions) is remembered


def test_every_older_value_is_distinct_and_substitutable(snapshots, assignment, tmp_path):
    """Any seed may make any KB-A fact stale, so every older value must work, not only the chosen ones."""
    _, kb_a, kb_b = snapshots
    all_stale = {fid: a if a.reason == NOT_IN_KB_A else MemoryAssignment("stale", a.reason)
                 for fid, a in assignment.items()}
    assert set(SPEC.older_values) == {fid for fid, a in assignment.items() if a.reason != NOT_IN_KB_A}
    for name, text in build_memory_pages(FACTS, all_stale, SPEC, CORPUS / "kb_a").items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    memory_chunks = load_documents(tmp_path)
    for fact in FACTS:
        memory = fact_value(fact, memory_chunks).value
        if all_stale[fact.fact_id].status == "stale":
            assert memory is not None and memory not in {fact_value(fact, kb).value for kb in (kb_a, kb_b)}, \
                fact.fact_id


def test_kb_0_cannot_be_a_retrieval_corpus(snapshots):
    kb_0, kb_a, kb_b = snapshots
    assert {chunk.knowledge_base for chunk in kb_0} == {"kb_0"}
    queries_a, _ = build_fact_queries(FACTS, *snapshots)
    with pytest.raises(ValueError):
        RLRAGEnv(queries_a, kb_0)
    with pytest.raises(ValueError):
        RLRAGEnv(queries_a, kb_a + kb_0[:1])
    env = RLRAGEnv(queries_a, kb_a)
    assert {chunk.knowledge_base for chunk in env.retriever.facts} == {"kb_a"}


def test_generator_uses_memory_only_as_its_closed_book_answer(snapshots):
    kb_0, kb_a, kb_b = snapshots
    queries_a, queries_b = build_fact_queries(FACTS, *snapshots)
    reader = MockAnswerGenerator()
    for query in queries_a + queries_b:
        fact = next(f for f in FACTS if f.fact_id == query.fact_id)
        assert query.memory_answer == fact_value(fact, kb_0).value
        assert reader.generate(query, []) == query.memory_answer
    env = RLRAGEnv(queries_b, kb_b)
    for index, query in enumerate(queries_b):
        env.reset(options={"query_index": index})
        info = env.step(ANSWER)[4]
        assert info["answer"] == query.memory_answer and info["drift_event"] == query.drift_event


# ── Fact trajectories ───────────────────────────────────────────────────────

def test_fact_trajectories_are_internally_consistent(snapshots, assignment):
    assert validate(FACTS, *snapshots, assignment) == []


def test_every_drift_type_follows_its_trajectory(snapshots, assignment):
    rows = fact_trajectories(FACTS, *snapshots, assignment)
    for row in rows:
        a, b, memory = row["kb_a_value"], row["kb_b_value"], row["kb_0_value"]
        kind = row["drift_type"]
        assert {"unchanged": a and a == b, "modified": a and b and a != b,
                "contradicted": a and b and a in row["kb_b_conflicting_values"].split("|"),
                "added": not a and b and not memory, "removed": a and not b,
                "absent": not a and not b and not memory}[kind], row["fact_id"]
        if row["kb_0_memory_status"] == "stale":
            assert memory and memory not in (a, b)  # grading never equals the stale memory value


def test_checked_in_trajectory_csv_is_current(snapshots, assignment):
    with open(CORPUS / "fact_trajectories.csv", encoding="utf-8") as handle:
        stored = list(csv.DictReader(handle))
    current = [{key: str(value) for key, value in row.items()}
               for row in fact_trajectories(FACTS, *snapshots, assignment)]
    assert tuple(stored[0]) == COLUMNS and stored == current
