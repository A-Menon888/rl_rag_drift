"""Fact-level ground truth extracted from documentation snapshots.

A fact is one answer slot on its owning reference page. Its value in any
snapshot is read from the documents with the fact's regex, so drift types are
computed from content rather than hand-labelled, and formatting-only edits
cannot count as drift.
"""
from dataclasses import dataclass
from pathlib import Path
import re

import yaml

from .generator import Query

DRIFT_TYPES = ("unchanged", "modified", "contradicted", "added", "removed", "absent")
MEMORY_STATUSES = ("correct", "stale", "unknown")


@dataclass(frozen=True)
class Fact:
    fact_id: str
    source: str
    pattern: str
    questions: tuple[str, ...]


@dataclass(frozen=True)
class FactValue:
    value: str | None               # stated on the owning page; None = not documented
    conflicting: tuple[str, ...]    # different values stated on other pages


def normalize_answer(value: str | None) -> str | None:
    return None if value is None else " ".join(value.lower().split()).rstrip(".")


def extract_values(pattern: str, text: str) -> list[str]:
    return [normalize_answer(match) for match in re.findall(pattern, text)]


def load_facts(path: str | Path) -> list[Fact]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["facts"]
    facts = [Fact(item["id"], item["source"], item["pattern"], tuple(item["questions"])) for item in raw]
    ids = [fact.fact_id for fact in facts]
    if len(ids) != len(set(ids)):
        raise ValueError("fact ids must be unique")
    for fact in facts:
        if re.compile(fact.pattern).groups != 1:
            raise ValueError(f"{fact.fact_id}: pattern must have exactly one capture group")
        if not fact.questions:
            raise ValueError(f"{fact.fact_id}: at least one question is required")
    return facts


def fact_value(fact: Fact, chunks) -> FactValue:
    """Read a fact from one snapshot. The owning page may state it at most once."""
    own = [value for chunk in chunks if chunk.source == fact.source
           for value in extract_values(fact.pattern, chunk.text)]
    if len(own) > 1:
        raise ValueError(f"{fact.fact_id}: pattern matches {len(own)} times on {fact.source}")
    value = own[0] if own else None
    others = {v for chunk in chunks if chunk.source != fact.source
              for v in extract_values(fact.pattern, chunk.text)}
    return FactValue(value, tuple(sorted(others - {value})))


def drift_type(before: FactValue, after: FactValue) -> str:
    if before.value is None and after.value is None:
        return "absent"  # never documented: unanswerable in both KBs, not drift
    if after.value is None:
        return "removed"
    if before.value is None:
        return "added"
    if after.conflicting:
        return "contradicted"
    return "unchanged" if before.value == after.value else "modified"


def memory_status(memory: str | None, gold: str | None) -> str:
    """Closed-book knowledge vs. gold: unknown (no memory), correct, or stale (wrong value)."""
    if memory is None:
        return "unknown"
    return "correct" if memory == gold else "stale"


def build_fact_queries(facts, kb_0_chunks, kb_a_chunks, kb_b_chunks) -> tuple[list[Query], list[Query]]:
    """Build KB-A and KB-B query sets from the fact manifest.

    KB-A asks about facts documented in KB-A plus never-documented ("absent")
    facts, so nothing about KB-B (e.g. which facts will be added) leaks into
    the pre-drift query set. KB-B asks about every fact: removed and absent
    facts are unanswerable (gold None) and added facts are new questions.
    """
    queries_a, queries_b = [], []
    for fact in facts:
        memory = fact_value(fact, kb_0_chunks).value
        a, b = fact_value(fact, kb_a_chunks), fact_value(fact, kb_b_chunks)
        change = drift_type(a, b)
        for index, question in enumerate(fact.questions):
            leaked = {v for v in (memory, a.value, b.value, *a.conflicting, *b.conflicting) if v}
            if any(re.search(rf"\b{re.escape(v)}\b", question.lower()) for v in leaked):
                raise ValueError(f"{fact.fact_id}: question {index} contains an answer value")
            common = dict(query_id=f"{fact.fact_id}.q{index}", fact_id=fact.fact_id, source=fact.source,
                          text=question, answer_pattern=fact.pattern, memory_answer=memory)
            if a.value is not None or change == "absent":
                queries_a.append(Query(**common, gold_answer=a.value, drift_type="baseline",
                                       memory_status=memory_status(memory, a.value)))
            queries_b.append(Query(**common, gold_answer=b.value, drift_type=change,
                                   memory_status=memory_status(memory, b.value)))
    return queries_a, queries_b


def describe_queries(queries) -> dict[str, dict[str, int]]:
    """Counts by memory status and drift type, for run logs and sanity checks."""
    return {
        "answerable": sum(q.gold_answer is not None for q in queries),
        "memory_status": {s: sum(q.memory_status == s for q in queries) for s in MEMORY_STATUSES},
        "drift_type": {t: sum(q.drift_type == t for q in queries)
                       for t in ("baseline", *DRIFT_TYPES) if any(q.drift_type == t for q in queries)},
    }
