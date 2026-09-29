"""Auditable kb_0 -> KB-A -> KB-B trajectory of every fact.

`python -m src.data.audit` validates the dataset and writes
data/documentation/fact_trajectories.csv (checked in; a test keeps it current).
"""
from pathlib import Path
import csv

from .documents import load_knowledge_base
from .facts import build_fact_queries, drift_type, extract_values, fact_value, load_facts, memory_status
from .memory import NOT_IN_KB_A, assign_memory, load_memory_spec

COLUMNS = ("fact_id", "source", "drift_event", "drift_type", "kb_0_memory_status", "kb_0_memory_rule",
           "kb_0_value", "kb_a_value", "kb_b_value", "kb_b_conflicting_values", "memory_status_on_kb_b",
           "kb_a_questions", "kb_b_questions")


def fact_trajectories(facts, kb_0, kb_a, kb_b, assignment) -> list[dict]:
    rows = []
    for fact in facts:
        memory = fact_value(fact, kb_0).value
        a, b = fact_value(fact, kb_a), fact_value(fact, kb_b)
        change = drift_type(a, b)
        rows.append({
            "fact_id": fact.fact_id, "source": fact.source, "drift_event": fact.event, "drift_type": change,
            "kb_0_memory_status": memory_status(memory, a.value), "kb_0_memory_rule": assignment[fact.fact_id].reason,
            "kb_0_value": memory or "", "kb_a_value": a.value or "", "kb_b_value": b.value or "",
            "kb_b_conflicting_values": "|".join(b.conflicting),
            "memory_status_on_kb_b": memory_status(memory, b.value),
            "kb_a_questions": len(fact.questions) if a.value is not None or change == "absent" else 0,
            "kb_b_questions": len(fact.questions),
        })
    return rows


def validate(facts, kb_0, kb_a, kb_b, assignment) -> list[str]:
    """Internal-consistency problems of the trajectories and queries (empty = valid)."""
    problems = []
    rows = {row["fact_id"]: row for row in fact_trajectories(facts, kb_0, kb_a, kb_b, assignment)}
    queries_a, queries_b = build_fact_queries(facts, kb_0, kb_a, kb_b)
    dates = {(chunk.knowledge_base, chunk.source): chunk.updated for chunk in kb_a + kb_b}
    for fact in facts:
        row, fid = rows[fact.fact_id], fact.fact_id
        memory = fact_value(fact, kb_0).value
        a, b = fact_value(fact, kb_a), fact_value(fact, kb_b)
        change = row["drift_type"]
        expected = {
            "unchanged": a.value is not None and a.value == b.value and not b.conflicting,
            "modified": None not in (a.value, b.value) and a.value != b.value and not b.conflicting,
            # A stale page restates the old KB-A value and is older than the owning page.
            "contradicted": None not in (a.value, b.value) and a.value in b.conflicting and all(
                dates[("kb_b", chunk.source)] < dates[("kb_b", fact.source)] for chunk in kb_b
                if chunk.source != fact.source and set(b.conflicting) & set(extract_values(fact.pattern, chunk.text))),
            "added": a.value is None and b.value is not None,
            "removed": a.value is not None and b.value is None,
            "absent": a.value is None and b.value is None and not a.conflicting and not b.conflicting,
        }[change]
        if not expected:
            problems.append(f"{fid}: inconsistent {change} trajectory {a} -> {b}")
        if a.conflicting:
            problems.append(f"{fid}: KB-A contradicts itself {a.conflicting}")
        status = assignment[fid].status
        if row["kb_0_memory_status"] != status:
            problems.append(f"{fid}: kb_0 memory is {row['kb_0_memory_status']}, rule assigned {status}")
        if (a.value is None) != (assignment[fid].reason == NOT_IN_KB_A):
            problems.append(f"{fid}: memory rule reason disagrees with KB-A coverage")
        if memory is not None and a.value != b.value and memory == b.value:
            problems.append(f"{fid}: memory holds the post-drift KB-B value")
        for kb_label, queries, gold, include in (("kb_a", queries_a, a.value, a.value is not None or change == "absent"),
                                                 ("kb_b", queries_b, b.value, True)):
            own = [q for q in queries if q.fact_id == fid]
            if len(own) != (len(fact.questions) if include else 0):
                problems.append(f"{fid}: {kb_label} has {len(own)} questions")
            for query in own:
                # Grading uses the owning page's value, never generator memory.
                if query.gold_answer != gold or query.source != fact.source:
                    problems.append(f"{query.query_id}: gold {query.gold_answer!r} is not the owning-page value")
                if query.memory_answer != memory or query.memory_status != memory_status(memory, gold):
                    problems.append(f"{query.query_id}: memory fields disagree with kb_0")
                if query.drift_event != fact.event:
                    problems.append(f"{query.query_id}: drift event {query.drift_event!r} != {fact.event!r}")
    known = {fact.fact_id for fact in facts}
    problems += [f"{q.query_id}: no owning fact" for q in queries_a + queries_b if q.fact_id not in known]
    return problems


def main():
    root = Path(__file__).resolve().parents[2] / "data" / "documentation"
    facts = load_facts(root / "facts.yaml")
    kb_0, kb_a, kb_b = (load_knowledge_base(root, name) for name in ("kb_0", "kb_a", "kb_b"))
    assignment = assign_memory(facts, kb_a, kb_b, load_memory_spec(root / "memory.yaml").seed)
    problems = validate(facts, kb_0, kb_a, kb_b, assignment)
    for problem in problems:
        print("PROBLEM", problem)
    rows = fact_trajectories(facts, kb_0, kb_a, kb_b, assignment)
    with open(root / "fact_trajectories.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} facts written; {len(problems)} problems")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
