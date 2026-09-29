"""Construction of kb_0, the frozen generator's closed-book memory snapshot.

kb_0 pages are generated from the KB-A pages: per fact, the statement is kept
(memory correct), its value replaced by a hand-written older value (stale), or
the statement removed (unknown). Which of the three a fact gets is decided by
`assign_memory`, a seeded rule documented in data/documentation/memory.yaml.
The result is checked in; `python -m src.data.memory` rebuilds it and
`python -m src.data.memory --check` verifies the checked-in pages match.

kb_0 is never a retrieval corpus: memory reaches the reader only as
Query.memory_answer (see src/environment/rl_rag_env.py).
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import random
import re

import yaml

from .documents import _split_front_matter, load_documents, load_knowledge_base
from .facts import MEMORY_STATUSES, drift_type, fact_value, load_facts, normalize_answer

NOT_IN_KB_A = "not documented in KB-A"


@dataclass(frozen=True)
class MemorySpec:
    seed: int
    updated: str
    older_values: dict   # fact_id -> str (replaces the capture group) | {"statement": str} (replaces the match)


@dataclass(frozen=True)
class MemoryAssignment:
    status: str   # correct | stale | unknown, relative to the KB-A value
    reason: str   # "allocated (stratum ...)" or NOT_IN_KB_A


def load_memory_spec(path) -> MemorySpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return MemorySpec(int(raw["seed"]), str(raw["updated"]), dict(raw["older_values"]))


def assign_memory(facts, kb_a_chunks, kb_b_chunks, seed: int) -> dict[str, MemoryAssignment]:
    """Seeded, stratified, systematic allocation of memory statuses.

    Units are drift events restricted to facts documented in KB-A (correlated
    facts share one status). Units are ordered by stratum (the set of KB-B
    drift types of their facts), shuffled within each stratum, then dealt
    correct / stale / unknown in turn. Facts absent from KB-A are unknown.
    """
    kb_a = {fact.fact_id: fact_value(fact, kb_a_chunks) for fact in facts}
    change = {fact.fact_id: drift_type(kb_a[fact.fact_id], fact_value(fact, kb_b_chunks)) for fact in facts}
    units: dict[str, list[str]] = {}
    assignment = {}
    for fact in facts:
        if kb_a[fact.fact_id].value is None:
            assignment[fact.fact_id] = MemoryAssignment("unknown", NOT_IN_KB_A)
        else:
            units.setdefault(fact.event, []).append(fact.fact_id)
    strata: dict[str, list[str]] = {}
    for unit, members in units.items():
        strata.setdefault("+".join(sorted({change[m] for m in members})), []).append(unit)
    rng = random.Random(seed)
    position = 0
    for stratum in sorted(strata):
        ordered = sorted(strata[stratum])
        rng.shuffle(ordered)
        for unit in ordered:
            status = MEMORY_STATUSES[position % len(MEMORY_STATUSES)]
            position += 1
            for fact_id in units[unit]:
                assignment[fact_id] = MemoryAssignment(status, f"allocated (stratum {stratum})")
    return {fact.fact_id: assignment[fact.fact_id] for fact in facts}


def _flexible(pattern: str) -> str:
    """Fact patterns match loader text, where line breaks became spaces."""
    return pattern.replace(" ", r"\s+")


def _sentence_span(text: str, start: int, end: int) -> tuple[int, int]:
    """The sentence (or line) containing text[start:end], with one adjacent space."""
    left = start
    while left > 0 and text[left - 1] != "\n" and not (text[left - 1] == " " and text[left - 2:left - 1] == "."):
        left -= 1
    right = end
    while right < len(text) and text[right] != "\n" and not (
            text[right] == "." and (right + 1 == len(text) or text[right + 1].isspace())):
        right += 1
    if right < len(text) and text[right] == ".":
        right += 1
    if text[right:right + 1] == " ":
        right += 1
    elif left > 0 and text[left - 1] == " ":
        left -= 1
    return left, right


def _drop_empty_structure(body: str) -> str:
    """Remove lead-ins and headings left without content by removed statements."""
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    changed = True
    while changed:
        changed = False
        for index, paragraph in enumerate(paragraphs):
            following = paragraphs[index + 1] if index + 1 < len(paragraphs) else None
            dangling_lead_in = paragraph.rstrip().endswith(":") and (
                following is None or following.startswith("#") or following.rstrip().endswith(":"))
            empty_section = paragraph.startswith("## ") and "\n" not in paragraph.strip() and (
                following is None or following.startswith("#"))
            if dangling_lead_in or empty_section:
                del paragraphs[index]
                changed = True
                break
    return "\n\n".join(paragraphs)


def build_memory_pages(facts, assignment, spec: MemorySpec, kb_a_dir) -> dict[str, str]:
    """kb_0 page text for every page that owns a fact, derived from its KB-A page."""
    pages = {}
    for source in sorted({fact.source for fact in facts}):
        path = Path(kb_a_dir) / source
        _, lines = _split_front_matter(path.read_text(encoding="utf-8").splitlines())
        body = "\n".join(lines)
        edits = []  # (start, end, replacement); applied right to left
        for fact in facts:
            status = assignment[fact.fact_id].status
            if fact.source != source or status == "correct":
                continue
            matches = list(re.finditer(_flexible(fact.pattern), body))
            if status == "unknown" and not matches:
                continue  # not on the KB-A page (added / absent)
            if len(matches) != 1:
                raise ValueError(f"{fact.fact_id}: expected one statement on KB-A {source}, found {len(matches)}")
            match = matches[0]
            if status == "stale":
                older = spec.older_values.get(fact.fact_id)
                if older is None:
                    raise ValueError(f"{fact.fact_id}: assigned stale but memory.yaml has no older value")
                if isinstance(older, dict):
                    edits.append((match.start(), match.end(), older["statement"]))
                else:
                    edits.append((match.start(1), match.end(1), str(older)))
            else:
                edits.append((*_sentence_span(body, match.start(1), match.end(1)), ""))
        for start, end, replacement in sorted(edits, reverse=True):
            body = body[:start] + replacement + body[end:]
        body = re.sub(r"[ \t]+\n", "\n", body)
        pages[source] = f"---\nupdated: {spec.updated}\n---\n{_drop_empty_structure(body)}\n"
    return pages


def verify_memory(facts, assignment, spec: MemorySpec, kb_0_chunks, kb_a_chunks, kb_b_chunks) -> None:
    """Every fact's kb_0 value matches its assigned status; no KB-B value leaks into memory."""
    for fact in facts:
        status = assignment[fact.fact_id].status
        memory = fact_value(fact, kb_0_chunks).value
        a, b = fact_value(fact, kb_a_chunks), fact_value(fact, kb_b_chunks)
        if status == "correct" and memory != a.value:
            raise ValueError(f"{fact.fact_id}: correct memory {memory!r} != KB-A {a.value!r}")
        if status == "unknown" and memory is not None:
            raise ValueError(f"{fact.fact_id}: unknown memory states {memory!r}")
        if status == "stale" and (memory is None or memory in {a.value, b.value, *a.conflicting, *b.conflicting}):
            raise ValueError(f"{fact.fact_id}: stale memory {memory!r} must differ from KB-A and KB-B values")
        if memory is not None and b.value != a.value and memory == b.value:
            raise ValueError(f"{fact.fact_id}: memory holds the post-drift KB-B value")
    for fact_id, older in spec.older_values.items():
        fact = next((f for f in facts if f.fact_id == fact_id), None)
        if fact is None:
            raise ValueError(f"memory.yaml: unknown fact {fact_id}")
        text = older["statement"] if isinstance(older, dict) else str(older)
        found = re.findall(fact.pattern, text) if isinstance(older, dict) else [text]
        values = {normalize_answer(v) for v in found}
        a, b = fact_value(fact, kb_a_chunks), fact_value(fact, kb_b_chunks)
        if len(values) != 1 or values & {a.value, b.value, *a.conflicting, *b.conflicting}:
            raise ValueError(f"memory.yaml: older value for {fact_id} must be one value distinct from KB-A/KB-B")
        if any(re.search(rf"\b{re.escape(v)}\b", q.lower()) for v in values for q in fact.questions):
            raise ValueError(f"memory.yaml: older value for {fact_id} appears in one of its questions")


def build(corpus_root, facts_path, spec_path, out_dir=None, check=False) -> dict[str, MemoryAssignment]:
    corpus_root = Path(corpus_root)
    facts, spec = load_facts(facts_path), load_memory_spec(spec_path)
    kb_a, kb_b = (load_knowledge_base(corpus_root, name) for name in ("kb_a", "kb_b"))
    assignment = assign_memory(facts, kb_a, kb_b, spec.seed)
    pages = build_memory_pages(facts, assignment, spec, corpus_root / "kb_a")
    out_dir = Path(out_dir) if out_dir else corpus_root / "kb_0"
    if check:
        existing = {path.name: path.read_text(encoding="utf-8") for path in out_dir.glob("*.md")}
        if existing != pages:
            raise SystemExit(f"{out_dir} differs from the memory rule; rebuild with `python -m src.data.memory`")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale_page in set(p.name for p in out_dir.glob("*.md")) - set(pages):
            (out_dir / stale_page).unlink()
        for name, text in pages.items():
            (out_dir / name).write_text(text, encoding="utf-8", newline="\n")
    verify_memory(facts, assignment, spec, load_documents(out_dir, knowledge_base="kb_0"), kb_a, kb_b)
    return assignment


def main():
    root = Path(__file__).resolve().parents[2] / "data" / "documentation"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify checked-in kb_0 instead of writing it")
    args = parser.parse_args()
    assignment = build(root, root / "facts.yaml", root / "memory.yaml", check=args.check)
    counts = {s: sum(a.status == s for a in assignment.values()) for s in MEMORY_STATUSES}
    print(f"kb_0 {'matches' if args.check else 'written'}: {counts}")


if __name__ == "__main__":
    main()
