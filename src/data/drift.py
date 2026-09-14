from dataclasses import replace
from typing import List
import random
from .generator import Fact, Query

DRIFT_TYPES = ("update", "addition", "deletion", "contradiction")

def apply_drift(facts: List[Fact], version: int, rate: float, drift_type: str = "update", seed: int = 7) -> tuple[List[Fact], List[dict]]:
    if not 0 <= rate <= 1 or drift_type not in DRIFT_TYPES:
        raise ValueError("rate must be in [0, 1] and drift_type must be supported")
    rng = random.Random(seed + version)
    selected = [fact for fact in facts if rng.random() < rate]
    updated = list(facts)
    events = []
    for fact in selected:
        if drift_type == "addition":
            addition = replace(fact, value=f"new-{fact.value}", version=version, valid_from=version, document_id=f"doc-v{version}-{fact.document_id}")
            updated.append(addition)
        elif drift_type == "deletion":
            updated = [item for item in updated if item.document_id != fact.document_id]
        else:
            value = f"updated-{fact.value}" if drift_type == "update" else f"contradictory-{fact.value}"
            updated = [item for item in updated if item.document_id != fact.document_id]
            updated.append(replace(fact, value=value, version=version, valid_from=version, document_id=f"doc-v{version}-{fact.document_id}"))
        events.append({"entity": fact.entity, "attribute": fact.attribute, "type": drift_type, "version": version})
    return updated, events

def update_queries(queries: List[Query], snapshots: dict[int, List[Fact]]) -> List[Query]:
    result = []
    for query in queries:
        answers = {}
        for version, facts in snapshots.items():
            matches = [f for f in facts if f.entity == query.entity and f.attribute == query.attribute and f.valid_from <= version]
            answers[version] = matches[-1].value if matches else "UNKNOWN"
        affected = len(set(answers.values())) > 1
        result.append(replace(query, answer_by_version=answers, affected_by_drift=affected))
    return result
