from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    query_id: str
    fact_id: str
    source: str
    text: str
    answer_pattern: str        # answer slot the reader extracts; never contains the value
    memory_answer: str | None  # closed-book answer from the KB-0 memory snapshot; None = "don't know"
    gold_answer: str | None    # value on the evaluated KB's owning page; None = unanswerable
    drift_type: str            # KB-A -> KB-B change of this fact; "baseline" for KB-A queries
                               # ("absent" = never documented: unanswerable, not drift)
    memory_status: str         # unknown (no memory) | correct | stale, relative to gold_answer
    drift_event: str = ""      # KB-A -> KB-B change this fact belongs to (facts.yaml); shared by
                               # correlated facts, otherwise the fact id
    split_unit: str = ""       # facts kept on one side of train/test (drift events + near-duplicates)

    @property
    def affected_by_drift(self) -> bool:
        return self.drift_type not in {"baseline", "unchanged", "absent"}
