from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Query:
    query_id: str
    entity: str
    attribute: str
    text: str
    memorized_answer: str   # KB-A text, frozen — models deployed agent's internal knowledge
    current_answer: str     # ground-truth text for whichever KB this query is evaluated against
    affected_by_drift: bool # True iff memorized_answer != current_answer

    def answer(self, version: int = 0) -> str:
        """Compatibility shim used by env and tests. Always returns current_answer."""
        return self.current_answer
