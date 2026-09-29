from src.data.facts import extract_values
from .base import AnswerGenerator


def evidence_candidates(query, chunks) -> list[tuple[str, str]]:
    """(value, page updated date) for every statement of the question's answer slot, in evidence order."""
    return [(value, chunk.updated) for chunk in chunks for value in extract_values(query.answer_pattern, chunk.text)]


class MockAnswerGenerator(AnswerGenerator):
    """Deterministic slot reader standing in for a frozen LLM.

    Explicit experimental assumptions:
    - Retrieved evidence overrides closed-book memory; memory (the KB-0
      snapshot) is used only when the evidence does not state the answer.
    - When evidence conflicts, newer documentation supersedes older: the value
      from the most recently updated page wins, ties broken by evidence order.
    The reader never knows which page owns a fact, so it answers wrongly when
    only a stale page was retrieved or when a stale page is the newer one.
    Returns None ("I don't know") when neither evidence nor memory answers.
    """

    def generate(self, query, context=None) -> str | None:
        candidates = evidence_candidates(query, context or [])
        if not candidates:
            return query.memory_answer
        newest = max(updated for _, updated in candidates)
        return next(value for value, updated in candidates if updated == newest)
