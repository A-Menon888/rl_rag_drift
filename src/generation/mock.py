from .base import AnswerGenerator


class MockAnswerGenerator(AnswerGenerator):
    def generate(self, query, context=None) -> str:
        if context:
            return context[0].value if hasattr(context[0], "value") else str(context[0])
        # DIRECT without cache: return what the agent memorized from KB-A training.
        # This is correct on KB-A, but wrong for any query whose content changed in KB-B.
        return query.memorized_answer
