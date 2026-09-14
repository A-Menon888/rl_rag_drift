from .base import AnswerGenerator

class MockAnswerGenerator(AnswerGenerator):
    def generate(self, query, context=None) -> str:
        if context:
            return context[0].value if hasattr(context[0], "value") else str(context[0])
        return query.direct_answer
