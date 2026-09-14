from abc import ABC, abstractmethod

class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, query, context=None) -> str:
        raise NotImplementedError
