import random

from src.environment.rl_rag_env import ANSWER, SEARCH_MORE


class AnswerDirectly:
    """Closed-book: answer immediately from memory."""
    def act(self, observation, info, explore=True): return ANSWER


class SearchThenAnswer:
    """Search a fixed number of times (bounded by the budget), then answer."""
    def __init__(self, searches): self.searches = searches
    def act(self, observation, info, explore=True):
        can_search = info["action_mask"][SEARCH_MORE]
        return SEARCH_MORE if info["searches"] < self.searches and can_search else ANSWER


class RandomPolicy:
    """Uniform over the currently available actions."""
    def __init__(self, seed=7): self.rng = random.Random(seed)
    def act(self, observation, info, explore=True):
        return self.rng.choice([action for action, allowed in enumerate(info["action_mask"]) if allowed])
