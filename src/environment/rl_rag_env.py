import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.data.documents import MEMORY_SNAPSHOT
from src.generation.mock import MockAnswerGenerator, evidence_candidates
from src.retrieval.retriever import Retriever

SEARCH_MORE, ANSWER, GIVE_UP = 0, 1, 2
ACTION_NAMES = ("search_more", "answer", "give_up")

# Policy-visible features. Each comes from the question, the retrieved
# evidence, or the frozen generator's own outputs; none uses the gold answer,
# drift type, or which page owns the fact. By default the observation is
# exactly these features; the query embedding is an opt-in ablation because a
# policy that sees it memorizes training questions instead of generalizing.
FEATURES = (
    "searches_used",                # searches so far / max_searches
    "can_search",                   # the retrieval budget allows another SEARCH_MORE
    "memory_has_answer",            # closed-book generator would answer (not "don't know")
    "evidence_has_answer",          # some retrieved chunk states the answer slot
    "evidence_conflict",            # retrieved chunks state more than one distinct value
    "evidence_agrees_with_memory",  # reader's evidence-based answer equals the memory answer
    "last_search_new_value",        # the latest search revealed a value not seen before
    "best_score",                   # highest retrieval similarity among retrieved chunks
    "last_search_score",            # highest retrieval similarity in the latest search
)


class RLRAGEnv(gym.Env):
    """One episode = one question.

    SEARCH_MORE reveals the next `top_k` unseen chunks of the question's
    ranking (paged search) at `search_cost`; it is masked once `max_searches`
    searches have been made. ANSWER asks the frozen reader to answer from the
    evidence gathered so far (closed-book if there is none). GIVE_UP abstains.
    ANSWER and GIVE_UP end the episode.

    The observation is FEATURES, preceded by the query embedding only when
    `observe_query_embedding` is True (ablation).

    Terminal rewards: ANSWER -> correct_reward if the answer equals the gold
    value, else incorrect_reward (including answering an unanswerable question).
    GIVE_UP -> correct_reward on an unanswerable question, else give_up_reward.
    """

    metadata = {"render_modes": []}

    def __init__(self, queries, chunks, *, top_k=3, max_searches=3, search_cost=0.10,
                 correct_reward=1.0, incorrect_reward=-1.0, give_up_reward=0.0, observe_query_embedding=False):
        self.queries = list(queries)
        chunks = list(chunks)
        # kb_0 is the frozen generator's closed-book memory; it reaches the
        # reader only through Query.memory_answer and must never be retrievable.
        if any(getattr(chunk, "knowledge_base", "") == MEMORY_SNAPSHOT for chunk in chunks):
            raise ValueError("kb_0 is generator memory and cannot be used as a retrieval corpus")
        self.retriever = Retriever(chunks)
        self.generator = MockAnswerGenerator()
        self.top_k = top_k
        self.max_searches = max_searches
        self.search_cost = search_cost
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.give_up_reward = give_up_reward
        self.observe_query_embedding = observe_query_embedding
        dimension = self.retriever.embedder.encode(["dimension probe"]).shape[1] if observe_query_embedding else 0
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(dimension + len(FEATURES),), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self._cursor = 0

    def reset(self, *, seed=None, options=None):
        """Start the question at options["query_index"], or the next one in order."""
        super().reset(seed=seed)
        index = (options or {}).get("query_index", self._cursor)
        self._cursor = (index + 1) % len(self.queries)
        self.query = self.queries[index]
        self._query_vector = (self.retriever.embedder.encode([self.query.text])[0]
                              if self.observe_query_embedding else np.empty(0, dtype=np.float32))
        self._ranking = self.retriever.search(self.query, self.top_k * self.max_searches)
        self.evidence = []
        self.searches = 0
        self._best_score = self._last_score = 0.0
        self._last_new_value = False
        self._conflict_seen = False
        return self._observation(), self._info()

    def action_mask(self) -> np.ndarray:
        return np.array([self.searches < self.max_searches, True, True])

    def _observation(self):
        values = [value for value, _ in evidence_candidates(self.query, self.evidence)]
        memory = self.query.memory_answer
        evidence_answer = self.generator.generate(self.query, self.evidence) if values else None
        features = [
            self.searches / self.max_searches,
            float(self.searches < self.max_searches),
            float(memory is not None),
            float(bool(values)),
            float(len(set(values)) > 1),
            float(evidence_answer is not None and evidence_answer == memory),
            float(self._last_new_value),
            self._best_score,
            self._last_score,
        ]
        return np.concatenate([self._query_vector, features]).astype(np.float32)

    def _info(self, **final):
        return {"action_mask": self.action_mask(), "searches": self.searches, **final}

    def step(self, action):
        action = int(action)
        if not self.action_mask()[action]:
            raise ValueError(f"{ACTION_NAMES[action]} is unavailable: retrieval budget exhausted")

        if action == SEARCH_MORE:
            start = self.searches * self.top_k
            batch = self._ranking[start:start + self.top_k]
            seen = {value for value, _ in evidence_candidates(self.query, self.evidence)}
            self.evidence += [result.fact for result in batch]
            self.searches += 1
            new_values = {value for value, _ in evidence_candidates(self.query, [r.fact for r in batch])}
            self._last_new_value = bool(new_values - seen)
            self._last_score = max((result.score for result in batch), default=0.0)
            self._best_score = max(self._best_score, self._last_score)
            self._conflict_seen = len(seen | new_values) > 1
            return self._observation(), -self.search_cost, False, False, self._info()

        gold = self.query.gold_answer
        if action == ANSWER:
            answer = self.generator.generate(self.query, self.evidence)
            correct = answer is not None and answer == gold
            reward = self.correct_reward if correct else self.incorrect_reward
        else:
            answer = None
            correct = gold is None
            reward = self.correct_reward if correct else self.give_up_reward
        retrieval_cost = self.search_cost * self.searches
        info = self._info(
            query_id=self.query.query_id,
            final_action=ACTION_NAMES[action],
            answer=answer,
            ground_truth=gold,
            correct=correct,
            terminal_reward=reward,
            episode_return=reward - retrieval_cost,
            retrieval_cost=retrieval_cost,
            evidence_conflict=self._conflict_seen,
            answerable=gold is not None,
            drift_type=self.query.drift_type,
            drift_event=self.query.drift_event,
            memory_status=self.query.memory_status,
            affected_by_drift=self.query.affected_by_drift,
        )
        return self._observation(), reward, True, False, info
