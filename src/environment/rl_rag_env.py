import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.generation.mock import MockAnswerGenerator
from src.retrieval.retriever import Retriever

class RLRAGEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, queries, snapshots, drift_events=None, seed=7, retrieval_cost=0.10, correct_reward=1.0, incorrect_reward=-1.0, top_k=3):
        self.queries = queries
        self.snapshots = snapshots
        self.drift_events = drift_events or {}
        self.retrieval_cost = retrieval_cost
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.top_k = top_k
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32)
        self.generator = MockAnswerGenerator()
        self.t = 0
        self.recent_rewards = []
        self.recent_retrievals = []
        self.database_version = 0
        self.cache = {}
        self.retriever = Retriever(self.snapshots[0])

    def _state(self, query):
        embedding = self.retriever.embedder.encode([query.text])[0][:7]
        cache_available = float(query.query_id in self.cache)
        return np.concatenate([embedding, [self.recent_rewards[-1] if self.recent_rewards else 0.0, np.mean(self.recent_retrievals[-20:]) if self.recent_retrievals else 0.0, float(self.database_version), cache_available]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.database_version = 0
        self.recent_rewards, self.recent_retrievals = [], []
        self.cache = {}
        self.retriever = Retriever(self.snapshots[0])
        return self._state(self.queries[0]), {"database_version": 0}

    def step(self, action):
        query = self.queries[self.t % len(self.queries)]
        drift_event = self.drift_events.get(self.t)
        if drift_event is not None:
            self.database_version = drift_event["version"]
            self.retriever = Retriever(self.snapshots[self.database_version])
        action = int(action)
        cache_hit = action == 0 and query.query_id in self.cache
        results = self.retriever.search(query, self.top_k) if action == 1 else []
        context = self.cache.get(query.query_id) if cache_hit else [result.fact for result in results] if results else None
        answer = self.generator.generate(query, context)
        ground_truth = query.answer(self.database_version)
        correct = answer == ground_truth
        if action == 1 and correct:
            self.cache[query.query_id] = [result.fact for result in results]
        retrieval_cost = self.retrieval_cost if action == 1 else 0.0
        reward = (self.correct_reward if correct else self.incorrect_reward) - retrieval_cost
        self.recent_rewards.append(reward)
        self.recent_retrievals.append(action)
        info = {"query_id": query.query_id, "action": action, "answer": answer, "ground_truth": ground_truth, "correct": correct, "reward": reward, "database_version": self.database_version, "drift_event": drift_event, "drift_rate": 0.0, "retrieval_cost": retrieval_cost, "cache_hit": cache_hit, "cache_size": len(self.cache)}
        self.t += 1
        next_query = self.queries[self.t % len(self.queries)]
        return self._state(next_query), reward, False, False, info
