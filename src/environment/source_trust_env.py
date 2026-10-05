"""Global source-trust environment for the source-reliability drift experiment.

A thin, step-wise wrapper around the validated pilot (`src/pilot/source_drift.py`),
frozen configuration: mirror cache, choose-source reader, page-concentrated
staleness, NO page feature. Everything (retrieval, reader, reward, observable
state) is computed by the pilot functions; this module only adds the
reset/step interface the RL/DPO code needs and the policy's vector encoding.

Actions (index = position in `ACTIONS`, the pilot's `Variant.actions` order):
answer_A, answer_B, give_up, search_A, search_B. Searches are masked after
`MAX_SEARCHES`. Rewards: search -cost(source); answer +1 / -1; give up 0.

Observation (9 binary inputs, a one-to-one encoding of the pilot state):
for each search position 0..2, [searched A, searched B]; then A stated the
answer slot, B stated the answer slot, A and B disagree. No question, page,
value, staleness or score information.
"""
import collections
import random

import numpy as np

from src.pilot import source_drift as P

VARIANT = P.Variant(cache="mirror", reader="choose", staleness="page", page_feature="none")
ACTIONS = VARIANT.actions
TERMINALS = VARIANT.terminals
OBS_SIZE = 2 * P.MAX_SEARCHES + 3


def encode(state) -> np.ndarray:
    seq, a_stated, b_stated, disagree = state
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    for i, source in enumerate(seq):
        obs[2 * i + (source == "B")] = 1.0
    obs[-3:] = (a_stated, b_stated, disagree)
    return obs


def action_mask(seq) -> np.ndarray:
    can_search = len(seq) < P.MAX_SEARCHES
    return np.array([a in TERMINALS or can_search for a in ACTIONS], dtype=bool)


class SourceTrustEnv:
    """One episode = one question. `world`/`ranked` come from the pilot builders."""

    def __init__(self, questions, world, ranked):
        self.queries = list(questions)
        self.world, self.ranked = world, ranked
        self.max_searches = P.MAX_SEARCHES
        self.observation_size = OBS_SIZE

    def state(self):
        return P.state_of(self.question, self.seq, self.ranked, VARIANT)

    def _info(self):
        return {"action_mask": action_mask(self.seq), "query_id": self.question.query_id}

    def reset(self, options=None):
        self.question = self.queries[(options or {})["query_index"]]
        self.seq, self.total = (), 0.0
        return encode(self.state()), self._info()

    def step(self, action: int):
        name = ACTIONS[action]
        if not action_mask(self.seq)[action]:
            raise ValueError(f"masked action {name}")
        if name in TERMINALS:
            found = P.evidence(self.question, self.seq, self.ranked)
            answer = P.reader_answer(found, name) if name != P.GIVE_UP else None
            reward = P.terminal_reward(name, answer, self.world.truth[self.question.fact_id])
            self.total += reward
            info = self._info()
            info.update(final_action=name, answer=answer, correct=reward > 0, episode_return=self.total,
                        seq=self.seq, searches=len(self.seq), searches_a=self.seq.count("A"),
                        searches_b=self.seq.count("B"), retrieval_cost=P.search_cost(self.seq),
                        cache_stale=self.question.fact_id in self.world.stale)   # audit only, never observed
            return encode(self.state()), reward, True, False, info
        source = name.removeprefix("search_")
        reward = -P.COSTS[source]
        self.seq += (source,)
        self.total += reward
        return encode(self.state()), reward, False, False, self._info()


def group_split(pool, test_fraction=0.4, seed=0):
    """The R2 diagnostic's split: split units (drift events + near-duplicate groups)
    stratified by the set of pages of their facts. Returns (train fact ids, test fact ids)."""
    units = collections.defaultdict(set)
    for fid, (fact, *_) in pool.items():
        units[fact.unit].add(fid)
    strata = collections.defaultdict(list)
    for unit, fids in units.items():
        strata["+".join(sorted({pool[f][0].source for f in fids}))].append(unit)
    rng, test = random.Random(seed), set()
    for stratum in sorted(strata):
        members = sorted(strata[stratum])
        rng.shuffle(members)
        n = 1 if len(members) == 1 else min(len(members) - 1, max(1, round(len(members) * test_fraction)))
        for unit in members[:n]:
            test |= units[unit]
    return set(pool) - test, test
