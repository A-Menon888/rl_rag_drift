"""Direct Preference Optimization (DPO) over retrieval trajectories.

DPO adapts the retrieval *policy* (SEARCH_MORE / ANSWER / GIVE_UP); the frozen
reader/generator is never trained. There is no human feedback: preferences are
derived automatically from environment episode returns on KB-B training
questions.

Pipeline:
1. `collect_trajectories`: for each KB-B training question, roll out every
   distinct stopping strategy the budget allows ("search s times, then ANSWER
   or GIVE_UP", s = 0..max_searches), recording each step's observation,
   action mask and action, plus the episode return.
2. `frozen_policy_actions` + `build_preference_pairs`: at most one pair per
   question. "Rejected" is the candidate the frozen pre-drift policy actually
   selects (its greedy action sequence); "chosen" is the highest-return
   candidate (first in enumeration order if several tie). A pair is created
   only when chosen return > frozen-policy return (questions the frozen
   policy already handles optimally contribute nothing) AND the two
   trajectories end in different answer outcomes (`Trajectory.outcome`):
   pairs that differ only in search count/retrieval cost are discarded.
3. `train_dpo`: minimize the DPO loss with the frozen pre-drift policy as the
   reference, starting the trainable policy from the same weights.

Returns (which use the gold answer, like the RL reward) only rank trajectories
for the loss. Observations are recorded exactly as the policy sees them and
never contain gold, drift type, or return.
"""
from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.environment.rl_rag_env import ACTION_NAMES, ANSWER, GIVE_UP, SEARCH_MORE


@dataclass(frozen=True)
class Trajectory:
    query_id: str
    observations: np.ndarray   # (steps, obs_dim), as seen by the policy
    action_masks: np.ndarray   # (steps, n_actions) bool
    actions: tuple[int, ...]
    episode_return: float
    correct: bool

    @property
    def label(self) -> str:
        return "+".join(ACTION_NAMES[action] for action in self.actions)

    @property
    def outcome(self) -> tuple[str, bool]:
        """Final answer outcome: (final action, correct). E.g. ("answer", False) is a wrong
        answer; ("give_up", True) is a correct abstention on an unanswerable question."""
        return ACTION_NAMES[self.actions[-1]], self.correct


@dataclass(frozen=True)
class PreferencePair:
    query_id: str
    chosen: Trajectory
    rejected: Trajectory


def stopping_strategies(max_searches: int) -> list[tuple[int, ...]]:
    """All distinct action sequences: SEARCH_MORE s times, then ANSWER or GIVE_UP."""
    return [(SEARCH_MORE,) * searches + (final,)
            for searches in range(max_searches + 1) for final in (ANSWER, GIVE_UP)]


def rollout(env, query_index: int, actions) -> Trajectory:
    observation, info = env.reset(options={"query_index": query_index})
    observations, masks = [], []
    terminated = False
    for action in actions:
        observations.append(observation)
        masks.append(info["action_mask"])
        observation, _, terminated, _, info = env.step(action)
    if not terminated:
        raise ValueError("action sequence did not end the episode")
    return Trajectory(info["query_id"], np.stack(observations).astype(np.float32),
                      np.stack(masks).astype(bool), tuple(actions), float(info["episode_return"]),
                      bool(info["correct"]))


def collect_trajectories(env) -> list[list[Trajectory]]:
    """One list of trajectories per question in `env.queries`."""
    strategies = stopping_strategies(env.max_searches)
    return [[rollout(env, index, actions) for actions in strategies] for index in range(len(env.queries))]


def frozen_policy_actions(agent, env) -> list[tuple[int, ...]]:
    """The greedy action sequence the (frozen) policy takes on each question."""
    sequences = []
    for index in range(len(env.queries)):
        observation, info = env.reset(options={"query_index": index})
        actions, terminated = [], False
        while not terminated:
            action = agent.act(observation, info, explore=False)
            observation, _, terminated, _, info = env.step(action)
            actions.append(action)
        sequences.append(tuple(actions))
    return sequences


def build_preference_pairs(trajectories_by_question, rejected_actions_by_question) -> list[PreferencePair]:
    """One pair per question: best candidate (chosen) vs the frozen policy's own
    trajectory (rejected), only when the best candidate has a strictly higher
    return and a different answer outcome (cost-only differences are dropped)."""
    pairs = []
    for trajectories, rejected_actions in zip(trajectories_by_question, rejected_actions_by_question, strict=True):
        rejected = next(t for t in trajectories if t.actions == tuple(rejected_actions))
        chosen = max(trajectories, key=lambda t: t.episode_return)  # first maximum in enumeration order
        if chosen.episode_return > rejected.episode_return and chosen.outcome != rejected.outcome:
            pairs.append(PreferencePair(chosen.query_id, chosen, rejected))
    return pairs


def trajectory_log_prob(agent, trajectory: Trajectory) -> torch.Tensor:
    """log pi(tau) = sum_t log pi(a_t | s_t). Transition terms are identical
    for the trained and reference policies and cancel in the DPO ratio."""
    distribution = agent._distribution(torch.as_tensor(trajectory.observations), trajectory.action_masks)
    return distribution.log_prob(torch.as_tensor(trajectory.actions)).sum()


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta: float) -> torch.Tensor:
    """-log sigmoid(beta * [(log pi(tau_w) - log pi_ref(tau_w)) - (log pi(tau_l) - log pi_ref(tau_l))])"""
    margin = (policy_chosen - reference_chosen) - (policy_rejected - reference_rejected)
    return -F.logsigmoid(beta * margin).mean()


def train_dpo(agent, reference, pairs, *, epochs: int, batch_size: int, beta: float, learning_rate: float,
              seed: int) -> dict:
    """Train `agent` in place against the frozen `reference`; returns budget and loss curve."""
    if not pairs:
        raise ValueError("no preference pairs: every question's trajectories tie")
    with torch.no_grad():
        reference_logps = [(trajectory_log_prob(reference, p.chosen), trajectory_log_prob(reference, p.rejected))
                           for p in pairs]
    optimizer = torch.optim.Adam(agent.policy.parameters(), lr=learning_rate)
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    curve, updates, policy_step_evaluations = [], 0, 0
    for _ in range(epochs):
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            chosen = torch.stack([trajectory_log_prob(agent, pairs[i].chosen) for i in batch])
            rejected = torch.stack([trajectory_log_prob(agent, pairs[i].rejected) for i in batch])
            loss = dpo_loss(chosen, rejected,
                            torch.stack([reference_logps[i][0] for i in batch]),
                            torch.stack([reference_logps[i][1] for i in batch]), beta)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.item()))
            updates += 1
            policy_step_evaluations += sum(len(pairs[i].chosen.actions) + len(pairs[i].rejected.actions)
                                           for i in batch)
        curve.append(float(np.mean(losses)))
    return {"loss_curve": curve, "gradient_updates": updates, "policy_step_evaluations": policy_step_evaluations}
