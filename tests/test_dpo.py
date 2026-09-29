import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src.agents.dpo import (
    PreferencePair, Trajectory, build_preference_pairs, collect_trajectories, dpo_loss, frozen_policy_actions,
    rollout, stopping_strategies, train_dpo, trajectory_log_prob,
)
from src.agents.rl_agent import RLAgent
from src.data.documents import load_knowledge_base
from src.data.facts import build_fact_queries, load_facts, select_facts, split_facts
from src.environment.rl_rag_env import ANSWER, GIVE_UP, SEARCH_MORE, RLRAGEnv

CORPUS = Path(__file__).parents[1] / "data" / "documentation"


@pytest.fixture(scope="module")
def kb_b_train():
    kb_0, kb_a, kb_b = (load_knowledge_base(CORPUS, name) for name in ("kb_0", "kb_a", "kb_b"))
    _, queries_b = build_fact_queries(load_facts(CORPUS / "facts.yaml"), kb_0, kb_a, kb_b)
    train, test = split_facts(queries_b, 0.4, 0)
    env = RLRAGEnv(select_facts(queries_b, train), kb_b, top_k=1, max_searches=3)
    return env, test


@pytest.fixture(scope="module")
def trajectories(kb_b_train):
    return collect_trajectories(kb_b_train[0])


def _trajectory(query_id, episode_return, actions=(ANSWER,), correct=None):
    steps = len(actions)
    return Trajectory(query_id, np.zeros((steps, 9), dtype=np.float32), np.ones((steps, 3), dtype=bool),
                      tuple(actions), episode_return, episode_return > 0 if correct is None else correct)


def test_stopping_strategies_enumerate_every_budgeted_stop():
    strategies = stopping_strategies(3)
    assert len(strategies) == len(set(strategies)) == 8
    for actions in strategies:
        assert actions[-1] in (ANSWER, GIVE_UP)
        assert set(actions[:-1]) <= {SEARCH_MORE} and len(actions) - 1 <= 3


def test_trajectories_record_policy_observations_and_returns(kb_b_train, trajectories):
    env, _ = kb_b_train
    assert len(trajectories) == len(env.queries)
    assert all(len(per_question) == 8 for per_question in trajectories)
    for trajectory in trajectories[0]:
        assert trajectory.observations.shape == (len(trajectory.actions), env.observation_space.shape[0])
        assert trajectory.action_masks.shape == (len(trajectory.actions), 3)
        assert all(trajectory.action_masks[t][a] for t, a in enumerate(trajectory.actions))
    # Replaying an action sequence reproduces the same trajectory (deterministic environment).
    again = rollout(env, 0, trajectories[0][3].actions)
    np.testing.assert_array_equal(again.observations, trajectories[0][3].observations)
    assert again.episode_return == trajectories[0][3].episode_return


def test_frozen_policy_actions_match_greedy_evaluation(kb_b_train):
    env, _ = kb_b_train
    agent = RLAgent(env.observation_space.shape[0], seed=3)
    sequences = frozen_policy_actions(agent, env)
    assert len(sequences) == len(env.queries)
    candidates = set(stopping_strategies(env.max_searches))
    assert all(sequence in candidates for sequence in sequences)  # greedy choices are enumerated candidates
    assert frozen_policy_actions(agent, env) == sequences          # deterministic


def test_preference_pairs_are_best_vs_frozen_policy_one_per_question(kb_b_train, trajectories):
    env, test_facts = kb_b_train
    agent = RLAgent(env.observation_space.shape[0], seed=3)
    frozen = frozen_policy_actions(agent, env)
    pairs = build_preference_pairs(trajectories, frozen)
    assert pairs
    assert len({pair.query_id for pair in pairs}) == len(pairs) <= len(env.queries)
    frozen_by_id = {per_question[0].query_id: actions for per_question, actions in zip(trajectories, frozen)}
    best = {per_question[0].query_id: max(t.episode_return for t in per_question) for per_question in trajectories}
    for pair in pairs:
        assert pair.rejected.actions == frozen_by_id[pair.query_id]
        assert pair.chosen.episode_return == best[pair.query_id] > pair.rejected.episode_return
        assert pair.chosen.outcome != pair.rejected.outcome
        assert pair.chosen.query_id == pair.rejected.query_id == pair.query_id
    # A question has a pair exactly when the frozen policy is suboptimal in answer outcome, not just cost.
    for per_question, actions in zip(trajectories, frozen):
        frozen_trajectory = next(t for t in per_question if t.actions == actions)
        chosen = max(per_question, key=lambda t: t.episode_return)
        has_pair = per_question[0].query_id in {pair.query_id for pair in pairs}
        assert has_pair == (frozen_trajectory.episode_return < chosen.episode_return
                            and frozen_trajectory.outcome != chosen.outcome)
    assert {pair.query_id for pair in pairs} <= {query.query_id for query in env.queries}
    assert not {pair.query_id.rsplit(".q", 1)[0] for pair in pairs} & test_facts


def test_preference_pairs_handle_ties():
    answer, give_up, search_answer = (ANSWER,), (GIVE_UP,), (SEARCH_MORE, ANSWER)
    # Frozen policy already achieves the best return: no pair.
    optimal = [[_trajectory("q", 1.0, answer), _trajectory("q", 0.0, give_up)]]
    assert build_preference_pairs(optimal, [answer]) == []
    # Tied frozen and best returns: no pair (strictly higher return required).
    tied = [[_trajectory("q", 0.5, answer), _trajectory("q", 0.5, give_up)]]
    assert build_preference_pairs(tied, [give_up]) == []
    # Several best candidates: the first in enumeration order is chosen; one pair only.
    two_best = [[_trajectory("q", 1.0, answer), _trajectory("q", 1.0, give_up), _trajectory("q", -1.0, search_answer)]]
    pairs = build_preference_pairs(two_best, [search_answer])
    assert len(pairs) == 1 and pairs[0].chosen.actions == answer and pairs[0].rejected.actions == search_answer


def test_cost_only_pairs_are_discarded_and_outcome_changes_kept():
    answer, give_up = (ANSWER,), (GIVE_UP,)
    search_answer, search_give_up = (SEARCH_MORE, ANSWER), (SEARCH_MORE, GIVE_UP)
    two_search_answer = (SEARCH_MORE, SEARCH_MORE, ANSWER)

    def pair(candidates, frozen):
        return build_preference_pairs([candidates], [frozen])

    # Right answer either way, fewer searches: cost-only -> no pair.
    assert pair([_trajectory("q", 1.0, answer, True), _trajectory("q", 0.9, search_answer, True)], search_answer) == []
    # Correct abstention either way: cost-only -> no pair.
    assert pair([_trajectory("q", 1.0, give_up, True), _trajectory("q", 0.9, search_give_up, True)], search_give_up) == []
    # Wrong -> right (e.g. a contradiction resolved by searching again): kept.
    kept = pair([_trajectory("q", -1.1, search_answer, False), _trajectory("q", 0.8, two_search_answer, True)],
                search_answer)
    assert len(kept) == 1 and kept[0].rejected.outcome == ("answer", False) and kept[0].chosen.outcome == ("answer", True)
    # Wrong -> GIVE_UP on an answerable question: kept.
    kept = pair([_trajectory("q", 0.0, give_up, False), _trajectory("q", -1.1, search_answer, False)], search_answer)
    assert len(kept) == 1 and kept[0].chosen.outcome == ("give_up", False)
    # Right -> wrong (possible only if the reward ever ranked it higher): kept, outcome differs.
    kept = pair([_trajectory("q", 0.5, answer, False), _trajectory("q", 0.2, two_search_answer, True)], two_search_answer)
    assert len(kept) == 1 and kept[0].rejected.outcome == ("answer", True)


def test_dpo_loss_is_log2_at_reference_and_falls_with_positive_margin():
    zero = torch.zeros(4)
    assert float(dpo_loss(zero, zero, zero, zero, beta=0.1)) == pytest.approx(math.log(2))
    better = dpo_loss(torch.full((4,), 2.0), zero, zero, zero, beta=0.1)
    worse = dpo_loss(torch.full((4,), -2.0), zero, zero, zero, beta=0.1)
    assert float(better) < math.log(2) < float(worse)


def test_trajectory_log_prob_sums_masked_step_log_probs(kb_b_train, trajectories):
    env, _ = kb_b_train
    agent = RLAgent(env.observation_space.shape[0], seed=3)
    trajectory = trajectories[0][-1]  # three searches then GIVE_UP
    with torch.no_grad():
        expected = sum(float(agent._distribution(obs, mask).log_prob(torch.tensor(action)))
                       for obs, mask, action in zip(trajectory.observations, trajectory.action_masks, trajectory.actions))
    assert float(trajectory_log_prob(agent, trajectory).detach()) == pytest.approx(expected, abs=1e-5)


def test_train_dpo_moves_policy_toward_chosen_and_leaves_reference_frozen(kb_b_train, trajectories, tmp_path):
    env, _ = kb_b_train
    reference = RLAgent(env.observation_space.shape[0], seed=3)
    pairs = build_preference_pairs(trajectories, frozen_policy_actions(reference, env))
    reference.save(tmp_path / "old.pt")
    policy = RLAgent(env.observation_space.shape[0], seed=99).load(tmp_path / "old.pt")
    frozen = {name: value.clone() for name, value in reference.policy.state_dict().items()}

    def mean_margin():
        with torch.no_grad():
            return float(np.mean([
                (trajectory_log_prob(policy, p.chosen) - trajectory_log_prob(reference, p.chosen))
                - (trajectory_log_prob(policy, p.rejected) - trajectory_log_prob(reference, p.rejected))
                for p in pairs]))

    assert mean_margin() == pytest.approx(0.0, abs=1e-5)  # starts exactly at the reference
    stats = train_dpo(policy, reference, pairs, epochs=3, batch_size=32, beta=0.1, learning_rate=0.01, seed=0)
    assert mean_margin() > 0.0
    assert stats["loss_curve"][-1] < stats["loss_curve"][0]
    assert stats["gradient_updates"] == 3 * math.ceil(len(pairs) / 32)
    assert all(torch.equal(frozen[name], value) for name, value in reference.policy.state_dict().items())
    with pytest.raises(ValueError):
        train_dpo(policy, reference, [], epochs=1, batch_size=32, beta=0.1, learning_rate=0.01, seed=0)


def test_checkpoint_save_and_load_round_trip(kb_b_train, tmp_path):
    env, _ = kb_b_train
    agent = RLAgent(env.observation_space.shape[0], seed=5)
    agent.save(tmp_path / "policy.pt")
    restored = RLAgent(env.observation_space.shape[0], seed=6).load(tmp_path / "policy.pt")
    observation, info = env.reset(options={"query_index": 0})
    with torch.no_grad():
        np.testing.assert_allclose(agent._distribution(observation, info["action_mask"]).probs,
                                   restored._distribution(observation, info["action_mask"]).probs)
