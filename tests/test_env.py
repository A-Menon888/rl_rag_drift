from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.agents.baselines import AnswerDirectly, RandomPolicy, SearchThenAnswer
from src.agents.rl_agent import RLAgent
from src.data.documents import DocumentChunk, load_knowledge_base
from src.data.facts import build_fact_queries, load_facts
from src.environment.rl_rag_env import ANSWER, FEATURES, GIVE_UP, SEARCH_MORE, RLRAGEnv
from src.generation.mock import MockAnswerGenerator
from experiments.run_all import approval_decision, evaluate_policy, recovery_decision, train_policy

CORPUS = Path(__file__).parents[1] / "data" / "documentation"


def _make_queries():
    kb_0, kb_a, kb_b = (load_knowledge_base(CORPUS, name) for name in ("kb_0", "kb_a", "kb_b"))
    return kb_a, kb_b, *build_fact_queries(load_facts(CORPUS / "facts.yaml"), kb_0, kb_a, kb_b)


def _feature(observation, name):
    return observation[len(observation) - len(FEATURES) + FEATURES.index(name)]


def _run(env, index, actions):
    observation, info = env.reset(options={"query_index": index})
    for action in actions:
        observation, reward, terminated, _, info = env.step(action)
    return observation, reward, terminated, info


def test_one_question_per_episode_and_step_contract():
    kb_a, _, queries_a, _ = _make_queries()
    env = RLRAGEnv(queries_a, kb_a, top_k=1, max_searches=3, search_cost=0.1)
    observation, info = env.reset(options={"query_index": 5})
    assert env.query is queries_a[5]
    assert observation.shape == env.observation_space.shape
    assert list(info["action_mask"]) == [True, True, True]
    _, reward, terminated, truncated, _ = env.step(SEARCH_MORE)
    assert reward == pytest.approx(-0.1) and not terminated and not truncated
    _, _, terminated, _, info = env.step(ANSWER)
    assert terminated
    assert {"query_id", "final_action", "answer", "ground_truth", "correct", "episode_return",
            "searches", "retrieval_cost", "drift_type", "memory_status", "evidence_conflict"} <= info.keys()
    assert info["query_id"] == queries_a[5].query_id and info["searches"] == 1


def test_search_more_reveals_new_unseen_chunks():
    kb_a, _, queries_a, _ = _make_queries()
    env = RLRAGEnv(queries_a, kb_a, top_k=2, max_searches=3)
    env.reset(options={"query_index": 0})
    revealed = []
    for _ in range(3):
        env.step(SEARCH_MORE)
        revealed.append([chunk.document_id for chunk in env.evidence])
    assert [len(ids) for ids in revealed] == [2, 4, 6]
    assert len(set(revealed[-1])) == 6


def test_retrieval_budget_masks_search_more():
    kb_a, _, queries_a, _ = _make_queries()
    env = RLRAGEnv(queries_a, kb_a, top_k=1, max_searches=2)
    _, _, _, info = _run(env, 0, [SEARCH_MORE, SEARCH_MORE])
    assert list(info["action_mask"]) == [False, True, True]
    with pytest.raises(ValueError):
        env.step(SEARCH_MORE)


def test_closed_book_answer_follows_memory():
    kb_a, _, queries_a, _ = _make_queries()
    env = RLRAGEnv(queries_a, kb_a)
    for index, query in enumerate(queries_a):
        _, _, _, info = _run(env, index, [ANSWER])
        assert info["correct"] is (query.memory_status == "correct" and query.gold_answer is not None)
        assert info["retrieval_cost"] == 0.0


def test_give_up_rewards_only_unanswerable_questions():
    _, kb_b, _, queries_b = _make_queries()
    env = RLRAGEnv(queries_b, kb_b, correct_reward=1.0, give_up_reward=0.0)
    for index, query in enumerate(queries_b):
        _, reward, _, info = _run(env, index, [GIVE_UP])
        assert info["correct"] is (query.gold_answer is None)
        assert reward == (1.0 if query.gold_answer is None else 0.0)


def test_answering_an_unanswerable_question_is_wrong():
    _, kb_b, _, queries_b = _make_queries()
    env = RLRAGEnv(queries_b, kb_b)
    for index, query in enumerate(queries_b):
        if query.gold_answer is None:
            _, reward, _, info = _run(env, index, [SEARCH_MORE, ANSWER])
            assert not info["correct"] and reward == -1.0


def test_reader_prefers_newest_source_regardless_of_rank_and_falls_back_to_memory():
    _, kb_b, _, queries_b = _make_queries()
    query = next(q for q in queries_b if q.drift_type == "contradicted")
    reader = MockAnswerGenerator()
    stating = [chunk for chunk in kb_b if reader.generate(replace(query, memory_answer=None), [chunk])]
    reference = next(chunk for chunk in stating if chunk.source == query.source)
    faq = next(chunk for chunk in stating if chunk.source == "faq.md")
    assert faq.updated < reference.updated
    assert reader.generate(query, [faq, reference]) == query.gold_answer
    assert reader.generate(query, [reference, faq]) == query.gold_answer
    assert reader.generate(query, [faq]) != query.gold_answer  # only the stale page was seen
    # Recency is an assumption, not an oracle: a newer wrong page wins.
    assert reader.generate(query, [replace(faq, updated="2099-01-01"), reference]) != query.gold_answer
    unrelated = next(chunk for chunk in kb_b if not reader.generate(replace(query, memory_answer=None), [chunk]))
    assert reader.generate(query, [unrelated]) == query.memory_answer
    assert reader.generate(replace(query, memory_answer=None), [unrelated]) is None


def test_contradictions_are_resolvable_by_searching_more():
    _, kb_b, _, queries_b = _make_queries()
    env = RLRAGEnv(queries_b, kb_b, top_k=1, max_searches=3)
    resolved = 0
    for index, query in enumerate(queries_b):
        if query.drift_type != "contradicted":
            continue
        _, _, _, first = _run(env, index, [SEARCH_MORE, ANSWER])
        observation, _, _, full = _run(env, index, [SEARCH_MORE] * 3 + [ANSWER])
        if not first["correct"] and full["correct"] and full["evidence_conflict"]:
            assert _feature(observation, "evidence_conflict") == 1.0
            resolved += 1
    assert resolved >= 1


def test_observation_never_depends_on_gold_or_drift_labels():
    _, kb_b, _, queries_b = _make_queries()
    query = next(q for q in queries_b if q.drift_type == "modified")
    relabelled = replace(query, gold_answer="something else", drift_type="unchanged", memory_status="correct")
    env = RLRAGEnv([query, relabelled], kb_b, top_k=1, max_searches=3)
    for actions in ([], [SEARCH_MORE], [SEARCH_MORE, SEARCH_MORE]):
        first = _run(env, 0, actions)[0] if actions else env.reset(options={"query_index": 0})[0]
        second = _run(env, 1, actions)[0] if actions else env.reset(options={"query_index": 1})[0]
        np.testing.assert_array_equal(first, second)


def test_retrieval_can_fail():
    """The environment must allow failure: exhaustive search is not an oracle."""
    kb_a, _, queries_a, _ = _make_queries()
    env = RLRAGEnv(queries_a, kb_a, top_k=1, max_searches=3)
    summary, _ = evaluate_policy(SearchThenAnswer(3), env)
    assert 0.0 < summary["accuracy"] < 1.0


def test_fixed_baselines_behave_as_named():
    kb_a, _, queries_a, _ = _make_queries()
    env = RLRAGEnv(queries_a, kb_a, top_k=1, max_searches=3)
    assert evaluate_policy(AnswerDirectly(), env)[0]["average_searches"] == 0.0
    assert evaluate_policy(SearchThenAnswer(1), env)[0]["average_searches"] == 1.0
    assert evaluate_policy(SearchThenAnswer(9), env)[0]["average_searches"] == 3.0
    _, infos = evaluate_policy(RandomPolicy(3), env, explore=True)
    assert all(info["searches"] <= 3 for info in infos)


def test_agent_never_takes_masked_action():
    kb_a, _, queries_a, _ = _make_queries()
    env = RLRAGEnv(queries_a, kb_a, top_k=1, max_searches=1)
    agent = RLAgent(env.observation_space.shape[0], seed=3)
    with torch.no_grad():
        agent.policy.net[-1].bias[SEARCH_MORE] = 100.0  # strongly prefers SEARCH_MORE
    observation, info = env.reset(options={"query_index": 0})
    observation, _, _, _, info = env.step(agent.act(observation, info))
    assert info["searches"] == 1
    assert all(agent.act(observation, info) != SEARCH_MORE for _ in range(20))


def test_evaluation_does_not_update_weights_and_training_does():
    kb_a, _, queries_a, _ = _make_queries()
    env = RLRAGEnv(queries_a[:8], kb_a)
    agent = RLAgent(env.observation_space.shape[0], learning_rate=0.01, seed=3)
    before = {name: value.clone() for name, value in agent.policy.state_dict().items()}
    evaluate_policy(agent, env)
    assert all(torch.equal(before[name], value) for name, value in agent.policy.state_dict().items())
    curve, budget = train_policy(agent, env, epochs=2, batch_size=4, seed=0)
    assert len(curve) == 2
    assert budget["adaptation_episodes"] == 16 and budget["gradient_updates"] == 4
    assert budget["policy_step_evaluations"] >= 16
    assert any(not torch.equal(before[name], value) for name, value in agent.policy.state_dict().items())


def test_default_observation_is_evidence_features_only():
    kb_a, _, queries_a, _ = _make_queries()
    env = RLRAGEnv(queries_a, kb_a)
    assert env.observe_query_embedding is False
    observation, _ = env.reset(options={"query_index": 0})
    assert observation.shape == env.observation_space.shape == (len(FEATURES),)
    # Two different questions with identical evidence state look identical to the policy.
    same_state = [q for q in queries_a if q.memory_answer is not None][:2]
    first = RLRAGEnv(same_state, kb_a).reset(options={"query_index": 0})[0]
    second = RLRAGEnv(same_state, kb_a).reset(options={"query_index": 1})[0]
    np.testing.assert_array_equal(first, second)


def test_query_embedding_observation_remains_available_as_ablation():
    kb_a, _, queries_a, _ = _make_queries()
    env = RLRAGEnv(queries_a, kb_a, observe_query_embedding=True)
    observation, _ = env.reset(options={"query_index": 0})
    dimension = env.retriever.embedder.encode(["probe"]).shape[1]
    assert observation.shape == (dimension + len(FEATURES),)
    np.testing.assert_allclose(observation[:dimension], env.retriever.embedder.encode([queries_a[0].text])[0])


def test_config_validation_rejects_missing_or_mistyped_keys():
    import yaml
    from experiments.run_all import validate_config
    config = yaml.safe_load((CORPUS.parents[1] / "configs" / "default.yaml").read_text(encoding="utf-8"))
    validate_config(config)
    assert config["observe_query_embedding"] is False
    with pytest.raises(ValueError, match="missing"):
        validate_config({k: v for k, v in config.items() if k != "observe_query_embedding"})
    with pytest.raises(ValueError, match="observe_query_embedding"):
        validate_config({**config, "observe_query_embedding": "no"})
    with pytest.raises(ValueError, match="dpo_epochs"):
        validate_config({**config, "dpo_epochs": 2.5})


def test_approval_bar_tracks_baseline():
    candidate = {"accuracy": 0.80, "retrieval_cost": 0.10}
    assert approval_decision(candidate, {"accuracy": 0.90, "retrieval_cost": 0.10})["approved"] is False
    assert approval_decision(candidate, {"accuracy": 0.70, "retrieval_cost": 0.10})["approved"] is True


def test_recovery_uses_old_policy_kb_a_accuracy():
    candidate = {"accuracy": 0.85}
    assert recovery_decision(candidate, {"accuracy": 0.85})["recovered"] is True
    assert recovery_decision(candidate, {"accuracy": 0.90})["recovered"] is False


def test_end_to_end_adaptation_path_uses_only_train_facts(tmp_path):
    """Old policy -> frozen checkpoint -> KB-B train trajectories -> DPO -> saved adapted policy."""
    import json
    import yaml
    from experiments import run_all
    from src.agents import dpo
    config = yaml.safe_load((CORPUS.parents[1] / "configs" / "default.yaml").read_text(encoding="utf-8"))
    config.update(train_epochs=2, batch_size=16, dpo_epochs=2)
    rl_questions, dpo_questions = [], []
    original_episode, original_rollout = RLAgent.run_episode, dpo.rollout

    def recording_run_episode(self, env, query_index):
        rl_questions.append(env.queries[query_index].fact_id)
        return original_episode(self, env, query_index)

    def recording_rollout(env, query_index, actions):
        dpo_questions.append((env.queries[query_index].fact_id, env.queries[query_index].drift_type))
        return original_rollout(env, query_index, actions)

    RLAgent.run_episode, dpo.rollout = recording_run_episode, recording_rollout
    try:
        rows = run_all.run_experiment(config, output_root=tmp_path)
    finally:
        RLAgent.run_episode, dpo.rollout = original_episode, original_rollout
    metrics = tmp_path / "metrics"
    split = json.loads((metrics / "split.json").read_text(encoding="utf-8"))
    test_facts = set(split["test_facts"])
    assert rl_questions and not set(rl_questions) & test_facts
    assert dpo_questions and not {fact for fact, _ in dpo_questions} & test_facts
    assert all(drift != "baseline" for _, drift in dpo_questions)  # DPO data comes from KB-B only

    keys = {(row["policy"], row["knowledge_base"], row["split"]) for row in rows}
    assert {("old_policy", "kb_a", "test"), ("old_policy", "kb_b", "test"), ("dpo_policy", "kb_b", "test"),
            ("rl_finetune_policy", "kb_b", "test"), ("full_retrain_policy", "kb_b", "test")} <= keys
    assert all(row["split"] == "test" for row in rows if row["policy"] not in
               {"old_policy", "dpo_policy", "rl_finetune_policy", "full_retrain_policy"})

    checkpoints = tmp_path / "checkpoints"
    obs_dim = len(FEATURES)
    old = RLAgent(obs_dim).load(checkpoints / "rl_old_policy_kb_a.pt")
    adapted = RLAgent(obs_dim).load(checkpoints / "dpo_policy_kb_b.pt")
    assert any(not torch.equal(a, b) for a, b in zip(old.policy.state_dict().values(),
                                                     adapted.policy.state_dict().values()))
    metadata = json.loads((metrics / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["observation"] == {"observe_query_embedding": False, "features": list(FEATURES),
                                       "dimension": obs_dim}
    assert metadata["dpo"]["preference_pairs"] > 0
    assert (metrics / "dpo_pairs.csv").exists()
    row = {(r["policy"], r["split"]): r for r in rows if r["knowledge_base"] == "kb_b"}
    assert row[("dpo_policy", "test")]["adaptation_episodes"] == metadata["dpo"]["adaptation_episodes"]
    # Equal-data control: RL fine-tuning uses exactly DPO's environment episodes, from the old weights.
    assert (row[("rl_finetune_policy", "test")]["adaptation_episodes"]
            == row[("dpo_policy", "test")]["adaptation_episodes"] == 8 * metadata["dpo"]["questions"])
    assert metadata["dpo"]["preference_pairs"] <= metadata["dpo"]["questions"]
    assert (checkpoints / "rl_finetune_policy_kb_b.pt").exists()
