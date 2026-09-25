from pathlib import Path
import torch

from src.data.documents import generate_document_queries, load_knowledge_base
from src.environment.rl_rag_env import RLRAGEnv
from src.agents.rl_agent import RLAgent
from experiments.run_all import approval_decision, evaluate_policy, recovery_decision, train_policy

CORPUS = Path(__file__).parents[1] / "data" / "documentation"


def _make_queries():
    kb_a = load_knowledge_base(CORPUS, "kb_a")
    kb_b = load_knowledge_base(CORPUS, "kb_b")
    return kb_a, kb_b, *generate_document_queries(kb_a, kb_b)


def test_environment_step_contract():
    kb_a, _, queries_a, _ = _make_queries()
    env = RLRAGEnv(queries_a, {0: kb_a})
    observation, info = env.reset()
    assert observation.shape == (388,)
    _, reward, terminated, truncated, step_info = env.step(0)
    assert isinstance(reward, float)
    assert not terminated and not truncated
    assert {"query_id", "answer", "ground_truth", "reward", "affected_by_drift"} <= step_info.keys()


def test_direct_correct_on_kb_a_without_cache():
    """DIRECT without cache should now be correct on KB-A (memorized == current)."""
    kb_a, _, queries_a, _ = _make_queries()
    # Pick a non-drifted query (all KB-A queries have no drift)
    query = queries_a[0]
    env = RLRAGEnv([query], {0: kb_a})
    env.reset()
    _, _, _, _, info = env.step(0)  # DIRECT
    assert info["correct"] is True, "DIRECT should be correct on KB-A (memorized == current)"
    assert info["cache_hit"] is False
    assert info["retrieval_cost"] == 0.0


def test_direct_wrong_on_drifted_kb_b_query():
    """DIRECT on a drifted KB-B query must be incorrect — memorized answer is stale."""
    kb_a, kb_b, _, queries_b = _make_queries()
    drifted = [q for q in queries_b if q.affected_by_drift]
    assert drifted, "Need at least one drifted query"
    query = drifted[0]
    env = RLRAGEnv([query], {0: kb_b})
    env.reset()
    _, _, _, _, info = env.step(0)  # DIRECT
    assert info["correct"] is False, "DIRECT on drifted KB-B query must be wrong"


def test_direct_correct_on_stable_kb_b_query():
    """DIRECT on a stable (unchanged) KB-B query must still be correct."""
    kb_a, kb_b, _, queries_b = _make_queries()
    stable = [q for q in queries_b if not q.affected_by_drift]
    assert stable, "Need at least one stable query"
    query = stable[0]
    env = RLRAGEnv([query], {0: kb_b})
    env.reset()
    _, _, _, _, info = env.step(0)  # DIRECT
    assert info["correct"] is True, "DIRECT on stable KB-B query must still be correct"


def test_retrieve_correct_regardless_of_kb():
    """RETRIEVE should find the correct chunk on both KB-A and KB-B."""
    kb_a, kb_b, queries_a, queries_b = _make_queries()
    for chunks, queries, label in [(kb_a, queries_a, "kb_a"), (kb_b, queries_b, "kb_b")]:
        env = RLRAGEnv(queries[:1], {0: chunks})
        env.reset()
        _, _, _, _, info = env.step(1)  # RETRIEVE
        assert info["correct"] is True, f"RETRIEVE must be correct on {label}"


def test_retrieval_caches_result_for_later_direct_action():
    kb_a, _, queries_a, _ = _make_queries()
    query = queries_a[0]
    env = RLRAGEnv([query, query], {0: kb_a})
    env.reset()
    _, _, _, _, retrieved = env.step(1)
    _, _, _, _, direct = env.step(0)
    assert retrieved["correct"] is True
    assert retrieved["cache_size"] == 1
    assert direct["cache_hit"] is True
    assert direct["correct"] is True
    assert direct["retrieval_cost"] == 0.0


def test_rl_evaluation_returns_computed_metrics():
    kb_a, _, queries_a, _ = _make_queries()
    env = RLRAGEnv(queries_a[:2], {0: kb_a})
    summary = evaluate_policy(RLAgent(388, learning_rate=0.001, seed=3), env, episodes=1)[0]
    assert {"accuracy", "average_reward", "retrieval_rate", "retrieval_cost"} <= summary.keys()
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert 0.0 <= summary["retrieval_rate"] <= 1.0


def test_evaluating_old_policy_does_not_update_weights():
    kb_a, _, queries_a, _ = _make_queries()
    env = RLRAGEnv(queries_a[:2], {0: kb_a})
    policy = RLAgent(388, learning_rate=0.001, seed=3)
    before = {name: value.detach().clone() for name, value in policy.policy.state_dict().items()}
    evaluate_policy(policy, env, episodes=1)
    after = policy.policy.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_adaptation_starts_from_old_weights_and_updates_on_kb_b():
    kb_a, kb_b, queries_a, queries_b = _make_queries()
    old_policy = RLAgent(388, learning_rate=0.001, seed=3)
    train_policy(old_policy, RLRAGEnv(queries_a, {0: kb_a}), episodes=1)
    adapted = RLAgent(388, learning_rate=0.001, seed=4)
    adapted.policy.load_state_dict(old_policy.policy.state_dict())
    before = {name: value.detach().clone() for name, value in adapted.policy.state_dict().items()}
    train_policy(adapted, RLRAGEnv(queries_b, {0: kb_b}), episodes=1)
    after = adapted.policy.state_dict()
    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_full_retrain_produces_checkpoint_and_summary_row(tmp_path):
    """Smoke test: full-retrain arm trains, saves a checkpoint, and its metrics are valid."""
    import copy
    kb_a, kb_b, queries_a, queries_b = _make_queries()
    retrain = RLAgent(388, learning_rate=0.001, seed=7)
    env = RLRAGEnv(queries_b, {0: kb_b})
    train_policy(retrain, env, episodes=2)
    eval_env = RLRAGEnv(queries_b, {0: kb_b})
    history = evaluate_policy(retrain, eval_env, episodes=1)
    assert history, "full-retrain evaluation must return at least one summary row"
    row = history[0]
    assert 0.0 <= row["accuracy"] <= 1.0
    assert "average_reward" in row


def test_approval_bar_tracks_always_retrieve_baseline():
    candidate = {"accuracy": 0.80, "retrieval_cost": 0.10}
    strong_baseline = {"accuracy": 0.90, "retrieval_cost": 0.10}
    weak_baseline = {"accuracy": 0.70, "retrieval_cost": 0.10}

    assert approval_decision(candidate, strong_baseline, margin=0.0)["approved"] is False
    assert approval_decision(candidate, weak_baseline, margin=0.0)["approved"] is True


def test_recovery_uses_old_policy_kb_a_accuracy():
    candidate = {"accuracy": 0.85}
    assert recovery_decision(candidate, {"accuracy": 0.85})["recovered"] is True
    assert recovery_decision(candidate, {"accuracy": 0.90})["recovered"] is False
