from pathlib import Path

from src.data.generator import generate_database, generate_queries
from src.environment.rl_rag_env import RLRAGEnv
from src.data.documents import generate_document_queries, load_knowledge_base
from src.agents.rl_agent import RLAgent
from experiments.run_all import evaluate_policy, train_policy

CORPUS = Path(__file__).parents[1] / "data" / "documentation"
from src.environment.rl_rag_env import RLRAGEnv

def test_environment_step_contract():
    facts = generate_database(seed=3, entities=2)
    queries = generate_queries(facts, 4, seed=3)
    env = RLRAGEnv(queries, {0: facts})
    observation, info = env.reset()
    assert observation.shape == (11,)
    _, reward, terminated, truncated, step_info = env.step(0)
    assert isinstance(reward, float)
    assert not terminated and not truncated
    assert {"query_id", "answer", "ground_truth", "reward"} <= step_info.keys()


def test_direct_without_cache_cannot_answer():
    documents = load_knowledge_base(CORPUS, "kb_a")
    query = generate_document_queries(documents)[0]
    env = RLRAGEnv([query], {0: documents})
    env.reset()
    _, _, _, _, info = env.step(0)
    assert info["correct"] is False
    assert info["cache_hit"] is False
    assert info["retrieval_cost"] == 0.0


def test_retrieval_caches_result_for_later_direct_action():
    documents = load_knowledge_base(CORPUS, "kb_a")
    query = generate_document_queries(documents)[0]
    env = RLRAGEnv([query, query], {0: documents})
    env.reset()
    _, _, _, _, retrieved = env.step(1)
    _, _, _, _, direct = env.step(0)
    assert retrieved["correct"] is True
    assert retrieved["cache_size"] == 1
    assert direct["cache_hit"] is True
    assert direct["correct"] is True
    assert direct["retrieval_cost"] == 0.0


def test_rl_evaluation_returns_computed_metrics():
    documents = load_knowledge_base(CORPUS, "kb_a")
    queries = generate_document_queries(documents)[:2]
    env = RLRAGEnv(queries, {0: documents})
    summary = evaluate_policy(RLAgent(11, learning_rate=0.01, seed=3), env, episodes=1)[0]
    assert {"accuracy", "average_reward", "retrieval_rate", "retrieval_cost"} <= summary.keys()
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert 0.0 <= summary["retrieval_rate"] <= 1.0
    assert summary["retrieval_cost"] == summary["retrieval_rate"] * 0.10

def test_evaluating_old_policy_does_not_update_weights():
    import torch

    documents = load_knowledge_base(CORPUS, "kb_b")
    queries = generate_document_queries(documents)[:2]
    env = RLRAGEnv(queries, {0: documents})
    policy = RLAgent(11, learning_rate=0.01, seed=3)
    before = {name: value.detach().clone() for name, value in policy.policy.state_dict().items()}
    evaluate_policy(policy, env, episodes=1)
    after = policy.policy.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_adaptation_starts_from_old_weights_and_updates_on_kb_b():
    import torch

    kb_a = load_knowledge_base(CORPUS, "kb_a")
    kb_b = load_knowledge_base(CORPUS, "kb_b")
    old_policy = RLAgent(11, learning_rate=0.01, seed=3)
    train_policy(old_policy, RLRAGEnv(generate_document_queries(kb_a), {0: kb_a}), episodes=1)
    adapted_policy = RLAgent(11, learning_rate=0.01, seed=4)
    adapted_policy.policy.load_state_dict(old_policy.policy.state_dict())
    before = {name: value.detach().clone() for name, value in adapted_policy.policy.state_dict().items()}
    train_policy(adapted_policy, RLRAGEnv(generate_document_queries(kb_b), {0: kb_b}), episodes=1)
    after = adapted_policy.policy.state_dict()
    assert any(not torch.equal(before[name], after[name]) for name in before)
