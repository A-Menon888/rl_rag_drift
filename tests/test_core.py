from src.data.generator import generate_database, generate_queries
from src.data.drift import apply_drift, update_queries
from src.retrieval.retriever import Retriever

def test_ground_truth_changes_after_update():
    facts = generate_database(seed=1, entities=2)
    queries = generate_queries(facts, count=1, seed=1)
    changed, _ = apply_drift(facts, 1, 1.0, "update", seed=1)
    query = update_queries(queries, {0: facts, 1: changed})[0]
    assert query.answer(0) != query.answer(1)

def test_retrieval_returns_metadata_and_score():
    facts = generate_database(seed=2, entities=2)
    result = Retriever(facts).search("What is the CEO of Nova00?", 1)[0]
    assert result.fact.entity == "Nova00"
    assert isinstance(result.score, float)
