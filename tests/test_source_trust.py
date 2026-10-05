"""Source-trust experiment: environment wrapper, split, and the R2 preference rule (incl. the 0% guard)."""
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from src.agents.state_preferences import action_indices, r2_preferences, solve, table_policy, to_dpo_pairs
from src.data.documents import load_knowledge_base
from src.data.facts import load_facts
from src.environment.source_trust_env import ACTIONS, TERMINALS, VARIANT, SourceTrustEnv, action_mask, encode, group_split
from src.pilot import source_drift as P
from src.retrieval.retriever import Retriever

CORPUS = Path(__file__).parents[1] / "data" / "documentation"


def _toy(truth_value, cache_value, n=1):
    questions = [P.PilotQuestion(f"f{i}.q0", f"f{i}", "what is x?", r"x is (\d)") for i in range(n)]
    filler = P.Entry("unrelated text", "A", "p")
    ranked = {}
    for q in questions:
        ranked[(q.query_id, "A")] = [P.Entry(f"x is {cache_value}", "A", "p"), filler, filler]
        ranked[(q.query_id, "B")] = [P.Entry(f"x is {truth_value}", "B", "p"), filler, filler]
    stale = frozenset(q.fact_id for q in questions) if truth_value != cache_value else frozenset()
    return questions, P.World("toy", {q.fact_id: str(truth_value) for q in questions}, {}, stale), ranked


def test_actions_and_observation_match_the_frozen_specification():
    assert ACTIONS == ("answer_A", "answer_B", "give_up", "search_A", "search_B")
    assert VARIANT.page_feature == "none" and VARIANT.cache == "mirror" and VARIANT.reader == "choose"
    assert P.COSTS == {"A": 0.05, "B": 0.10} and P.MAX_SEARCHES == 3
    states = {(seq, a, b, a and b and d) for seq in P.sequences() for a, b, d in product((False, True), repeat=3)}
    codes = {tuple(encode(s)) for s in states}
    assert len(codes) == len(states)                     # one-to-one: nothing added, nothing lost
    assert not action_mask(("A", "B", "A"))[ACTIONS.index("search_A")]
    assert action_mask(("A",)).all()


def test_env_returns_equal_pilot_strategy_returns():
    questions, world, ranked = _toy(2, 1)
    env = SourceTrustEnv(questions, world, ranked)
    for seq, end in P.strategies(VARIANT):
        env.reset(options={"query_index": 0})
        for action in action_indices(seq, end):
            obs, _, done, _, info = env.step(action)
        assert done and info["episode_return"] == pytest.approx(P.strategy_return(questions[0], world, ranked, seq, end))
        assert info["cache_stale"]


def test_r2_prefers_the_reference_after_drift_and_gives_nothing_without_drift():
    questions, current, ranked_current = _toy(1, 1, n=3)
    frozen = table_policy(solve(questions, current, ranked_current))     # exact pre-drift optimum
    assert frozen(P.ROOT) == "search_A"
    assert r2_preferences(questions, current, ranked_current, frozen) == []
    _, drifted, ranked_drifted = _toy(2, 1, n=3)
    prefs = r2_preferences(questions, drifted, ranked_drifted, frozen)
    assert len(prefs) == 3 and {p.state for p in prefs} == {P.ROOT}      # one consistent state-level preference
    assert {(p.frozen_action, p.preferred_action) for p in prefs} == {("search_A", "search_B")}
    assert all(p.outcome_changed and p.chosen == (("B",), "answer_B") for p in prefs)
    pairs = to_dpo_pairs(prefs, SourceTrustEnv(questions, drifted, ranked_drifted))
    assert [pair.chosen.episode_return for pair in pairs] == pytest.approx([p.chosen_return for p in prefs])
    assert np.array_equal(pairs[0].chosen.observations[0], encode(P.ROOT))


def test_r2_uses_only_the_questions_it_is_given():
    """Preferences come from the population passed in; labels of other questions cannot matter."""
    questions, drifted, ranked = _toy(2, 1, n=2)
    current_questions, current, ranked_current = _toy(1, 1, n=2)
    frozen = table_policy(solve(current_questions, current, ranked_current))
    alone = r2_preferences(questions[:1], drifted, ranked, frozen)
    assert [p.query_id for p in alone] == ["f0.q0"]


@pytest.fixture(scope="module")
def real():
    facts = load_facts(CORPUS / "facts.yaml")
    kb_a, kb_b = (load_knowledge_base(CORPUS, name) for name in ("kb_a", "kb_b"))
    pool = P.pool_facts(facts, kb_a, kb_b)
    return kb_a, kb_b, pool, P.pilot_questions(pool)


def test_group_split_keeps_units_together_and_covers_every_page(real):
    _, _, pool, questions = real
    train, test = group_split(pool)
    assert not train & test and train | test == set(pool)
    units = {}
    for fid, (fact, *_) in pool.items():
        units.setdefault(fact.unit, set()).add(fid in test)
    assert all(len(sides) == 1 for sides in units.values())
    pages = lambda fids: {pool[f][0].source for f in fids}
    assert pages(train) == pages(test)
    assert (sum(q.fact_id in train for q in questions), sum(q.fact_id in test for q in questions)) == (105, 101)
    assert group_split(pool) == (train, test)


def test_r2_guard_no_outcome_changing_pairs_at_zero_drift(real):
    """The diagnostic's guard: with the exact KB-A policy (fitted on train questions) as the frozen
    policy, R2 on KB-B at 0% drift gives no outcome-changing preference."""
    kb_a, kb_b, pool, questions = real
    train, _ = group_split(pool)
    train_q = [q for q in questions if q.fact_id in train]
    world_a = P.build_world(pool, kb_a, kb_b, "A", cache="mirror")
    ranked_a = P.rankings(world_a, train_q, Retriever)
    frozen = table_policy(solve(train_q, world_a, ranked_a))
    world_b = P.build_world(pool, kb_a, kb_b, "B", P.stale_facts(pool, 0.0, mode="page"), cache="mirror")
    prefs = r2_preferences(train_q, world_b, P.rankings(world_b, train_q, Retriever), frozen)
    assert not any(p.outcome_changed for p in prefs)
    assert len(prefs) <= 1
    stale = P.stale_facts(pool, 0.25, seed=0, mode="page")
    world_25 = P.build_world(pool, kb_a, kb_b, "B", stale, cache="mirror")
    assert sum(p.outcome_changed for p in
               r2_preferences(train_q, world_25, P.rankings(world_25, train_q, Retriever), frozen)) > 0
