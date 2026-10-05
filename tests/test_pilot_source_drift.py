"""Source-reliability drift pilot: source assignment, drift generation, observable states."""
from pathlib import Path

import pytest

from src.data.documents import load_knowledge_base
from src.data.facts import extract_values, load_facts
from src.pilot.source_drift import (ANSWER, GIVE_UP, ROOT, Entry, PilotQuestion, Variant, World, build_world, compare,
                                    evidence, observe, pool_facts, reader_answer, solve_observable,
                                    stale_facts, state_of)

CORPUS = Path(__file__).parents[1] / "data" / "documentation"


@pytest.fixture(scope="module")
def real():
    facts = load_facts(CORPUS / "facts.yaml")
    kb_a, kb_b = (load_knowledge_base(CORPUS, name) for name in ("kb_a", "kb_b"))
    return kb_a, kb_b, pool_facts(facts, kb_a, kb_b)


def _cache_values(world, pool):
    by_fact = {}
    for fid, (fact, *_) in pool.items():
        by_fact[fid] = {v for e in world.entries["A"] for v in extract_values(fact.pattern, e.text)}
    return by_fact


# ── Source assignment ──────────────────────────────────────────────────────

def test_pool_holds_only_facts_stated_in_both_snapshots(real):
    _, _, pool = real
    assert {kind for _, kind, *_ in pool.values()} == {"unchanged", "modified", "contradicted"}
    assert all(a is not None and b is not None for _, _, a, b in pool.values())


def test_sources_cover_every_pool_fact_and_the_faq_is_not_a_source(real):
    kb_a, kb_b, pool = real
    world = build_world(pool, kb_a, kb_b, "A")
    assert len(world.entries["A"]) == len(pool)
    assert all(e.source == "A" for e in world.entries["A"]) and all(e.source == "B" for e in world.entries["B"])
    assert "faq.md" not in {e.page for e in world.entries["B"]}
    for fid, values in _cache_values(world, pool).items():
        assert values == {world.truth[fid]}, fid   # each fact stated once in the cache, correctly
        (fact, *_) = pool[fid]
        assert {v for e in world.entries["B"] if e.page == fact.source
                for v in extract_values(fact.pattern, e.text)} == {world.truth[fid]}


# ── Drift generation ───────────────────────────────────────────────────────

def test_staleness_is_nested_deterministic_and_limited_to_changed_facts(real):
    _, _, pool = real
    changed = {fid for fid, (_, _, a, b) in pool.items() if a != b}
    levels = [stale_facts(pool, p) for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert levels[0] == frozenset() and levels[-1] == changed
    assert [len(s) for s in levels] == [round(p * len(changed)) for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(small <= large for small, large in zip(levels, levels[1:]))
    assert stale_facts(pool, 0.5) == stale_facts(pool, 0.5) != stale_facts(pool, 0.5, seed=1)


def test_stale_cache_entries_state_the_old_value_and_reference_stays_current(real):
    kb_a, kb_b, pool = real
    stale = stale_facts(pool, 0.5)
    world = build_world(pool, kb_a, kb_b, "B", stale)
    cache = _cache_values(world, pool)
    for fid, (fact, _, a, b) in pool.items():
        assert world.truth[fid] == b
        assert cache[fid] == ({a} if fid in stale else {b}), fid
    clean = build_world(pool, kb_a, kb_b, "B")
    assert all(values == {clean.truth[fid]} for fid, values in _cache_values(clean, pool).items())


# ── Observable states and exact solution (toy world) ──────────────────────

def _toy(truth_value, cache_value):
    q = PilotQuestion("f.q0", "f", "what is x?", r"x is (\d)")
    filler = Entry("unrelated text", "A", "p")
    ranked = {("f.q0", "A"): [Entry(f"x is {cache_value}", "A", "p"), filler, filler],
              ("f.q0", "B"): [Entry(f"x is {truth_value}", "B", "p"), filler, filler]}
    return [q], World("toy", {"f": str(truth_value)}, {}, frozenset()), ranked


def test_observation_holds_only_search_history_and_agreement_flags():
    (q,), _, ranked = _toy(2, 1)
    found = evidence(q, ("A", "B"), ranked)
    assert found == [("A", "1"), ("B", "2")]
    assert observe(("A", "B"), found) == (("A", "B"), True, True, True)
    assert observe(("B",), evidence(q, ("B",), ranked)) == (("B",), False, True, False)
    assert observe((), []) == ROOT
    assert reader_answer(found) == "2" and reader_answer([]) is None


def test_mirror_cache_matches_the_reference_except_stale_values(real):
    kb_a, kb_b, pool = real
    current = build_world(pool, kb_a, kb_b, "A", cache="mirror")
    assert [e.text for e in current.entries["A"]] == [e.text for e in current.entries["B"]]
    stale = stale_facts(pool, 0.5)
    drifted = build_world(pool, kb_a, kb_b, "B", stale, cache="mirror")
    assert len(drifted.entries["A"]) == len(drifted.entries["B"])
    differing = sum(a.text != b.text for a, b in zip(drifted.entries["A"], drifted.entries["B"]))
    assert 0 < differing <= len(stale)
    for fid, values in _cache_values(drifted, pool).items():
        _, _, a, b = pool[fid]
        assert values == ({a} if fid in stale else {b}), fid


def test_page_staleness_fills_whole_pages_in_a_nested_order(real):
    _, _, pool = real
    changed_pages = {}
    for fid, (fact, _, a, b) in pool.items():
        if a != b:
            changed_pages.setdefault(fact.source, set()).add(fid)
    levels = [stale_facts(pool, p, mode="page") for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(small <= large for small, large in zip(levels, levels[1:]))
    assert levels[-1] == set().union(*changed_pages.values())
    for stale in levels:
        partial = [page for page, fids in changed_pages.items() if 0 < len(fids & stale) < len(fids)]
        assert len(partial) <= 1   # at most one page is partly stale


def test_choose_reader_answers_with_the_selected_source_and_page_feature_is_optional():
    (q,), world, ranked = _toy(2, 1)
    found = evidence(q, ("B", "A"), ranked)
    assert reader_answer(found) == "1"                       # last retrieved wins
    assert reader_answer(found, "answer_B") == "2" and reader_answer(found, "answer_A") == "1"
    assert reader_answer(evidence(q, ("A",), ranked), "answer_B") is None
    choose = Variant(reader="choose")
    table = solve_observable([q], world, ranked, choose)
    assert table[(("B", "A"), True, True, True)]["best"] == {"answer_B"}
    assert state_of(q, ("A", "B"), ranked, Variant(page_feature="first"))[-1] == ("p",)
    assert len(state_of(q, ("A", "B"), ranked, Variant(page_feature="steps"))[-1]) == 2
    assert state_of(q, ("A", "B"), ranked) == observe(("A", "B"), found[::-1])


def test_exact_solution_switches_source_only_when_the_cache_is_stale():
    questions, current, ranked_current = _toy(1, 1)
    table = solve_observable(questions, current, ranked_current)
    assert table[ROOT]["best"] == {"search_A"}                      # cheaper source, equally right
    assert table[(("A",), True, False, False)]["best"] == {ANSWER}
    _, drifted, ranked_drifted = _toy(2, 1)
    result = compare(questions, current, ranked_current, drifted, ranked_drifted)
    assert solve_observable(questions, drifted, ranked_drifted)[ROOT]["best"] == {"search_B"}
    assert result["states_optimal_action_changed"] >= 2             # root and "cache answered"
    assert result["preference_pairs"] == 1 and result["questions_optimum_changed"] == 1
    no_drift = compare(questions, current, ranked_current, current, ranked_current)
    assert no_drift["states_optimal_action_changed"] == 0 and no_drift["preference_pairs"] == 0
    assert GIVE_UP in table[ROOT]["q"]
