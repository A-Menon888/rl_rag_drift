"""Isolated diagnostic pilot: does source-reliability drift change the optimal
retrieval action in states the policy can observe?

Nothing here is used by the main experiment. No policy is trained: every
retrieval strategy is enumerated and the optimal policy is computed exactly.

Setup
- Pool: facts documented in both KB-A and KB-B (unchanged, modified,
  contradicted). Added/removed/absent facts are left out: they have no stale
  version, so they cannot show source-reliability drift.
- Source B ("reference", expensive): every chunk of the reference pages
  (the FAQ page is excluded; the cache plays its role).
- Source A ("cache", cheap), two constructions (`Variant.cache`):
  - "sentence": one entry per pool fact, the sentence stating it;
  - "mirror": a copy of Source B's chunks with stale values substituted, so
    both sources have the same retrieval quality by construction.
- Pilot KB-A: truth = KB-A value; both sources current.
- Pilot KB-B at drift level p: truth = KB-B value; Source B current; a nested
  seeded fraction p of the *changed* facts (modified + contradicted) are stale
  in the cache (still state the KB-A value). `Variant.staleness`: "random"
  facts, or "page" (whole pages in a seeded page order).
- Reader (`Variant.reader`): "last" = the most recently retrieved statement
  wins (one ANSWER action); "choose" = ANSWER_A / ANSWER_B answer with the
  value the chosen source stated. No dates and no built-in source priority.
- No closed-book memory: without evidence the reader says "don't know".
- Observation: the ordered sources searched, whether A / B stated the answer
  slot, whether they disagree; `Variant.page_feature` optionally adds page
  identity: "steps" = the page of every retrieved entry, "first" = only the
  page of the first retrieved entry (one categorical value).
"""
from dataclasses import dataclass
from itertools import product
import random
import re

from src.data.facts import drift_type, extract_values, fact_value

SOURCES = ("A", "B")
SEARCH = {"A": "search_A", "B": "search_B"}
ANSWER, GIVE_UP = "answer", "give_up"
ANSWER_FROM = {"A": "answer_A", "B": "answer_B"}
COSTS = {"A": 0.05, "B": 0.10}   # cache is half the price of the reference search
CORRECT, INCORRECT, ABSTAIN = 1.0, -1.0, 0.0
MAX_SEARCHES = 3
DRIFT_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
TOL = 1e-9
ROOT = ((), False, False, False)   # observation before any search (no page feature)


@dataclass(frozen=True)
class Variant:
    cache: str = "sentence"        # "sentence" | "mirror"
    reader: str = "last"           # "last" | "choose"
    staleness: str = "random"      # "random" | "page"
    page_feature: str = "none"    # "none" | "first" | "steps"

    @property
    def terminals(self) -> tuple:
        answers = (ANSWER,) if self.reader == "last" else (ANSWER_FROM["A"], ANSWER_FROM["B"])
        return (*answers, GIVE_UP)

    @property
    def actions(self) -> tuple:          # also the greedy tie-break order
        return (*self.terminals, SEARCH["A"], SEARCH["B"])


DEFAULT = Variant()


@dataclass(frozen=True)
class Entry:
    text: str
    source: str            # "A" (cache) or "B" (reference)
    page: str


@dataclass(frozen=True)
class PilotQuestion:
    query_id: str
    fact_id: str
    text: str
    pattern: str


@dataclass(frozen=True)
class World:
    label: str
    truth: dict            # fact_id -> gold value
    entries: dict          # source -> list[Entry]
    stale: frozenset       # fact ids whose cache statement is stale


# ── Construction ────────────────────────────────────────────────────────────

def pool_facts(facts, kb_a, kb_b):
    """Facts stated in both snapshots, with their drift type."""
    pool = {}
    for fact in facts:
        a, b = fact_value(fact, kb_a), fact_value(fact, kb_b)
        if a.value is not None and b.value is not None:
            pool[fact.fact_id] = (fact, drift_type(a, b), a.value, b.value)
    return pool


def statement_sentence(pattern: str, text: str) -> str:
    """The sentence of a chunk that states the fact (the whole regex match)."""
    match = re.search(pattern, text)
    left = text.rfind(". ", 0, match.start())
    left = 0 if left < 0 else left + 2
    if text[match.end() - 1] == ".":
        right = match.end()
    else:
        right = text.find(". ", match.end())
        right = len(text) if right < 0 else right + 1
    return text[left:right].strip()


def _owning_chunk(fact, chunks):
    return next(c for c in chunks if c.source == fact.source and extract_values(fact.pattern, c.text))


def cache_entry(fact, chunks) -> Entry:
    return Entry(statement_sentence(fact.pattern, _owning_chunk(fact, chunks).text), "A", fact.source)


def mirror_entries(pool, kb_a, current, stale) -> list[Entry]:
    """Source-B chunks with each stale fact's value replaced by its raw KB-A value."""
    texts = {id(c): c.text for c in current}
    for fid in sorted(stale):
        fact = pool[fid][0]
        old = re.search(fact.pattern, _owning_chunk(fact, kb_a).text).group(1)
        chunk = _owning_chunk(fact, current)
        match = re.search(fact.pattern, texts[id(chunk)])
        texts[id(chunk)] = texts[id(chunk)][:match.start(1)] + old + texts[id(chunk)][match.end(1):]
    return [Entry(texts[id(c)], "A", c.source) for c in current if c.source != "faq.md"]


def stale_facts(pool, level: float, seed: int = 0, mode: str = "random") -> frozenset:
    """Nested staleness: the first round(level * n) changed facts of one seeded order.

    "random" shuffles facts; "page" shuffles pages and lists each page's changed
    facts together, so staleness fills whole pages (one page may be partial).
    """
    changed = sorted(fid for fid, (_, kind, a, b) in pool.items() if a != b)
    rng = random.Random(seed)
    if mode == "random":
        rng.shuffle(changed)
    elif mode == "page":
        pages = sorted({pool[fid][0].source for fid in changed})
        rng.shuffle(pages)
        changed = [fid for page in pages for fid in changed if pool[fid][0].source == page]
    else:
        raise ValueError(mode)
    return frozenset(changed[:round(level * len(changed))])


def build_world(pool, kb_a, kb_b, label: str, stale=frozenset(), cache: str = "sentence") -> World:
    """label 'A': both sources current (KB-A); label 'B': KB-B truth, `stale` cache statements."""
    current = kb_a if label == "A" else kb_b
    truth = {fid: (a if label == "A" else b) for fid, (_, _, a, b) in pool.items()}
    if cache == "sentence":
        source_a = [cache_entry(fact, kb_a if fid in stale else current) for fid, (fact, *_) in sorted(pool.items())]
    elif cache == "mirror":
        source_a = mirror_entries(pool, kb_a, current, stale)
    else:
        raise ValueError(cache)
    reference = [Entry(c.text, "B", c.source) for c in current if c.source != "faq.md"]
    return World(label, truth, {"A": source_a, "B": reference}, frozenset(stale))


def pilot_questions(pool) -> list[PilotQuestion]:
    return [PilotQuestion(f"{fid}.q{i}", fid, q, fact.pattern)
            for fid, (fact, *_) in sorted(pool.items()) for i, q in enumerate(fact.questions)]


# ── Simulation and observation ──────────────────────────────────────────────

def rankings(world: World, questions, retriever_factory) -> dict:
    """(query_id, source) -> top MAX_SEARCHES entries of that source for the question."""
    out = {}
    for source in SOURCES:
        retriever = retriever_factory(world.entries[source])
        for q in questions:
            out[(q.query_id, source)] = [r.fact for r in retriever.search(q.text, MAX_SEARCHES)]
    return out


def sequences():
    """Every ordered list of sources of length 0..MAX_SEARCHES."""
    return [seq for n in range(MAX_SEARCHES + 1) for seq in product(SOURCES, repeat=n)]


def retrieved(question, seq, ranked) -> list[Entry]:
    used, out = {s: 0 for s in SOURCES}, []
    for source in seq:
        out.append(ranked[(question.query_id, source)][used[source]])
        used[source] += 1
    return out


def statements(question, entries) -> list[tuple[str, str]]:
    return [(e.source, v) for e in entries for v in extract_values(question.pattern, e.text)]


def evidence(question, seq, ranked) -> list[tuple[str, str]]:
    """(source, value) for each slot statement revealed by searching `seq`, in order."""
    return statements(question, retrieved(question, seq, ranked))


def observe(seq, found, pages=None) -> tuple:
    """Everything the policy may see: the sources searched so far (in order),
    whether each source has stated the answer slot, whether the two sources
    disagree, and (page feature only) the page of each retrieved entry. No
    question identity, values, dates or scores."""
    a = {v for s, v in found if s == "A"}
    b = {v for s, v in found if s == "B"}
    state = (tuple(seq), bool(a), bool(b), bool(a) and bool(b) and a != b)
    return state if pages is None else (*state, tuple(pages))


def state_of(question, seq, ranked, variant=DEFAULT) -> tuple:
    entries = retrieved(question, seq, ranked)
    pages = {"none": None, "first": [e.page for e in entries[:1]],
             "steps": [e.page for e in entries]}[variant.page_feature]
    return observe(seq, statements(question, entries), pages)


def reader_answer(found, end=ANSWER):
    """ANSWER: the most recently retrieved statement wins. ANSWER_A/B: the latest
    statement from that source. None = don't know."""
    if end != ANSWER:
        found = [(s, v) for s, v in found if ANSWER_FROM[s] == end]
    return found[-1][1] if found else None


def terminal_reward(action, answer, gold) -> float:
    if action == GIVE_UP:
        return ABSTAIN          # every pilot question is answerable
    return CORRECT if answer is not None and answer == gold else INCORRECT


def search_cost(seq) -> float:
    return sum(COSTS[s] for s in seq)


def strategies(variant=DEFAULT):
    """All enumerated strategies: a search sequence, then a terminal action."""
    return [(seq, end) for seq in sequences() for end in variant.terminals]


def _end_reward(question, world, found, end) -> float:
    return terminal_reward(end, reader_answer(found, end) if end != GIVE_UP else None, world.truth[question.fact_id])


def strategy_return(question, world, ranked, seq, end) -> float:
    return _end_reward(question, world, evidence(question, seq, ranked), end) - search_cost(seq)


def outcome(question, world, ranked, seq, end) -> tuple:
    """(answered or gave up, correct): which source answered does not count."""
    reward = _end_reward(question, world, evidence(question, seq, ranked), end)
    return (GIVE_UP if end == GIVE_UP else ANSWER), reward > 0


# ── Exact optimal policies ──────────────────────────────────────────────────

def question_optima(questions, world, ranked, variant=DEFAULT) -> dict:
    """query_id -> (best return, set of optimal strategies)."""
    out = {}
    for q in questions:
        returns = {s: strategy_return(q, world, ranked, *s) for s in strategies(variant)}
        best = max(returns.values())
        out[q.query_id] = (best, {s for s, r in returns.items() if r >= best - TOL})
    return out


def solve_observable(questions, world, ranked, variant=DEFAULT) -> dict:
    """Optimal policy over observable states, by backward induction.

    Each state is the set of questions that produce that observation; the
    observation contains the search sequence, so the states form a tree and
    the induction is exact. Returns state -> {"n", "value", "q" (action ->
    value), "best" (optimal action set)}.
    """
    table = {}

    def visit(seq, members):
        groups = {}
        for q in members:
            groups.setdefault(state_of(q, seq, ranked, variant), []).append(q)
        return groups

    def solve(state, members):
        seq = state[0]
        q_values = {end: sum(_end_reward(q, world, evidence(q, seq, ranked), end) for q in members) / len(members)
                    for end in variant.terminals}
        if len(seq) < MAX_SEARCHES:
            for source in SOURCES:
                children = visit(seq + (source,), members)
                q_values[SEARCH[source]] = -COSTS[source] + sum(
                    len(child) * solve(key, child) for key, child in children.items()) / len(members)
        value = max(q_values.values())
        table[state] = {"n": len(members), "value": value, "q": q_values,
                        "best": {a for a, v in q_values.items() if v >= value - TOL}}
        return value

    (root, members), = visit((), list(questions)).items()
    solve(root, members)
    table["__root__"] = root
    return table


def greedy(best: set, variant=DEFAULT) -> str:
    return next(a for a in variant.actions if a in best)


def run_policy(policy_table, question, world, ranked, variant=DEFAULT):
    """Follow a state table's greedy action; states it never saw answer (or answer_A).

    Returns (seq, end, return, visited states, number of unseen states met).
    """
    seq, visited, unseen = (), [], 0
    while True:
        state = state_of(question, seq, ranked, variant)
        visited.append(state)
        if state in policy_table:
            action = greedy(policy_table[state]["best"], variant)
        else:
            unseen += 1
            action = variant.terminals[0]
        if action in variant.terminals:
            return seq, action, strategy_return(question, world, ranked, seq, action), visited, unseen
        seq += (action.removeprefix("search_"),)


# ── Diagnostic ──────────────────────────────────────────────────────────────

def compare(questions, world_a, ranked_a, world_b, ranked_b, variant=DEFAULT) -> dict:
    """Question-level and state-level adaptation counts between two worlds."""
    opt_a = question_optima(questions, world_a, ranked_a, variant)
    opt_b = question_optima(questions, world_b, ranked_b, variant)
    table_a = solve_observable(questions, world_a, ranked_a, variant)
    table_b = solve_observable(questions, world_b, ranked_b, variant)
    root_a, root_b = table_a.pop("__root__"), table_b.pop("__root__")
    shared = set(table_a) & set(table_b)
    changed_states = {s for s in shared if not table_a[s]["best"] & table_b[s]["best"]}

    frozen, adapted, pairs, cost_only, affected, unseen = [], [], 0, 0, 0, 0
    stale_q = [q for q in questions if q.fact_id in world_b.stale]
    for q in questions:
        seq, end, ret, visited, n_unseen = run_policy(table_a, q, world_b, ranked_b, variant)
        frozen.append(ret)
        unseen += n_unseen > 0
        adapted.append(run_policy(table_b, q, world_b, ranked_b, variant)[2])
        affected += any(state in changed_states for state in visited)
        best_return, best_set = opt_b[q.query_id]
        chosen = next(s for s in strategies(variant) if s in best_set)   # first in enumeration order
        if best_return > ret + TOL:
            if outcome(q, world_b, ranked_b, *chosen) != outcome(q, world_b, ranked_b, seq, end):
                pairs += 1
            else:
                cost_only += 1
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    by_id = {q.query_id: i for i, q in enumerate(questions)}
    oracle = [opt_b[q.query_id][0] for q in questions]
    return {
        "questions": len(questions),
        "stale_questions": len(stale_q),
        "questions_optimum_changed": sum(not opt_a[q.query_id][1] & opt_b[q.query_id][1] for q in questions),
        "states_a": len(table_a), "states_b": len(table_b), "states_shared": len(shared),
        "states_new_after_drift": len(set(table_b) - set(table_a)),
        "states_optimal_action_changed": len(changed_states),
        "questions_through_changed_state_under_frozen_policy": affected,
        "questions_meeting_unseen_state_under_frozen_policy": unseen,
        "changed_states": sorted((s, sorted(table_a[s]["best"]), sorted(table_b[s]["best"]), table_b[s]["n"])
                                 for s in changed_states),
        "return_frozen_on_a": table_a[root_a]["value"],
        "return_frozen_on_b": mean(frozen),
        "return_adapted_on_b": table_b[root_b]["value"],
        "return_oracle_on_b": mean(oracle),
        "observable_gap": mean(oracle) - table_b[root_b]["value"],
        # Adapted-vs-oracle gap split by whether the question's cache statement is stale.
        "observable_gap_stale": mean([oracle[by_id[q.query_id]] - adapted[by_id[q.query_id]] for q in stale_q]),
        "observable_gap_fresh": mean([oracle[i] - adapted[i] for i, q in enumerate(questions)
                                      if q.fact_id not in world_b.stale]),
        "preference_pairs": pairs,
        "cost_only_pairs": cost_only,
    }


def retrieval_quality(questions, ranked) -> dict:
    """Share of questions whose top-1 / top-3 entry of each source states the answer slot."""
    out = {}
    for source in SOURCES:
        hits = [next((i + 1 for i, e in enumerate(ranked[(q.query_id, source)])
                      if extract_values(q.pattern, e.text)), 0) for q in questions]
        out[source] = {"top1": sum(h == 1 for h in hits) / len(hits), "top3": sum(h > 0 for h in hits) / len(hits)}
    return out
