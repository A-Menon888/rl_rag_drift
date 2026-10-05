"""R2: observation-compatible, state-level preference construction.

Validated in the R2 diagnostic (2026-09-30) and frozen for the source-trust
experiment. It replaces the per-question hindsight rule ("best strategy for q vs
the frozen trajectory for q"), which gives DPO information the policy cannot
observe.

For the KB-B *training* questions only:
1. Q(s, a) for every observable state s is solved exactly by backward
   induction over the training questions that reach s (the pilot's
   `solve_observable`): the expected return of a at s, followed by the
   training-optimal continuation, averaged over those questions.
2. Each training question follows the frozen policy's greedy trajectory. At the
   first state s where the frozen action is not Q-optimal, one pair is made:
   chosen = the same prefix, then the training-optimal greedy continuation;
   rejected = the frozen trajectory. Questions on which the frozen policy only
   takes Q-optimal actions give no pair.

Held-out questions, per-question optima and staleness labels are never used.
Because both the preferred and the frozen action depend only on s, the rule
cannot produce conflicting preferences for one state. Pairs are kept whether
or not they change the answer outcome, and even when the chosen continuation is
worse for that particular question (a state-level, not per-question, rule).
"""
from dataclasses import dataclass

from src.agents.dpo import PreferencePair, rollout
from src.environment.source_trust_env import ACTIONS, TERMINALS, VARIANT
from src.pilot import source_drift as P


def solve(questions, world, ranked) -> dict:
    """Exact observable-state Q table over `questions` (root key removed)."""
    table = P.solve_observable(questions, world, ranked, VARIANT)
    table.pop("__root__")
    return table


def table_policy(table):
    """Greedy state -> action of a Q table (pilot tie-break order)."""
    return lambda state: P.greedy(table[state]["best"], VARIANT)


def follow(act, question, ranked, seq=()) -> tuple:
    """(search sequence, terminal action) of a greedy state -> action policy from prefix `seq`."""
    while True:
        action = act(P.state_of(question, seq, ranked, VARIANT))
        if action in TERMINALS:
            return seq, action
        seq += (action.removeprefix("search_"),)


def action_indices(seq, end) -> tuple:
    return tuple(ACTIONS.index(P.SEARCH[s]) for s in seq) + (ACTIONS.index(end),)


@dataclass(frozen=True)
class StatePreference:
    query_id: str
    state: tuple                 # the observable state where the pair diverges
    preferred_action: str        # training-optimal action at `state`
    frozen_action: str
    chosen: tuple                # (search sequence, terminal action)
    rejected: tuple
    chosen_return: float
    rejected_return: float
    outcome_changed: bool        # (answered or gave up, correct) differs


def r2_preferences(train_questions, world, ranked, frozen_act, table=None) -> list[StatePreference]:
    """One preference per training question at most (see module docstring)."""
    table = table if table is not None else solve(train_questions, world, ranked)
    out = []
    for q in train_questions:
        seq = ()
        while True:
            state = P.state_of(q, seq, ranked, VARIANT)
            frozen = frozen_act(state)
            if frozen not in table[state]["best"]:
                chosen = follow(table_policy(table), q, ranked, seq)
                rejected = follow(frozen_act, q, ranked)
                out.append(StatePreference(
                    q.query_id, state, P.greedy(table[state]["best"], VARIANT), frozen, chosen, rejected,
                    P.strategy_return(q, world, ranked, *chosen), P.strategy_return(q, world, ranked, *rejected),
                    P.outcome(q, world, ranked, *chosen) != P.outcome(q, world, ranked, *rejected)))
                break
            if frozen in TERMINALS:
                break
            seq += (frozen.removeprefix("search_"),)
    return out


def to_dpo_pairs(preferences, env) -> list[PreferencePair]:
    """Replay both sides in `env` (the KB-B training env) to record the policy's observations."""
    index = {q.query_id: i for i, q in enumerate(env.queries)}
    return [PreferencePair(p.query_id, rollout(env, index[p.query_id], action_indices(*p.chosen)),
                           rollout(env, index[p.query_id], action_indices(*p.rejected)))
            for p in preferences]
