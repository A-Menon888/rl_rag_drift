"""Source-reliability drift experiment: frozen vs DPO (R2 preferences) vs RL adaptation.

Stage 1: REINFORCE on KB-A train questions (no drift), then frozen.
Stage 2: frozen policy evaluated on KB-B at each drift level x page seed.
Stage 3: DPO from the frozen checkpoint on R2 state-level preferences (KB-B train).
Stage 4: REINFORCE fine-tuning from the frozen checkpoint, same environment-episode budget.
References: held-out observable-state optimum (exact), train-fitted exact policy, per-question oracle.

One process per training seed; each writes runs_seed<k>.jsonl under --output-root.
Usage: python experiments/run_source_trust.py --output-root <dir> [--jobs 5]
       python experiments/analyze_source_trust.py --output-root <dir>
"""
import argparse
import collections
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.dpo import train_dpo
from src.agents.rl_agent import RLAgent
from src.agents.state_preferences import r2_preferences, solve, table_policy, to_dpo_pairs
from src.data.documents import load_knowledge_base
from src.data.facts import load_facts
from src.environment.source_trust_env import (ACTIONS, OBS_SIZE, TERMINALS, VARIANT, SourceTrustEnv, action_mask,
                                              encode, group_split)
from src.pilot import source_drift as P
from src.retrieval.retriever import Retriever

KEY_STATES = {   # observable states reported for every policy
    "root": ((), False, False, False),
    "A_answered": (("A",), True, False, False),
    "A_silent": (("A",), False, False, False),
    "B_answered": (("B",), False, True, False),
    "AB_disagree": (("A", "B"), True, True, True),
}


def mlp_act(agent):
    return lambda state: ACTIONS[agent.act(encode(state), {"action_mask": action_mask(state[0])}, explore=False)]


def fallback_act(table, fallback=TERMINALS[0]):
    """Table policy; states the table never saw answer from the cache (as in the R2 diagnostic)."""
    act = table_policy(table)
    return lambda state: act(state) if state in table else fallback


def evaluate(act, env) -> dict:
    """Greedy episode on every question of `env` with a state -> action-name policy."""
    infos = []
    for i in range(len(env.queries)):
        env.reset(options={"query_index": i})
        done = False
        while not done:
            _, _, done, _, info = env.step(ACTIONS.index(act(env.state())))
        infos.append(info)
    n = len(infos)
    mean = lambda key: sum(float(x[key]) for x in infos) / n
    first = collections.Counter(("search_" + x["seq"][0]) if x["seq"] else x["final_action"] for x in infos)
    final = collections.Counter(x["final_action"] for x in infos)
    stale = [x for x in infos if x["cache_stale"]]
    fresh = [x for x in infos if not x["cache_stale"]]
    return {
        "n": n, "accuracy": mean("correct"), "return": mean("episode_return"), "searches": mean("searches"),
        "searches_a": mean("searches_a"), "searches_b": mean("searches_b"), "retrieval_cost": mean("retrieval_cost"),
        "first_action": {a: first[a] / n for a in sorted(first)},
        "final_action": {a: final[a] / n for a in sorted(final)},
        "stale_cache_questions": len(stale),
        "accuracy_stale_cache": sum(x["correct"] for x in stale) / len(stale) if stale else None,
        "accuracy_fresh_cache": sum(x["correct"] for x in fresh) / len(fresh) if fresh else None,
        "per_question": {x["query_id"]: [x["episode_return"], x["final_action"], "".join(x["seq"])] for x in infos},
    }


def key_actions(act, table=None) -> dict:
    """Greedy action at each key state; None where `table` has no question reaching it."""
    return {name: act(state) if table is None or state in table else None for name, state in KEY_STATES.items()}


def root_probs(agent) -> dict:
    out = {}
    for name in ("root", "A_answered"):
        state = KEY_STATES[name]
        with torch.no_grad():
            probs = agent._distribution(encode(state), action_mask(state[0])).probs.tolist()
        out[name] = {a: round(p, 4) for a, p in zip(ACTIONS, probs)}
    return out


def reinforce(agent, env, epochs, batch_size, seed) -> dict:
    """Same loop as run_all.train_policy: seeded permutation per epoch, one update per batch."""
    rng = np.random.default_rng(seed)
    budget = {"episodes": 0, "gradient_updates": 0, "policy_step_evaluations": 0, "env_search_calls": 0}
    curve = []
    for _ in range(epochs):
        order = rng.permutation(len(env.queries))
        returns = []
        for start in range(0, len(order), batch_size):
            infos = agent.train_batch(env, order[start:start + batch_size].tolist())
            budget["gradient_updates"] += 1
            budget["episodes"] += len(infos)
            budget["policy_step_evaluations"] += sum(x["searches"] + 1 for x in infos)
            budget["env_search_calls"] += sum(x["searches"] for x in infos)
            returns += [x["episode_return"] for x in infos]
        curve.append(float(np.mean(returns)))
    budget["curve_first_last"] = [curve[0], curve[-1]]
    return budget


def new_agent(config, seed, checkpoint=None):
    agent = RLAgent(OBS_SIZE, config["learning_rate"], seed, entropy_coef=config["entropy_coef"], actions=len(ACTIONS))
    return agent.load(checkpoint) if checkpoint else agent


def setup(config):
    corpus = ROOT / config["corpus_dir"]
    kb_a, kb_b = (load_knowledge_base(corpus, name) for name in ("kb_a", "kb_b"))
    pool = P.pool_facts(load_facts(ROOT / config["facts_path"]), kb_a, kb_b)
    questions = P.pilot_questions(pool)
    train_f, test_f = group_split(pool, config["test_fraction"], config["split_seed"])
    train_q = [q for q in questions if q.fact_id in train_f]
    test_q = [q for q in questions if q.fact_id in test_f]
    assert not {q.fact_id for q in train_q} & {q.fact_id for q in test_q}
    return kb_a, kb_b, pool, questions, train_q, test_q


def pref_stats(prefs, t_test) -> dict:
    states = collections.Counter(p.state for p in prefs)
    supported = [p for p in prefs if p.state in t_test]
    return {
        "pairs": len(prefs),
        "outcome_changing": sum(p.outcome_changed for p in prefs),
        "individually_worse": sum(p.chosen_return < p.rejected_return - P.TOL for p in prefs),
        "pair_states": {repr(s): n for s, n in states.items()},
        "preferred_actions": dict(collections.Counter(f"{p.frozen_action}->{p.preferred_action}" for p in prefs)),
        "at_root": sum(p.state == KEY_STATES["root"] for p in prefs),
        "heldout_supported": len(supported),
        "heldout_match": sum(p.preferred_action in t_test[p.state]["best"] for p in supported),
    }


def run_seed(args):
    config, seed, out_dir = args
    out_dir = Path(out_dir)
    kb_a, kb_b, pool, questions, train_q, test_q = setup(config)
    log = open(out_dir / f"runs_seed{seed}.jsonl", "w", encoding="utf-8")
    write = lambda record: (log.write(json.dumps(record, default=str) + "\n"), log.flush())

    # Stage 1: KB-A
    world_a = P.build_world(pool, kb_a, kb_b, "A", cache=VARIANT.cache)
    ranked_a = P.rankings(world_a, questions, Retriever)
    env_a_train, env_a_test = SourceTrustEnv(train_q, world_a, ranked_a), SourceTrustEnv(test_q, world_a, ranked_a)
    frozen = new_agent(config, seed)
    t0 = time.perf_counter()
    stage1_budget = reinforce(frozen, env_a_train, config["train_epochs"], config["batch_size"], seed)
    stage1_budget["wall_seconds"] = time.perf_counter() - t0
    checkpoint = out_dir / f"frozen_kb_a_seed{seed}.pt"
    frozen.save(checkpoint)
    frozen_act = mlp_act(frozen)
    t_a_train, t_a_test = solve(train_q, world_a, ranked_a), solve(test_q, world_a, ranked_a)
    prefs_a = r2_preferences(train_q, world_a, ranked_a, frozen_act, t_a_train)   # no drift at all
    write({"kind": "stage1", "train_seed": seed, "budget": stage1_budget,
           "kb_a_train": evaluate(frozen_act, env_a_train), "kb_a_test": evaluate(frozen_act, env_a_test),
           "kb_a_test_optimum": evaluate(table_policy(t_a_test), env_a_test),
           "frozen_key_actions": key_actions(frozen_act), "frozen_probs": root_probs(frozen),
           "r2_on_kb_a": pref_stats(prefs_a, t_a_test)})

    n_strategies = len(P.strategies(VARIANT))
    rl_epochs = n_strategies   # RL episodes = DPO enumeration episodes (45 per train question)
    worlds = {}
    for level in config["drift_levels"]:
        for page_seed in config["page_seeds"]:
            stale = P.stale_facts(pool, level, seed=page_seed, mode="page")
            if stale not in worlds:
                w = P.build_world(pool, kb_a, kb_b, "B", stale, cache=VARIANT.cache)
                worlds[stale] = (w, P.rankings(w, questions, Retriever))
            world_b, ranked_b = worlds[stale]
            env_train, env_test = SourceTrustEnv(train_q, world_b, ranked_b), SourceTrustEnv(test_q, world_b, ranked_b)
            t_train, t_test = solve(train_q, world_b, ranked_b), solve(test_q, world_b, ranked_b)
            oracle = P.question_optima(test_q, world_b, ranked_b, VARIANT)
            record = {"kind": "cell", "train_seed": seed, "level": level, "page_seed": page_seed,
                      "stale_facts": len(stale), "stale_pages": sorted({pool[f][0].source for f in stale}),
                      "stale_test_questions": sum(q.fact_id in stale for q in test_q)}
            # references (independent of the training seed)
            record["heldout_optimum"] = evaluate(table_policy(t_test), env_test)
            record["heldout_optimum_key_actions"] = key_actions(table_policy(t_test), t_test)
            record["train_fitted_exact"] = evaluate(fallback_act(t_train), env_test)
            record["train_fitted_exact_key_actions"] = key_actions(table_policy(t_train), t_train)
            record["oracle_return"] = float(np.mean([v[0] for v in oracle.values()]))
            record["train_optimum_return"] = t_train[KEY_STATES["root"]]["value"]
            # Stage 2: frozen
            record["frozen"] = {"test": evaluate(frozen_act, env_test), "train": evaluate(frozen_act, env_train)}
            # Stage 3: DPO on R2 preferences
            t0 = time.perf_counter()
            prefs = r2_preferences(train_q, world_b, ranked_b, frozen_act, t_train)
            pair_seconds = time.perf_counter() - t0
            dpo = new_agent(config, seed, checkpoint)
            dpo_budget = {"episodes": n_strategies * len(train_q), "env_search_calls_enumerated":
                          sum(len(s) for s, _ in P.strategies(VARIANT)) * len(train_q),
                          "unique_search_nodes": (2 + 4 + 8) * len(train_q), "pairs": len(prefs),
                          "pair_construction_seconds": pair_seconds}
            if prefs:
                reference = new_agent(config, seed, checkpoint)
                t0 = time.perf_counter()
                stats = train_dpo(dpo, reference, to_dpo_pairs(prefs, env_train), epochs=config["dpo_epochs"],
                                  batch_size=config["dpo_batch_size"], beta=config["dpo_beta"],
                                  learning_rate=config["dpo_learning_rate"], seed=seed)
                dpo_budget.update(gradient_updates=stats["gradient_updates"],
                                  policy_step_evaluations=stats["policy_step_evaluations"],
                                  loss_first_last=[stats["loss_curve"][0], stats["loss_curve"][-1]],
                                  train_seconds=time.perf_counter() - t0)
            else:   # no preference signal: the policy is left unchanged
                dpo_budget.update(gradient_updates=0, policy_step_evaluations=0, train_seconds=0.0)
            dpo_act = mlp_act(dpo)
            record["r2"] = pref_stats(prefs, t_test)
            record["dpo"] = {"test": evaluate(dpo_act, env_test), "train": evaluate(dpo_act, env_train),
                             "budget": dpo_budget, "key_actions": key_actions(dpo_act), "probs": root_probs(dpo),
                             "pair_states_now_preferred": sum(dpo_act(p.state) == p.preferred_action for p in prefs)}
            # Stage 4: RL fine-tuning, same episode budget
            rl = new_agent(config, seed, checkpoint)
            t0 = time.perf_counter()
            rl_budget = reinforce(rl, env_train, rl_epochs, config["batch_size"], seed)
            rl_budget["train_seconds"] = time.perf_counter() - t0
            assert rl_budget["episodes"] == dpo_budget["episodes"]
            rl_act = mlp_act(rl)
            record["rl"] = {"test": evaluate(rl_act, env_test), "train": evaluate(rl_act, env_train),
                            "budget": rl_budget, "key_actions": key_actions(rl_act), "probs": root_probs(rl),
                            "pair_states_now_preferred": sum(rl_act(p.state) == p.preferred_action for p in prefs)}
            record["frozen_key_actions"] = key_actions(frozen_act)
            write(record)
            print(f"seed {seed} level {level} page {page_seed}: frozen {record['frozen']['test']['return']:.3f} "
                  f"dpo {record['dpo']['test']['return']:.3f} rl {record['rl']['test']['return']:.3f} "
                  f"opt {record['heldout_optimum']['return']:.3f} pairs {len(prefs)}", flush=True)
    log.close()
    return seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "source_trust.yaml"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    jobs = [(config, seed, str(out)) for seed in config["train_seeds"]]
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            pool.map(run_seed, jobs)
    else:
        for job in jobs:
            run_seed(job)


if __name__ == "__main__":
    main()
