"""Source-reliability drift pilot (diagnostic only, no training). See src/pilot/source_drift.py.

Usage: python experiments/pilot_source_drift.py [--variants v0 v1 ...] [--output report.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.documents import load_knowledge_base
from src.data.facts import load_facts
from src.pilot.source_drift import (COSTS, DRIFT_LEVELS, Variant, build_world, compare, pilot_questions,
                                    pool_facts, rankings, retrieval_quality, stale_facts)
from src.retrieval.retriever import Retriever

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "v0_original": Variant(),
    "v1_mirror": Variant(cache="mirror"),
    "v2_mirror_choose": Variant(cache="mirror", reader="choose"),
    "v3_mirror_choose_page": Variant(cache="mirror", reader="choose", staleness="page"),
    "v4_mirror_choose_page_steps": Variant(cache="mirror", reader="choose", staleness="page", page_feature="steps"),
    "v5_mirror_choose_page_first": Variant(cache="mirror", reader="choose", staleness="page", page_feature="first"),
    "v6_mirror_choose_random_page_first": Variant(cache="mirror", reader="choose", staleness="random",
                                                  page_feature="first"),
}
SUMMARY = ("states_a", "stale_questions", "questions_optimum_changed", "states_shared", "states_new_after_drift",
           "states_optimal_action_changed", "questions_through_changed_state_under_frozen_policy",
           "return_frozen_on_b", "return_adapted_on_b", "return_oracle_on_b", "observable_gap",
           "observable_gap_stale", "observable_gap_fresh", "preference_pairs", "cost_only_pairs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS)[:4], choices=list(VARIANTS))
    parser.add_argument("--output", help="optional JSON report path (keep it out of results/)")
    args = parser.parse_args()
    corpus = ROOT / "data" / "documentation"
    kb_a, kb_b = (load_knowledge_base(corpus, name) for name in ("kb_a", "kb_b"))
    pool = pool_facts(load_facts(corpus / "facts.yaml"), kb_a, kb_b)
    questions = pilot_questions(pool)
    print(f"pool: {len(pool)} facts ({sum(a != b for *_, a, b in pool.values())} changed), "
          f"{len(questions)} questions; costs {COSTS}")
    report = {}
    for name in args.variants:
        variant = VARIANTS[name]
        world_a = build_world(pool, kb_a, kb_b, "A", cache=variant.cache)
        ranked_a = rankings(world_a, questions, Retriever)
        print(f"\n######## {name}: {variant}")
        print(f"  KB-A retrieval quality: {retrieval_quality(questions, ranked_a)}")
        report[name] = []
        for level in DRIFT_LEVELS:
            stale = stale_facts(pool, level, mode=variant.staleness)
            world_b = build_world(pool, kb_a, kb_b, "B", stale, cache=variant.cache)
            result = compare(questions, world_a, ranked_a, world_b, rankings(world_b, questions, Retriever), variant)
            result.update(drift_level=level, stale_facts=len(stale),
                          stale_pages=sorted({pool[f][0].source for f in stale}))
            report[name].append(result)
            print(f"  drift {level:.2f} ({len(stale)} stale facts): return_frozen_on_a="
                  f"{result['return_frozen_on_a']:.3f}")
            print("    " + ", ".join(f"{k}={result[k]:.3f}" if isinstance(result[k], float) else f"{k}={result[k]}"
                                     for k in SUMMARY if result[k] is not None))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, default=list), encoding="utf-8")


if __name__ == "__main__":
    main()
