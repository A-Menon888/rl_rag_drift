# Project Context: Documentation-Based RL-RAG

## Purpose

This repository is a deterministic research prototype for testing whether a reinforcement-learning retrieval policy (SEARCH_MORE / ANSWER / GIVE_UP over a frozen reader) learned on one documentation knowledge base degrades when the knowledge base drifts, and how cheaply it can be adapted.

The active experiment trains on Knowledge Base A, evaluates the frozen policy on A and B, continues training from the old weights on B, trains a separate policy from scratch on B, and compares all policies with deterministic evaluation metrics.

The corpus is local Markdown. No LLM or external API is used by the active code path.

## Repository map

```text
configs/default.yaml           Experiment configuration
data/documentation/kb_0/*.md   Generator memory snapshot (closed-book knowledge; never retrieved; generated)
data/documentation/kb_a/*.md   Original deployed documentation snapshot
data/documentation/kb_b/*.md   Drifted documentation snapshot
data/documentation/facts.yaml  Fact manifest: answer-slot regex + hand-written questions + drift events
data/documentation/memory.yaml kb_0 construction spec: seed, date, older value per KB-A fact
data/documentation/fact_trajectories.csv  Generated audit table: kb_0 -> KB-A -> KB-B per fact
experiments/run_all.py         Training, evaluation, plots, CSV, checkpoints
experiments/analyze_adaptation.py Paired seed analysis, power, bootstrap, control validation
src/data/documents.py          Markdown loader and chunker
src/data/facts.py              Fact extraction, drift typing, memory status, query builder
src/data/memory.py             kb_0 memory assignment rule and page builder (python -m src.data.memory [--check])
src/data/audit.py              Fact-trajectory validation and CSV (python -m src.data.audit)
src/data/generator.py          Shared Query record
src/retrieval/embeddings.py    Hashed-token or optional sentence-transformer embeddings
src/retrieval/retriever.py     Top-k inner-product retrieval
src/generation/mock.py         Deterministic slot reader (closed-book memory / open-book extraction)
src/environment/rl_rag_env.py  Gymnasium environment: one question per episode, paged search with a budget
src/agents/baselines.py        Answer-directly, search-N-then-answer, random policies
src/agents/rl_agent.py         MLP policy, Monte Carlo REINFORCE, checkpoint save/load
src/agents/dpo.py              Trajectory collection, preference pairs, DPO objective and training
src/evaluation/metrics.py      Evaluation and retrieval metric helpers
src/evaluation/plots.py        Matplotlib training-curve helper
tests/test_documents.py        Document and retrieval tests
tests/test_env.py              Environment, evaluation, training, config and end-to-end tests
tests/test_dpo.py              Trajectory pairs, DPO objective/training, checkpoint round trip
tests/test_dataset.py          Chunking, drift events, kb_0 memory rule, fact-trajectory invariants
```

## Where to work first

Use this path when making changes:

1. **Facts, queries and drift:** start with `data/documentation/facts.yaml` and `src/data/facts.py`. `build_fact_queries()` extracts each fact's value from KB-0/KB-A/KB-B, computes drift type and memory status, and emits one query per hand-written question. Query IDs are `<fact_id>.q<index>`; add questions at the end of a fact's list to keep existing IDs stable.
2. **Reader:** `src/generation/mock.py`. The two explicit reader assumptions (evidence overrides memory; newest page wins a conflict) live here and nowhere else.
3. **Observation and environment behavior:** `src/environment/rl_rag_env.py`. `FEATURES` lists every policy-visible feature; `_observation()` builds them; `step()` executes an action and computes reward. This is the control point for feature engineering, not `rl_agent.py`.
4. **RL logic:** `src/agents/rl_agent.py`. `run_episode()` samples one question; `update()` performs batched REINFORCE; `train_batch()` does both. Change this only when changing the learning algorithm itself.
5. **Metrics:** `src/evaluation/metrics.py`. `summarize()` computes per-episode metrics from terminal `info` records. `retrieval_diagnostics()` is separate ranking analysis.
6. **Experiment wiring and decisions:** `experiments/run_all.py`. `run_experiment()` builds the A/B environments, trains/evaluates old/adapted/retrained policies, runs baselines, and writes `summary.csv` and the KB-B sidecar. `approval_decision()` compares candidates to `search_once/kb_b`; `recovery_decision()` compares KB-B candidates to `old_policy/kb_a`.
7. **Repeated seeds:** read `experiments/run_seeds.py`. It overrides `config["seed"]` for each run, which controls RL initialization and action randomness, writes per-seed summaries, and aggregates metrics. It reuses `run_experiment()` rather than duplicating the benchmark.
8. **Experiment knobs:** `configs/default.yaml` (see Configuration).

For an RL or accuracy change, the usual reading order is:
`facts.py` -> `mock.py` -> `rl_rag_env.py` -> `rl_agent.py` -> `metrics.py` -> `run_all.py` -> the matching test file.

For a corpus/fact change, use:
`facts.yaml` -> `facts.py` -> `test_documents.py` (which enforces uniqueness, coverage and no answer leakage in questions).

The generated files under `results/` are outputs, not source of truth. Recreate them with `.venv\Scripts\python.exe experiments/run_all.py` or `.venv\Scripts\python.exe experiments/run_seeds.py --n-seeds 10`.

## Active documentation corpus

Three snapshots of the same pages form a timeline: memory (`kb_0`) -> deployed (`kb_a`) -> drifted (`kb_b`).

- `api.md`, `configuration.md`, `deployment.md`, `middleware.md`, `security.md`, `sessions.md`: synthetic references added in the 120-fact expansion, one statement per line under `##` sections.
- `authentication.md`: Django auth API reference (real text; formatting artifacts such as pilcrows and curly apostrophes were removed so only intended edits differ).
- `users.md`, `payments.md`: synthetic API reference, one topic per paragraph.
- `faq.md` (KB-A and KB-B only): restates a few facts. It is deliberately not updated in KB-B (its `updated:` date stays 2025-03-20), which creates contradictions.

Every page starts with front matter (`---` / `updated: YYYY-MM-DD` / `---`), parsed by the loader into `DocumentChunk.updated`. KB-0 pages are dated 2024-01-15, KB-A reference pages 2025-02/03, KB-B reference pages 2026-05/06.

**Chunking** (`src/data/documents.py`): blank-line paragraphs, heading lines dropped, split at 80 words. A paragraph ending in `:` is a lead-in ("Create a user with:") and is joined with the following block ("`POST /v1/users`"), never across a heading. Before this fix (2026-09-30) the value-bearing code line of `api.md` (and the Django field lists) was a separate chunk from its sentence: `api.md` gold-chunk top-3 recall was 0.67 (KB-A) / 0.63 (KB-B), now 0.83 / 0.85. Adding heading context to chunks was measured and gave no net change (overall top-3 0.91 -> 0.91 on KB-A), so headings are still dropped. Every chunk records its snapshot in `DocumentChunk.knowledge_base`; `RLRAGEnv` raises if given any `kb_0` chunk.

Chunks (KB-A / KB-B): 243 / 252; authentication.md 143 / 145 (mean 19 words), payments/users/faq 6-11 each (17-20 words), the six expanded pages 11-16 each (7-8 words). Every documented fact is stated in exactly one chunk of its owning page (tested); `authentication.md` chunks are distractors, not redundant answers. Gold-chunk top-3 recall with all-MiniLM-L6-v2 is 0.91 (KB-A) / 0.92 (KB-B), 0.82-1.00 per page; short one-sentence chunks are easier to retrieve, not harder, and top-3 recall is similar across drift types (modified 0.91, unchanged 0.92, added 0.93; contradicted 1.00 top-3 but 0.17 top-1 because the FAQ ranks first).

**kb_0 (generator memory)** is generated from the KB-A pages by `python -m src.data.memory` following `data/documentation/memory.yaml`, and checked in (a test fails if it drifts from the rule). Rule (`assign_memory`): the unit is the drift event (correlated facts share one memory status); facts not documented in KB-A (added, absent) are unknown; every other unit gets correct / stale / unknown by seeded systematic allocation, stratified by KB-B drift type (1/3 each, +-1 unit per stratum; seed 0). Correct keeps the KB-A statement, stale substitutes the hand-written older value from `memory.yaml`, unknown removes the sentence. Older values are distinct from the KB-A and KB-B values and appear in no question (checked when building), so no KB-B value can enter memory. Status is relative to KB-A. The pre-2026-09-30 kb_0 was hand-edited and covered only authentication/payments/users.

KB-B edits are exact, reviewable replacements: `/v2` endpoints, PUT -> PATCH, EUR, required idempotency keys valid for 48 hours, larger page sizes, a pending default status, reversible deletion, longer Django field limits, additions (refund window, restore window, session idle timeout, dev port) and removals (manual-review threshold, listing rate limit).

## Facts and queries

`facts.yaml` lists facts as `id`, owning `source` page, a `pattern` with exactly one capture group, and hand-written `questions`. No values are written in the manifest. `src/data/facts.py`:

- Optional `drift_event`: facts sharing one KB-A -> KB-B cause, declared with a description under top-level `drift_events` (`Fact.event` defaults to the fact id). Three events: `api_v2_migration` (10 facts on api/users/payments: every KB-B value is the KB-A value with the version path segment replaced by `/v2`; tested to be exactly that set), `user_list_max_size` (users.max_page_size and user_list_limit, one quantity on two pages), `webhook_api_added` (two endpoints in one new section). `Query.drift_event`, env terminal `info["drift_event"]`, the KB-B sidecar and `describe_queries` (facts and drift events per drift type) carry it; `summarize()` adds `drifted_event_accuracy` (each drift event weighted once) and `drifted_events`.
- `fact_value(fact, chunks)` reads the value from the owning page (at most one match, otherwise error) plus any different values stated on other pages (`conflicting`). Values are normalized (lowercase, collapsed whitespace), so formatting-only edits are not drift.
- `drift_type(A, B)`: `removed` (absent in B), `added` (absent in A), `contradicted` (another B page disagrees), `modified`, `unchanged`, or `absent` (never documented; unanswerable, not drift).
- `memory_status(memory, gold)`: `unknown` (memory has no answer), `correct`, or `stale` (a wrong value, including any value for an unanswerable question).
- `build_fact_queries(facts, kb_0, kb_a, kb_b)`: KB-A queries cover facts documented in KB-A plus `absent` facts (no leakage of KB-B additions); KB-B queries cover every fact, so removed and absent facts are asked with `gold_answer=None`. It rejects questions containing any value of their fact.

### Train/test split

`split_facts(queries_b, test_fraction, seed)` splits **facts**, not questions, so paraphrases of one fact (same answer) never straddle the split. It is stratified by KB-B drift type: every type with two or more facts contributes at least one fact to each split. `select_facts(queries, fact_ids)` filters a query list. Both splits query the same full KB (documents are never split away from retrieval); only the questions differ. **Split unit (2026-09-30):** the split assigns *split units*, not single facts: connected components of drift events and the near-duplicate `split_groups` declared in `facts.yaml` (`creation_status`, `payment_idempotency`, `rate_limits`: different facts whose questions ask for the same property). `Fact.unit` / `Query.split_unit` carry it; strata are the set of drift types in a unit. 106 units, 6 with several facts. With `test_fraction: 0.4, split_seed: 0`: train 71 facts / 62 units (KB-A 135, KB-B 151 questions), test 49 facts / 44 units (KB-A 89, KB-B 103 questions). KB-B test facts: unchanged 13, modified 23, contradicted 2, added 7, removed 3, absent 1; train: unchanged 17, modified 40, contradicted 1, added 8, removed 3, absent 2. The whole API migration (10 facts) is in train; every other multi-fact unit is in test. Only 1 contradicted fact (2 questions) is in train. The KB-A test questions are a subset of the KB-B test questions (same held-out facts; KB-B adds the held-out `added` facts). `split_seed` is fixed across training seeds, so all seeds share one test set.

`Query` fields: `query_id`, `fact_id`, `source`, `text`, `answer_pattern`, `memory_answer`, `gold_answer`, `drift_type` (`baseline` on KB-A), `memory_status`, and the derived `affected_by_drift`.

Current counts (2026-09-30): 120 facts, 254 questions. Facts by drift type (drift events in brackets): unchanged 30 (30), modified 63 (54), added 15 (14), removed 6 (6), contradicted 3 (3), absent 3 (3). Memory vs KB-A by unit: correct 31, stale 31, unknown 30; by fact: correct 31, stale 31, unknown 58 (the 10-fact API migration unit drew unknown; 18 facts are unknown because KB-A lacks them). KB-A has 224 queries, KB-B 254 (236 answerable; unchanged 64, modified 136, contradicted 6, added 30, removed 12, absent 6). `data/documentation/fact_trajectories.csv` lists every fact; `python -m src.data.audit` validates all trajectories (0 problems).

## Control flow (one episode = one question)

```text
question --> observation (query embedding + FEATURES)
                 |
        policy (masked) chooses
   SEARCH_MORE ------------------ ANSWER ------------------ GIVE_UP
   reveal next top_k unseen       reader answers from        abstain
   chunks of the question's       evidence (newest page      (+1 if the question is
   ranking; -search_cost;         wins conflicts), else      unanswerable, else
   masked after max_searches      from KB-0 memory;          give_up_reward)
        |                         +1 / -1 vs gold_answer
        +--> next observation            \_____ episode ends ______/
```

The policy only selects actions. The environment owns retrieval, the reader, correctness, reward, and audit `info`.

## Retrieval and generation

`Retriever` embeds every chunk's `.text` and returns `RetrievalResult(fact, score)` (`fact` holds a `DocumentChunk`). Search uses normalized inner products. The environment builds its retriever once and, at reset, ranks the question's top `top_k * max_searches` chunks; each SEARCH_MORE reveals the next `top_k` of that ranking (paged search), so every search adds unseen evidence.

`Embedder` supports a sentence-transformers backend (`all-MiniLM-L6-v2`, the default) and a hashed bag-of-token fallback. Vectors are cached by text. The YAML embedding settings are still not passed through `make_env()`; `Retriever` defaults are used.

`MockAnswerGenerator` is a deterministic slot reader standing in for a frozen LLM, with two explicit assumptions:

1. **Evidence overrides memory.** It extracts every statement of the question's answer slot (`answer_pattern`) from the evidence; if there is none, it answers from `memory_answer` (KB-0), or `None` ("don't know").
2. **Newer documentation supersedes older.** When evidence states different values, the value from the page with the latest `updated:` date wins; ties go to evidence order.

The reader never knows which page owns a fact. So it is wrong when only a stale page was retrieved, and it would be wrong if a stale page carried a newer date (tested). Grading still uses the owning page's value (`gold_answer`), which is independent of the reader rule.

## Environment

`RLRAGEnv(queries, chunks, *, top_k, max_searches, search_cost, correct_reward, incorrect_reward, give_up_reward)`:

- `reset(options={"query_index": i})` starts question `i` (default: the next in order). Episodes end on ANSWER or GIVE_UP.
- Actions: `0 = SEARCH_MORE`, `1 = ANSWER`, `2 = GIVE_UP`. `action_mask()` / `info["action_mask"]` disables SEARCH_MORE after `max_searches`; stepping a masked action raises.
- Rewards: SEARCH_MORE `-search_cost`. ANSWER `correct_reward` iff the answer equals gold (a non-None answer), else `incorrect_reward`, including answering an unanswerable question. GIVE_UP `correct_reward` on an unanswerable question, else `give_up_reward` (default 0, between right and wrong).
- Observation (default, `observe_query_embedding: false`): exactly `FEATURES`. With `observe_query_embedding: true` (ablation only) the 384-d query embedding is prepended. `FEATURES` are: searches used, can search, memory has answer, evidence has answer, evidence conflict, evidence agrees with memory, last search revealed a new value, best score, last search score. All are computed from the question, the evidence, and the generator's own outputs; a test checks that relabelling gold/drift/memory status leaves observations unchanged.
- Terminal `info`: query ID, final action, answer, ground truth, correct, terminal reward, episode return, searches, retrieval cost, evidence conflict, answerable, drift type, memory status, `affected_by_drift`.

The old cache, recent-reward/retrieval-rate features, database version and `drift_events` were removed.

## RL and baseline policies

`PolicyNetwork` is `Linear(obs, 64) -> Tanh -> Linear(64, 3)` (obs = 9 by default). `RLAgent.save(path)` / `load(path)` write and read the policy state dict. `RLAgent.act(observation, info, explore)` samples from the masked categorical distribution (argmax when `explore=False`). `update()` is batched Monte Carlo REINFORCE: undiscounted rewards-to-go within each episode (never across questions), normalized over the batch as the baseline, plus an entropy bonus (`entropy_coef`). `train_policy()` runs `train_epochs` passes over the question set in a seeded random order, one update per `batch_size` episodes, and returns the per-epoch mean return.

## DPO adaptation (`src/agents/dpo.py`)

DPO adapts the retrieval policy only; the reader/generator stays frozen. Preferences are generated automatically from environment returns (no human feedback; not RLHF).

1. **Reference and init:** both the frozen reference `pi_ref` and the trainable `pi_theta` are loaded from the saved old-policy checkpoint (`rl_old_policy_kb_a.pt`).
2. **Trajectories:** for every KB-B **train** question, `collect_trajectories` rolls out all `2 * (max_searches + 1)` = 8 stopping strategies ("SEARCH_MORE s times, then ANSWER or GIVE_UP"), recording each step's policy observation, action mask and action, plus the episode return. These rollouts do not depend on `pi_ref`.
3. **Pairs (one per question at most):** `rejected` is the candidate the frozen policy actually selects (`frozen_policy_actions`: its greedy action sequence, always one of the 8 candidates); `chosen` is the highest-return candidate (first in enumeration order on ties). A pair exists only when chosen return > frozen return **and** the two trajectories end in different answer outcomes (`Trajectory.outcome` = (final action, correct)): pairs that differ only in search count/retrieval cost are discarded; wrong->right, wrong->GIVE_UP, right->wrong and other outcome changes are kept. `dpo_pairs.csv` records each side's outcome. Returns use gold only as the RL reward does; they never enter observations.
4. **Objective:** `log pi(tau) = sum_t log pi(a_t | s_t)` under the action mask (transition terms cancel), and
   `L = -mean log sigmoid(beta * [(log pi_theta(tau_w) - log pi_ref(tau_w)) - (log pi_theta(tau_l) - log pi_ref(tau_l))])`,
   with Adam, `dpo_epochs` passes over shuffled pairs in minibatches of `dpo_batch_size`.
5. Outputs: `checkpoints/dpo_policy_kb_b.pt`, `metrics/dpo_pairs.csv` (every pair's action sequences and returns), a DPO loss curve, and budgets in `metadata.json` and summary rows.

**Equal-data control (`rl_finetune_policy`):** REINFORCE fine-tuning from the same old-policy checkpoint on KB-B train with exactly DPO's environment episodes (352 = 8 epochs x 44 questions; asserted), same `batch_size`, `learning_rate`, `entropy_coef` as the other RL arms.

Budget per run (default config): DPO-352 352 episodes, 20 gradient updates, 960 policy step evaluations (11 pairs from 44 questions). RL-352 352 episodes, 24 updates, about 680 step evaluations. Full RL retraining (unchanged) 13,200 episodes, 900 updates, about 34,000-40,000 step evaluations.

Baselines: `answer_directly` (closed-book), `search_once` (1 search then answer), `search_all` (use the whole budget then answer), and seeded `random` over available actions.

## Current experiment runner

`experiments/run_all.py`:

1. Loads KB-0/KB-A/KB-B, builds fact queries, splits facts into train/test, and builds four environments: (kb_a, train), (kb_a, test), (kb_b, train), (kb_b, test). It asserts that no test fact is in a training environment and writes `split.json`.
2. Trains `old_policy` on **KB-A train** questions, then evaluates it frozen (greedy) on KB-A train (fit), **KB-A test** (pre-drift baseline) and **KB-B test** (degradation on the same held-out facts).
3. Builds `dpo_policy` by DPO and `rl_finetune_policy` (equal-data REINFORCE control) from the saved old-policy checkpoint using KB-B **train** questions (see DPO adaptation); trains `full_retrain_policy` with REINFORCE from a fresh initialization on KB-B train (unchanged); evaluates both on KB-B train and **KB-B test**.
4. Evaluates the four baselines on KB-A test and KB-B test.
5. Computes baseline-relative decisions on held-out rows:

```text
approval: candidate kb_b/test accuracy > search_once/kb_b/test accuracy - approval_margin
          and candidate retrieval cost <= search_once/kb_b/test cost + approval_margin
recovery: candidate kb_b/test accuracy >= old_policy/kb_a/test accuracy
```

No training ever happens on test questions of either KB.

The runner validates the config (`validate_config`: required keys and types), then writes 17 CSV rows keyed by `policy`, `knowledge_base` and `split` (old policy on kb_a train/test and kb_b test; `dpo_policy`, `rl_finetune_policy` and full retrain on kb_b train/test; four baselines on kb_a/kb_b test) with `accuracy`, `average_reward` (mean episode return), `average_searches`, `retrieval_rate` (episodes with at least one search), `retrieval_cost`, `unnecessary_retrieval_rate` (searched although closed-book memory was correct), `give_up_rate`, `wrong_answer_rate`, `unanswerable_accuracy`, `drifted_accuracy`, `stable_accuracy`, `training_final_return`, the budget fields `adaptation_episodes`, `gradient_updates`, `policy_step_evaluations` (for rows of the policy's training KB), and the approval/recovery fields. `metadata.json` records the observation definition (flag, feature names, dimension) and the DPO/full-retrain/old-policy budgets. The sidecar `kb_b_query_results.csv` covers KB-B **test** questions only, with per-question `fact_id`, `drift_event`, `drift_type`, `memory_status`, and each trained policy's `_correct`, `_action` (final action) and `_searches`.

**Smoke run on the reviewed 120-fact dataset** (3 seeds, default config, split units, 2026-09-30; outputs in a scratch directory, not `results/`). KB-A test 89 questions, KB-B test 103:

| row | acc | return | searches | give-up |
|---|---|---|---|---|
| old_policy kb_a test | 0.955 | 0.804 | 1.28 | 0.045 |
| old_policy kb_b test (frozen) | 0.893 | 0.667 | 1.39 | 0.058 |
| dpo_policy (DPO-1208) | 0.893 | 0.667 | 1.39 | 0.058 |
| rl_finetune_policy (RL-1208) | 0.906 (0.89-0.93) | 0.699 (0.67-0.76) | 1.39 | 0.078 |
| full_retrain_policy | 0.932 | 0.764 | 1.39 | 0.117 |
| search_once / search_all kb_b | 0.757 / 0.883 | 0.415 / 0.467 | 1 / 3 | 0 |

- Adaptation budget is now 1208 episodes (8 epochs x 151 KB-B train questions). DPO: 7 pairs per seed (identical across seeds), 20 updates, loss 0.693 -> ~0.59. RL-1208: 80 updates. Full retrain: 45,300 episodes, 3000 updates.
- DPO leaves every greedy action unchanged on every KB-B test question in all 3 seeds (identical to frozen).
- All four trained policies use the same search depth on every test question (search until evidence states the slot, up to 3). They differ only in the final action after 3 fruitless searches: the frozen policy answers from memory, full retrain gives up. Full retrain's whole gain is the 2 removed facts with stale memory (4 questions), plus two answerable stale-memory questions where it gives up instead of answering wrongly (return, not accuracy). RL-1208 learns this give-up in seed 1 only.
- No policy resolves contradictions: every policy answers after 1 search on all 4 contradicted test questions (accuracy 0.25; `users.max_page_size.q0` is right only because users.md ranks above the FAQ). Train holds 1 contradicted fact (`default_currency`, memory = the stale FAQ value); its 2 DPO pairs (search 1 -> search 2+) do not change behavior, and full retrain does not learn to search again either.
- Approval (cost <= search_once cost) is DECLINED for every arm because all search ~1.39 times; recovery (>= old_policy kb_a 0.955) is not reached by any arm. Both are heuristics.

Held-out results, 3 seeds (0-2), default config with outcome-changing pairs only, 2026-09-30, KB-B test facts. **36-fact corpus and old kb_0; superseded dataset, not rerun on the 120-fact dataset:**

| row | acc | return | searches |
|---|---|---|---|
| old_policy kb_a test | 1.000 | 0.900 | 1.00 |
| old_policy kb_b test (frozen) | 0.821 | 0.543 | 1.00 |
| dpo_policy (DPO-352) | 0.821 | 0.542 (0.539, 0.543, 0.543) | 1.01 |
| rl_finetune_policy (RL-352) | 0.821 | 0.543 | 1.00 |
| full_retrain_policy | 0.929 | 0.682 | 1.75 |
| search_once / search_all | 0.750 / 0.857 | 0.400 / 0.414 | 1 / 3 |

- Pairs: 11 of 44 train questions, identical across seeds: `search+answer` (wrong) -> `search+search+answer` (right) x6 (contradictions and one modified), `search+answer` (wrong) -> `give_up` (answerable, 0 reward) x4, `search+answer` (wrong) -> 3 searches + answer (right) x1. DPO loss 0.693 -> about 0.61.
- DPO-352 and RL-352 both leave behavior essentially at the frozen policy (DPO adds one extra search on one question in seed 0). Neither recovers contradictions (0/6); full RL recovers 6/6 but searches 2.5 times on unchanged facts (vs 1.0) to do so.
- Observability: after one search, 3 of 4 contradicted train questions have exactly the same non-score features as 5 unchanged/memory-correct questions (the stale FAQ agrees with stale memory, so evidence_conflict = 0 and evidence_agrees_with_memory = 1); only retrieval scores differ. Learning "search again" there necessarily also costs on those questions.

Earlier variants (same day, not current): "best vs every worse" pairs (308 pairs) gave DPO return -0.036; best-vs-frozen including cost-only pairs (29 pairs, 18 cost-only) gave 0.460 (0.543, 0.543, 0.293).

Old policy, RL-352 and full retrain are deterministic enough that seed variation is near zero; varying `split_seed` would be needed to measure variance. `train_dpo` raises if a run yields zero pairs.

With `observe_query_embedding: true` (ablation) the policy memorizes training questions and underperforms search-once on held-out facts (3 seeds, previous REINFORCE-adaptation pipeline: old policy kb_a train/test accuracy 0.975/0.653, return 0.923/0.449). Results under `results/` predate this pipeline and are stale.

`experiments/run_seeds.py` reuses `run_experiment()` for independent deterministic seeds. `--config` selects the YAML configuration, `--n-seeds` defaults to 10, and `--base-seed` defaults to 0; `--jobs` optionally runs independent seeds in separate processes without changing their seed configuration, and `--resume` reuses a seed only when both its summary and paired query sidecar exist. Each seed writes `results/metrics/seed_<seed>/summary.csv`, checkpoints, figures, and config. It also writes `results/metrics/aggregate_summary.csv`, containing mean, sample standard deviation, minimum, maximum, and JSON raw per-seed values for accuracy, average reward, average searches, retrieval rate, retrieval cost, unnecessary retrieval rate, give-up rate, drifted accuracy, and stable accuracy. It prints mean +/- standard deviation for accuracy and retrieval rate. Queries no longer depend on the seed; `alignment_issues()` checks once that every KB-A query exists in KB-B with the same memory answer.

`configs/control_high_cost.yaml` is an explicit reduced validation control with 60 training epochs and search cost 0.50; keep its outputs isolated from default metrics.

Adapted and full-retrain evaluation within a seed use the same `queries_b` object and deterministic ordering. Each seed also writes the `kb_b_query_results.csv` sidecar described above. `run_seeds.py` accepts `--output-root` for isolated runs and `--retrieval-cost` for explicit control experiments.

`experiments/analyze_adaptation.py` compares `dpo_policy` (constant `ADAPTED_POLICY`) with `full_retrain_policy`; it reads `aggregate_summary.csv` plus the authoritative per-seed summaries and query-result sidecars. It reports the recovery ratio, full-retrain-minus-adapted paired accuracy differences, the configurable `z_value * SE` decision threshold, paired Cohen's d and magnitude bin, an exact two-sided paired-t minimum detectable effect at configurable alpha/power, and diagnostic within-seed query-bootstrap interval widths. It can compare a prefix such as 10 seeds with a 30-seed analysis and validate a separate adversarial control directory. The seed is the inferential unit; query bootstrap output does not affect the decision.

Outputs:

- `results/metrics/summary.csv`
- `results/metrics/config.json`
- `results/metrics/adaptation_analysis.json`
- `results/metrics/seed_<seed>/kb_b_query_results.csv`
- training curves under `results/figures/`
- PyTorch checkpoints under `results/checkpoints/`

## Configuration

`configs/default.yaml` currently defines:

```yaml
seed: 7
corpus_dir: data/documentation
facts_path: data/documentation/facts.yaml
test_fraction: 0.4   # share of facts held out per drift type (fact-level split)
split_seed: 0        # fixed across training seeds, so all seeds share one test set
knowledge_bases: [kb_a, kb_b]
top_k: 1            # chunks revealed per SEARCH_MORE (paged search)
max_searches: 3     # retrieval budget per question
observe_query_embedding: false  # true = ablation: prepend the query embedding to FEATURES
use_semantic_embeddings: true
embedding_model: all-MiniLM-L6-v2
# Set this to true only when an explicit deterministic offline fallback is desired.
allow_embedding_fallback: false
train_epochs: 300   # passes over the question set
batch_size: 16      # episodes per REINFORCE update
learning_rate: 0.001
entropy_coef: 0.01
retrieval_cost: 0.10   # per SEARCH_MORE
correct_reward: 1.0
incorrect_reward: -1.0
give_up_reward: 0.0  # GIVE_UP on an answerable question (correct_reward if unanswerable)
# DPO adaptation of the frozen KB-A policy from KB-B train trajectory pairs
dpo_epochs: 20
dpo_batch_size: 32
dpo_beta: 0.1
dpo_learning_rate: 0.001
approval_margin: 0.0
rolling_window: 30
```

`run_all.py` consumes (and `validate_config` requires) `seed`, `corpus_dir`, `facts_path`, `test_fraction`, `split_seed`, `top_k`, `max_searches`, `observe_query_embedding`, `dpo_epochs`, `dpo_batch_size`, `dpo_beta`, `dpo_learning_rate`, `train_epochs`, `batch_size`, `learning_rate`, `entropy_coef`, `retrieval_cost`, `correct_reward`, `incorrect_reward`, `give_up_reward`, and `approval_margin`. `knowledge_bases`, embedding settings and `rolling_window` are present but not wired in. `run_seeds.py` overrides `seed` per run; `split_seed` stays fixed. Aggregates are grouped by (policy, knowledge base, split), and `analyze_adaptation.py` reads the `test` rows.

`top_k: 1` is deliberate: with 3 chunks per search, the first search already surfaced both the stale FAQ and the newer reference page for every contradiction, so SEARCH_MORE was never needed. With 1 chunk per search, 5 of 6 contradictions need a second search, and the best fixed search depth differs between KB-A (1 search) and KB-B (2 searches).

## Tests and commands

From the repository root:

```powershell
$env:PYTHONPATH='.'
python -m pytest -q tests
python experiments/run_all.py
python experiments/run_seeds.py --n-seeds 30 --jobs 4 --resume
python experiments/analyze_adaptation.py --n-seeds 30 --compare-n-seeds 10
```

The current suite contains 69 tests covering: split units never straddling train/test; api.md agreeing with payments.md on creation status and idempotency; lead-in chunking and one answer chunk per fact; drift-event declarations (the API migration is exactly the version-path substitutions); the kb_0 memory rule (explicit auditable status for every fact, reproducible and matching the checked-in kb_0, balanced per drift type, shared within drift events, no KB-B-only value in memory, every older value valid, kb_0 rejected as a retrieval corpus, generator uses memory only closed-book); fact-trajectory validation and the checked-in trajectory CSV; DPO stopping strategies, trajectory recording and replay, frozen-policy greedy actions, best-vs-frozen pairs (one per question, only when strictly better and the answer outcome differs, train questions only), cost-only pair removal, tie handling, the RL-352 equal-episode budget, the DPO loss, trajectory log-probabilities, DPO training (moves toward chosen, reference frozen), checkpoint round trip; the evidence-only default observation and the embedding ablation; config validation; an end-to-end old-policy -> checkpoint -> DPO -> saved-policy run using only train facts; the fact-level split (disjoint, stratified, paraphrases together, seeded, KB-A test within KB-B test); an end-to-end run proving no test fact is trained on; corpus formatting and front matter; value-based drift typing and memory status; drift-type coverage; no KB-B leakage into KB-A; non-trivial KB-A memory; no answer values in questions; one question per episode; SEARCH_MORE revealing unseen chunks; budget masking; closed-book, GIVE_UP and unanswerable rewards; newest-source reader resolution (including that it can be wrong); contradictions resolvable by searching more; observations independent of gold/drift labels; retrieval that can fail; baseline behavior; the agent never taking a masked action; frozen evaluation; training and adaptation weight updates; approval/recovery decisions; and the adaptation analysis.

## Current limitations and next work

- The corpus (120 facts) is partly synthetic; the slot reader measures retrieval/policy behavior, not language quality. The results above are from the 36-fact corpus and the previous kb_0; nothing has been rerun on the 120-fact dataset.
- Drift cases that require a different retrieval policy after drift are scarce: 3 contradicted facts (1 in test) and 6 removed (2 in test). Modified facts are mostly handled by the same evidence-first behavior. Contradiction/removal results will be anecdotal; drift types were left unchanged rather than edited toward a target distribution.
- Fixed 2026-09-30: KB-A api.md now says the payment endpoint "supports" (was "requires") an `Idempotency-Key` header, matching payments.md; KB-B api.md resource creation returns 201 (was 202), matching payments.md/users.md (`create_status` is now unchanged). Remaining known cross-page inconsistency, inert for the slot reader: KB-A payments.md/users.md unversioned paths vs api.md `/v1`. The `payment_idempotency` questions still say "required", which is only true in KB-B.
- DPO-352 (outcome-changing best-vs-frozen pairs) does not move the policy away from the frozen behavior in the 3-seed smoke run: 11 pairs and 20 updates; contradictions are partly unobservable after one search (see results). No tuning has been done.
- The default observation is evidence-only; the policy cannot tell apart questions with identical evidence state, which is intended (it must generalize) but makes per-state preference balance matter for DPO.
- The active runner relies on `Retriever` defaults because the YAML embedding settings are not passed through `make_env()`; explicit embedding-model/fallback configuration remains future wiring work.
- Run the benchmark with `.venv\Scripts\python.exe` so it does not depend on an unrelated system Python installation.
- Retrieval diagnostics exist in `src/evaluation/metrics.py`, but the runner does not currently write per-query retrieval diagnostics.
- The environment is stationary within each experiment arm; it does not apply scheduled drift during an episode.
- Approval is a baseline-relative heuristic, not a statistical significance test; its reference row and margin are recorded in the CSV for auditability.
- Per-step logs, recovery-time analysis, and richer plots remain future work.
- Multi-seed aggregate infrastructure is implemented; the current 10-seed run is measurement infrastructure, not a scientific conclusion about margin selection.

Do not claim that adaptation improves KB-B performance until the generated evaluation CSV and deterministic comparison support that conclusion. Preserve the separation between policy, environment, retrieval, and reader.

The important files are now clearly mapped:

Facts, drift types, memory status: facts.yaml and facts.py
Reader assumptions: mock.py
Observation, actions, budget, reward flow: rl_rag_env.py
Policy network and REINFORCE training: rl_agent.py
Accuracy and metric calculations: metrics.py
DPO adaptation and pair construction: dpo.py
Main experiment wiring and approval/recovery decisions: run_all.py
Multi-seed execution and aggregation: run_seeds.py
Active experiment settings: default.yaml
Relevant tests: test_documents.py and test_env.py
