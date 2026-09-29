# Project Context: Documentation-Based RL-RAG

## Purpose

This repository is a deterministic research prototype for testing whether a reinforcement-learning retrieval policy (SEARCH_MORE / ANSWER / GIVE_UP over a frozen reader) learned on one documentation knowledge base degrades when the knowledge base drifts, and how cheaply it can be adapted.

The active experiment trains on Knowledge Base A, evaluates the frozen policy on A and B, continues training from the old weights on B, trains a separate policy from scratch on B, and compares all policies with deterministic evaluation metrics.

The corpus is local Markdown. No LLM or external API is used by the active code path.

## Repository map

```text
configs/default.yaml           Experiment configuration
data/documentation/kb_0/*.md   Generator memory snapshot (closed-book knowledge; never retrieved)
data/documentation/kb_a/*.md   Original deployed documentation snapshot
data/documentation/kb_b/*.md   Drifted documentation snapshot
data/documentation/facts.yaml  Fact manifest: answer-slot regex + hand-written questions
experiments/run_all.py         Training, evaluation, plots, CSV, checkpoints
experiments/analyze_adaptation.py Paired seed analysis, power, bootstrap, control validation
src/data/documents.py          Markdown loader and chunker
src/data/facts.py              Fact extraction, drift typing, memory status, query builder
src/data/generator.py          Shared Query record
src/retrieval/embeddings.py    Hashed-token or optional sentence-transformer embeddings
src/retrieval/retriever.py     Top-k inner-product retrieval
src/generation/mock.py         Deterministic slot reader (closed-book memory / open-book extraction)
src/environment/rl_rag_env.py  Gymnasium environment: one question per episode, paged search with a budget
src/agents/baselines.py        Answer-directly, search-N-then-answer, random policies
src/agents/rl_agent.py         MLP policy and Monte Carlo REINFORCE
src/evaluation/metrics.py      Evaluation and retrieval metric helpers
src/evaluation/plots.py        Matplotlib training-curve helper
tests/test_documents.py        Document and retrieval tests
tests/test_env.py              Environment, evaluation, and training tests
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

- `authentication.md`: Django auth API reference (real text; formatting artifacts such as pilcrows and curly apostrophes were removed so only intended edits differ).
- `users.md`, `payments.md`: synthetic API reference, one topic per paragraph.
- `faq.md` (KB-A and KB-B only): restates a few facts. It is deliberately not updated in KB-B (its `updated:` date stays 2025-03-20), which creates contradictions.

Every page starts with front matter (`---` / `updated: YYYY-MM-DD` / `---`), parsed by the loader into `DocumentChunk.updated`. KB-0 pages are dated 2024-01-15, KB-A reference pages 2025-02/03, KB-B reference pages 2026-05/06.
- `kb_0` contains only the reference pages, with some facts absent (unknown to memory) or at older values (stale memory).

KB-B edits are exact, reviewable replacements: `/v2` endpoints, PUT -> PATCH, EUR, required idempotency keys valid for 48 hours, larger page sizes, a pending default status, reversible deletion, longer Django field limits, additions (refund window, restore window, session idle timeout, dev port) and removals (manual-review threshold, listing rate limit).

## Facts and queries

`facts.yaml` lists facts as `id`, owning `source` page, a `pattern` with exactly one capture group, and hand-written `questions`. No values are written in the manifest. `src/data/facts.py`:

- `fact_value(fact, chunks)` reads the value from the owning page (at most one match, otherwise error) plus any different values stated on other pages (`conflicting`). Values are normalized (lowercase, collapsed whitespace), so formatting-only edits are not drift.
- `drift_type(A, B)`: `removed` (absent in B), `added` (absent in A), `contradicted` (another B page disagrees), `modified`, `unchanged`, or `absent` (never documented; unanswerable, not drift).
- `memory_status(memory, gold)`: `unknown` (memory has no answer), `correct`, or `stale` (a wrong value, including any value for an unanswerable question).
- `build_fact_queries(facts, kb_0, kb_a, kb_b)`: KB-A queries cover facts documented in KB-A plus `absent` facts (no leakage of KB-B additions); KB-B queries cover every fact, so removed and absent facts are asked with `gold_answer=None`. It rejects questions containing any value of their fact.

`Query` fields: `query_id`, `fact_id`, `source`, `text`, `answer_pattern`, `memory_answer`, `gold_answer`, `drift_type` (`baseline` on KB-A), `memory_status`, and the derived `affected_by_drift`.

Current counts: 36 facts (3 never documented). KB-A has 64 queries (58 answerable; memory correct 40 / stale 12 / unknown 12). KB-B has 72 queries (62 answerable; unchanged 22, modified 26, contradicted 6, added 8, removed 4, absent 6; memory correct 18 / stale 34 / unknown 20).

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
- Observation: query embedding (384) + `FEATURES`: searches used, can search, memory has answer, evidence has answer, evidence conflict, evidence agrees with memory, last search revealed a new value, best score, last search score. All are computed from the question, the evidence, and the generator's own outputs; a test checks that relabelling gold/drift/memory status leaves observations unchanged.
- Terminal `info`: query ID, final action, answer, ground truth, correct, terminal reward, episode return, searches, retrieval cost, evidence conflict, answerable, drift type, memory status, `affected_by_drift`.

The old cache, recent-reward/retrieval-rate features, database version and `drift_events` were removed.

## RL and baseline policies

`PolicyNetwork` is `Linear(obs, 64) -> Tanh -> Linear(64, 3)`. `RLAgent.act(observation, info, explore)` samples from the masked categorical distribution (argmax when `explore=False`). `update()` is batched Monte Carlo REINFORCE: undiscounted rewards-to-go within each episode (never across questions), normalized over the batch as the baseline, plus an entropy bonus (`entropy_coef`). `train_policy()` runs `train_epochs` passes over the question set in a seeded random order, one update per `batch_size` episodes, and returns the per-epoch mean return.

Baselines: `answer_directly` (closed-book), `search_once` (1 search then answer), `search_all` (use the whole budget then answer), and seeded `random` over available actions.

## Current experiment runner

`experiments/run_all.py`:

1. Loads KB-0/KB-A/KB-B and builds fact queries; prints counts per KB.
2. Trains `old_policy` on KB-A, saves `rl_old_policy_kb_a.pt` and a training curve, and evaluates it (frozen, greedy) on KB-A and KB-B.
3. Copies old weights into `adapted_policy` and trains it on KB-B (same epochs as a full retrain); trains `full_retrain_policy` from a fresh initialization on KB-B; evaluates both on KB-B.
4. Evaluates the four baselines on both KBs.
5. Computes baseline-relative decisions:

```text
approval: candidate accuracy > search_once/kb_b accuracy - approval_margin
          and candidate retrieval cost <= search_once/kb_b cost + approval_margin
recovery: candidate KB-B accuracy >= old_policy/kb_a accuracy
```

The runner writes 12 CSV rows (old policy on A/B, adapted and full retrain on B, four baselines on A/B) with `accuracy`, `average_reward` (mean episode return), `average_searches`, `retrieval_rate` (episodes with at least one search), `retrieval_cost`, `unnecessary_retrieval_rate` (searched although closed-book memory was correct), `give_up_rate`, `wrong_answer_rate`, `unanswerable_accuracy`, `drifted_accuracy`, `stable_accuracy`, `training_final_return`, and the approval/recovery fields. The sidecar `kb_b_query_results.csv` has per-question `drift_type`, `memory_status`, and each trained policy's `_correct`, `_action` (final action) and `_searches`.

Fixed-strategy reference (deterministic, top_k=1, max_searches=3, search cost 0.10), accuracy/return: KB-A answer-directly 0.625/0.250, search-once 0.875/0.650, search-all 0.891/0.481, per-question oracle 0.984/0.956. KB-B answer-directly 0.250/-0.500, search-once 0.667/0.233, search-twice 0.778/0.356, search-all 0.806/0.311, oracle 0.944/0.872.

The 2026-09-29 single-seed default run (seed 7, 300 epochs, batch 16, about 5 minutes), accuracy/return: old policy on KB-A 0.984/0.950 (0.34 searches per question, gives up on all unanswerable questions). Frozen on KB-B 0.597/0.183 with a 0.375 wrong-answer rate: it keeps trusting stale memory (modified 0.42, contradicted 0.17, added 0.25, removed 0.25; unchanged and absent 1.00). Continued REINFORCE (adapted) 0.847/0.685; full retrain 0.944/0.839. Full retrain resolves all contradictions by searching (1.8 searches); adapted stays at 0.17 on contradictions. The retrain policy gives up on removed facts without searching, which it can only know by memorizing the question: evidence of per-question memorization while training and evaluation use the same questions. Single seed; not a generalization claim. Results under results/ predate this environment and are stale.

`experiments/run_seeds.py` reuses `run_experiment()` for independent deterministic seeds. `--config` selects the YAML configuration, `--n-seeds` defaults to 10, and `--base-seed` defaults to 0; `--jobs` optionally runs independent seeds in separate processes without changing their seed configuration, and `--resume` reuses a seed only when both its summary and paired query sidecar exist. Each seed writes `results/metrics/seed_<seed>/summary.csv`, checkpoints, figures, and config. It also writes `results/metrics/aggregate_summary.csv`, containing mean, sample standard deviation, minimum, maximum, and JSON raw per-seed values for accuracy, average reward, average searches, retrieval rate, retrieval cost, unnecessary retrieval rate, give-up rate, drifted accuracy, and stable accuracy. It prints mean +/- standard deviation for accuracy and retrieval rate. Queries no longer depend on the seed; `alignment_issues()` checks once that every KB-A query exists in KB-B with the same memory answer.

`configs/control_high_cost.yaml` is an explicit reduced validation control with 60 training epochs and search cost 0.50; keep its outputs isolated from default metrics.

Adapted and full-retrain evaluation within a seed use the same `queries_b` object and deterministic ordering. Each seed also writes the `kb_b_query_results.csv` sidecar described above. `run_seeds.py` accepts `--output-root` for isolated runs and `--retrieval-cost` for explicit control experiments.

`experiments/analyze_adaptation.py` reads `aggregate_summary.csv` plus the authoritative per-seed summaries and query-result sidecars. It reports the recovery ratio, full-retrain-minus-adapted paired accuracy differences, the configurable `z_value * SE` decision threshold, paired Cohen's d and magnitude bin, an exact two-sided paired-t minimum detectable effect at configurable alpha/power, and diagnostic within-seed query-bootstrap interval widths. It can compare a prefix such as 10 seeds with a 30-seed analysis and validate a separate adversarial control directory. The seed is the inferential unit; query bootstrap output does not affect the decision.

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
knowledge_bases: [kb_a, kb_b]
top_k: 1            # chunks revealed per SEARCH_MORE (paged search)
max_searches: 3     # retrieval budget per question
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
approval_margin: 0.0
rolling_window: 30
```

`run_all.py` consumes `seed`, `corpus_dir`, `facts_path`, `top_k`, `max_searches`, `train_epochs`, `batch_size`, `learning_rate`, `entropy_coef`, `retrieval_cost`, `correct_reward`, `incorrect_reward`, `give_up_reward`, and `approval_margin`. `knowledge_bases`, embedding settings and `rolling_window` are present but not wired in. `run_seeds.py` overrides `seed` per run.

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

The current suite contains 34 tests covering: corpus formatting and front matter; value-based drift typing and memory status; drift-type coverage; no KB-B leakage into KB-A; non-trivial KB-A memory; no answer values in questions; one question per episode; SEARCH_MORE revealing unseen chunks; budget masking; closed-book, GIVE_UP and unanswerable rewards; newest-source reader resolution (including that it can be wrong); contradictions resolvable by searching more; observations independent of gold/drift labels; retrieval that can fail; baseline behavior; the agent never taking a masked action; frozen evaluation; training and adaptation weight updates; approval/recovery decisions; and the adaptation analysis.

## Current limitations and next work

- The corpus is small (36 facts) and partly synthetic; the slot reader measures retrieval/policy behavior, not language quality. The policy is trained and evaluated on the same questions (no held-out split yet).
- `adapted_policy` is continued REINFORCE with the same budget as full retraining, not yet a small preference-based adaptation; its cost advantage cannot show up yet.
- The observation includes the query embedding, so the policy can still memorize per question.
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
Main experiment wiring and approval/recovery decisions: run_all.py
Multi-seed execution and aggregation: run_seeds.py
Active experiment settings: default.yaml
Relevant tests: test_documents.py and test_env.py
