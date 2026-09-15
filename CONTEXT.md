# Project Context: Documentation-Based RL-RAG

## Purpose

This repository is a deterministic research prototype for testing whether a reinforcement-learning policy learns when retrieval is worth its cost after a documentation knowledge base changes.

The active experiment trains on Knowledge Base A, evaluates the frozen policy on A and B, continues training from the old weights on B, trains a separate policy from scratch on B, and compares all policies with deterministic evaluation metrics.

The corpus is local Markdown. No LLM or external API is used by the active code path.

## Repository map

```text
configs/default.yaml           Experiment configuration
data/documentation/kb_a/*.md   Original documentation snapshot
data/documentation/kb_b/*.md   Changed documentation snapshot
experiments/run_all.py         Training, evaluation, plots, CSV, checkpoints
src/data/documents.py          Markdown loader, chunker, KB pairing, query builder
src/data/generator.py          Shared Query record and compatibility answer method
src/retrieval/embeddings.py    Hashed-token or optional sentence-transformer embeddings
src/retrieval/retriever.py     Top-k inner-product retrieval
src/generation/mock.py         Deterministic answer generator
src/environment/rl_rag_env.py  Gymnasium environment
src/agents/baselines.py        Always-direct, always-retrieve, random policies
src/agents/rl_agent.py         MLP policy and Monte Carlo REINFORCE
src/evaluation/metrics.py      Evaluation and retrieval metric helpers
src/evaluation/plots.py        Matplotlib reward-series helper
tests/test_documents.py        Document and retrieval tests
tests/test_env.py              Environment, evaluation, and training tests
```

## Active documentation corpus

Both snapshots currently contain exactly three files:

- `authentication.md`: token endpoints, token expiry, refresh tokens, and authentication behavior.
- `users.md`: user endpoints, account status, and pagination.
- `payments.md`: payment endpoints, currency/status values, and idempotency behavior.

KB-B changes content while preserving the chunk count for every file. Examples include `/v2/...` paths, authentication expiry changing from 3600 to 7200 seconds, user status changing from `active` to `pending`, pagination changing from 20/100 to 50/200, payment currency changing from USD to EUR, and payment status/idempotency changes.

## Documents and query pairing

`DocumentChunk` is a frozen dataclass with:

```python
text, document_id, source, title, chunk_id
```

`load_documents()` finds sorted Markdown files, extracts the first `#` heading as the title, removes heading lines from text, splits paragraphs, and further splits paragraphs over `max_words` (default 80). `value` is a compatibility alias for `text`.

`load_knowledge_base(root, "kb_a" | "kb_b")` selects one explicit snapshot and rejects other names.

`generate_document_queries(kb_a_chunks, kb_b_chunks)` aligns chunks by source and chunk order. It raises `ValueError` when any source has a different chunk count between snapshots. It returns paired query lists with stable IDs:

- KB-A queries have `memorized_answer == current_answer` and `affected_by_drift == False`.
- KB-B queries retain KB-A text as `memorized_answer`, use KB-B text as `current_answer`, and set `affected_by_drift` when the text differs.
- Query text is generated from the first eight words of the KB-A chunk.

`Query.answer()` is a compatibility method that returns `current_answer`; it is not used to provide ground truth in the observation.

## Control flow

```text
documentation query
        |
        v
11-value observation
        |
        v
policy chooses DIRECT (0) or RETRIEVE (1)
        |                         |
memorized KB-A answer       search current snapshot
or cached context           and use top retrieved chunks
        \_________________________/
                    |
             deterministic mock answer
                    |
       compare with query.current_answer
                    |
                 reward/info
                    |
              optional policy update
```

The policy selects an action. The environment owns retrieval, answer generation, caching, correctness, reward, and audit information.

## Retrieval and generation

`Retriever` embeds every chunk's `.text` and returns `RetrievalResult(fact, score)`. The field is still named `fact` for compatibility, but contains a `DocumentChunk`. Search uses normalized vector inner products; FAISS is optional and disabled by the active runner.

`Embedder` supports a deterministic hashed bag-of-token fallback and a `sentence-transformers` backend. `Retriever` now defaults to requesting semantic embeddings. If model loading fails while fallback is allowed, `Embedder` records the exception and emits a warning before using hashed embeddings. The active environment still does not pass the YAML embedding configuration through, so it uses `Retriever` defaults; the configured `all-MiniLM-L6-v2` model now loads successfully in the active environment after installing `tf-keras`, and reports backend `sentence-transformers`. The config also does not currently control `correct_reward`, `incorrect_reward`, or `allow_embedding_fallback` in the environment.

Embedding instances are shared by configuration and cache vectors by input text. Corpus chunks are encoded once per configured embedder, even when the environment rebuilds a retriever on every reset. Query text is also encoded once and reused by both observation construction and retrieval. The project `.venv` contains CPU Torch, sentence-transformers, and the pinned `tf-keras` compatibility dependency required by Transformers/Keras 3.

`MockAnswerGenerator` returns the first retrieved context item's `.value`, or `query.memorized_answer` when no context is supplied. Consequently, uncached DIRECT is correct for KB-A and for unchanged KB-B queries, but wrong for drifted KB-B queries. A successful retrieval caches its retrieved chunks by query ID; cached DIRECT then has no retrieval cost.

## Environment

`RLRAGEnv(queries, snapshots, drift_events=None, ...)` has:

- Actions: `0 = DIRECT`, `1 = RETRIEVE`.
- Observation shape `(11,)`: the first seven dimensions of the current query embedding, latest reward, mean retrieval frequency over the last 20 actions, database version, and cache availability.
- Reward defaults: correct `+1.0`, incorrect `-1.0`, and retrieval cost `0.10`, so correct retrieval is `+0.9` and incorrect retrieval is `-1.1`.

`reset()` clears time, recent history, database version, cache, and recreates the retriever from `snapshots[0]`. `step()` uses the query at `t % len(queries)`, retrieves only for action 1, generates an answer, compares it with `current_answer`, caches only correct retrievals, records reward/cost/history, and returns `(observation, reward, False, False, info)`. Episodes are ended by the caller after `len(env.queries)` steps; the environment itself never returns terminated or truncated as true.

The `drift_events` argument is accepted for compatibility but is not used. The active experiment supplies one stationary snapshot `{0: chunks}` per environment, so `database_version` remains 0 and `drift_event` is always `None`. `info` includes query ID, action, answer, ground truth, correctness, reward, database version, drift fields, retrieval cost, cache hit, cache size, and `affected_by_drift`.

## RL and baseline policies

`PolicyNetwork` is `Linear(11, 32) -> Tanh -> Linear(32, 2)`. `RLAgent.act()` samples a categorical action during exploration and chooses the argmax when `explore=False`.

`train_episode()` collects log probabilities, rewards, and entropies for one externally sized episode. `update()` computes discounted returns with `gamma=0.99`, normalizes them when there is more than one return, applies an entropy bonus (`entropy_coef=0.05`), and updates the Adam optimizer once. This is REINFORCE, not PPO; there is no critic or replay buffer.

Baselines are `AlwaysDirect`, `AlwaysRetrieve`, and seeded `RandomPolicy`. The runner evaluates baselines with stochastic/random actions as defined by each policy. RL evaluation calls `act(..., explore=False)` and does not update weights.

## Current experiment runner

`experiments/run_all.py`:

1. Loads KB-A and KB-B and creates aligned query pairs.
2. Trains `old_policy` on KB-A for `train_episodes` and saves `rl_old_policy_kb_a.pt`.
3. Evaluates the old policy deterministically on KB-A and KB-B.
4. Copies old weights into `adapted_policy`, trains it on KB-B, and saves `rl_adapted_policy_kb_b.pt`.
5. Trains `full_retrain_policy` from a new initialization on KB-B and saves `rl_full_retrain_policy_kb_b.pt`.
6. Evaluates the adapted and full-retrain policies deterministically on KB-B.
7. Evaluates each of the three baselines on both KBs.
8. Computes each RL policy's KB-B approval score against `old_policy`:

```text
(accuracy delta * 10) + (average reward delta) - (retrieval cost delta)
```

Scores greater than zero are marked `APPROVED`; otherwise `DECLINED`. Only the adapted and full-retrain KB-B rows receive an approval status; other rows use `N/A`.

The runner writes 10 CSV rows: old policy on A/B, adapted policy on B, full retrain on B, and three baselines on A/B. Each row contains `accuracy`, `average_reward`, `retrieval_rate`, `retrieval_cost`, drifted/stable accuracy fields, and `training_average_reward` for trained RL rows. Evaluation metrics are computed from actual `info` records by `summarize()`.

Outputs:

- `results/metrics/summary.csv`
- `results/metrics/config.json`
- reward plots under `results/figures/`
- PyTorch checkpoints under `results/checkpoints/`

## Configuration

`configs/default.yaml` currently defines:

```yaml
seed: 7
corpus_dir: data/documentation
knowledge_bases: [kb_a, kb_b]
top_k: 3
use_semantic_embeddings: true
embedding_model: all-MiniLM-L6-v2
allow_embedding_fallback: false
train_episodes: 1000
learning_rate: 0.001
retrieval_cost: 0.10
correct_reward: 1.0
incorrect_reward: -1.0
recovery_threshold: 0.90
rolling_window: 30
```

Only `seed`, `corpus_dir`, `top_k`, `train_episodes`, `learning_rate`, and `retrieval_cost` are currently consumed by `run_all.py`. `knowledge_bases`, embedding settings, reward settings, `recovery_threshold`, and `rolling_window` are present but not wired into the active runner.

## Tests and commands

From the repository root:

```powershell
$env:PYTHONPATH='.'
python -m pytest -q tests
python experiments/run_all.py
```

The current suite contains 17 passing tests covering chunk metadata, explicit KB selection, drift pairing, mismatch errors, retrieval metadata, environment behavior, caching, deterministic RL evaluation metrics, frozen evaluation, adaptation weight updates, and full-retrain smoke behavior.

## Current limitations and next work

- The corpus and generator are synthetic/documentary placeholders; the mock generator measures exact retrieval/policy behavior, not language quality.
- The active runner relies on `Retriever` defaults because the YAML embedding settings are not passed through `make_env()`; explicit embedding-model/fallback configuration remains future wiring work.
- The environment still reconstructs retrievers on reset, but shared text-vector caching prevents repeated transformer work. The benchmark should be run with `.venv\Scripts\python.exe` so it does not depend on an unrelated system Python installation.
- Retrieval diagnostics exist in `src/evaluation/metrics.py`, but the runner does not currently write per-query retrieval diagnostics.
- The environment is stationary within each experiment arm; it does not apply scheduled drift during an episode.
- The approval score is a project heuristic, not a statistical significance test.
- Repeated seeds, per-step logs, recovery-time analysis, and richer plots remain future work.

Do not claim that adaptation improves KB-B performance until the generated evaluation CSV and deterministic comparison support that conclusion. Preserve the separation between policy, action, environment, retrieval, generation, and cache state.
