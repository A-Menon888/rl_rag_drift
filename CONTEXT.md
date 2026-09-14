# Project Context: Documentation-Based RL-RAG

## Purpose

This is a standalone handoff for a collaborator or another LLM. It describes the current implementation of the RL-RAG research prototype using two manually prepared documentation snapshots.

The current research question is whether a policy trained on Knowledge Base A degrades on changed Knowledge Base B, and whether continuing training on B recovers performance.

> Train on KB A, evaluate the frozen old policy on KB B, adapt on KB B, and compare old versus adapted policy performance.

- Local hypothetical Markdown documentation corpus.
- Document loader and paragraph chunker with metadata.
Evaluation status:

- RL training reward is kept separately as `training_average_reward`.
- After training, the policy is evaluated with `explore=False`.
- RL rows contain accuracy, average reward, retrieval rate, and retrieval cost calculated from evaluation `info` records.
- Old-policy evaluation does not update weights; adaptation starts from the old checkpoint and trains only on KB B.

## Active architecture

```text
Documentation query
        |
        v
Observation/state
        |
        v
Policy chooses action
        |
        +-------------------+
        |                   |
   DIRECT / 0         RETRIEVE / 1
        |                   |
        cached answer or UNKNOWN     vector search over current KB
        |                   |
        +---------+---------+
                  v
          deterministic mock answer
                  |
                  v
          exact chunk ground truth
                  |
                  v
                reward
                  |
                  v
          RL policy update
```

The policy chooses the action; the environment executes it and calculates reward. Retrieval remains independent of the RL algorithm.

## Repository map

```text
configs/default.yaml           Active experiment configuration
data/documentation/*.md        Hypothetical Knowledge Base A
experiments/run_all.py         KB-A -> KB-B adaptation experiment runner
src/data/documents.py          Markdown loader, chunker, query builder
src/data/generator.py          Shared Query record; old synthetic helpers
src/data/drift.py              Legacy drift helpers, not used by current runner
src/retrieval/embeddings.py   Deterministic fallback embeddings
src/retrieval/retriever.py    Top-k retrieval and result scores
src/generation/base.py         AnswerGenerator interface
src/generation/mock.py         Deterministic answer generator
src/environment/rl_rag_env.py Gymnasium environment
src/agents/baselines.py        Baseline policies
src/agents/rl_agent.py         MLP policy and REINFORCE
src/evaluation/metrics.py      Aggregate metrics
src/evaluation/plots.py        Matplotlib plot helper
tests/test_documents.py        Documentation corpus tests
tests/test_core.py             Legacy data/retrieval tests
tests/test_env.py              Environment contract test
```

## Knowledge Base A

The active corpus is in `data/documentation/` and contains two fixed snapshot folders:

```text
data/documentation/kb_a/
data/documentation/kb_b/
```

Each snapshot contains the same documentation topics with manually introduced changes. The files are hypothetical placeholders and can later be replaced with real documentation.

KB-A and KB-B are selected explicitly by `load_knowledge_base()`; the active runner never generates drift.

The current topics are:

- `authentication.md`: token creation, bearer headers, expiry, and HTTP 401 behavior.
- `users.md`: user creation, retrieval, and deletion endpoints.
- `payments.md`: payment creation, status values, and idempotency keys.
- `deployment.md`: environment variables, production command, port, and health check.
- `database.md`: PostgreSQL version, migration command, connection URL, and pool size.

The files are intentionally understandable placeholders. They can later be replaced with actual software/API documentation without changing the RL code.

## Document loading and chunking

`src/data/documents.py` defines:

```python
@dataclass(frozen=True)
class DocumentChunk:
    text: str
    document_id: str
    source: str
    title: str
    chunk_id: int
```

`load_documents(corpus_dir, max_words=80)`:

1. Finds Markdown files in sorted order.
2. Reads the first Markdown heading as the title.
3. Splits the document into paragraphs.
4. Removes heading lines from retrieval text.
5. Splits paragraphs exceeding the word limit.
6. Preserves source filename, title, document ID, and chunk ID.

Each chunk exposes `value` as an alias for `text`. This preserves the existing `MockAnswerGenerator` contract without changing the environment or generator architecture.

## Documentation queries

`generate_document_queries(chunks)` creates one query per document chunk. Each query is represented by the existing shared `Query` record:

```python
Query(
    query_id,
    entity,              # source filename for compatibility
    attribute,           # document title for compatibility
    text,
    answer_by_version,
    direct_answer,
    affected_by_drift,
)
```

For each loaded snapshot:

- `answer_by_version` is `{0: chunk.text}`.
- `direct_answer` is `UNKNOWN`.
- `affected_by_drift` is `False` within that snapshot.

This means direct answering represents answering without consulting the documentation, while retrieval can return the exact chunk used as ground truth. The setup remains objective and deterministic without an LLM.

## Retrieval

`Retriever` accepts document chunks just as it previously accepted facts. It embeds each chunk's `.text`, searches the current corpus, and returns:

```python
RetrievalResult(
    fact=<DocumentChunk>,
    score=<float>,
)
```

The field remains named `fact` for compatibility, but its runtime value is now a `DocumentChunk`.

Embedding behavior:

- Default: deterministic hashed bag-of-token vectors.
- Optional: sentence-transformers support exists in `Embedder` but is not enabled by default.
- Optional: FAISS support exists but is not enabled by default.
- NumPy inner-product search is the default because it is lightweight and avoids a Windows FAISS/PyTorch OpenMP conflict.

The fallback is lexical/hash based, not a semantic embedding model. Retrieval quality is therefore a known limitation and must not be confused with RL quality.

## Answer generation

The current generator is `MockAnswerGenerator`:

- With retrieval context, return the first retrieved chunk's `.value`, which is its exact text.
- Without context, return `query.direct_answer`, currently `UNKNOWN`.

No LLM or external API is used. The current milestone evaluates retrieval correctness and policy behavior, not natural-language generation quality.

## Environment

`src/environment/rl_rag_env.py` defines `RLRAGEnv(gymnasium.Env)`.

### Actions

```text
0 = DIRECT
1 = RETRIEVE
```

### Observation

The observation has shape `(11,)`:

- First 7 values: first seven query-embedding dimensions.
- Value 8: most recent reward.
- Value 9: recent retrieval frequency over the last 20 actions.
- Value 10: current database version.
- Value 11: whether the current query has cached retrieved context.

Ground truth, the correct answer, and future drift information are not provided to the agent. Each A/B environment is stationary with one snapshot at version `0`; the experiment changes the corpus between evaluation/training phases rather than changing it inside an episode.

### Step behavior

The cache is a dictionary keyed by `query_id`. It is part of environment state and is cleared by every `reset()`, so training and evaluation episodes cannot leak cached answers into each other.

For each action, the environment:

1. Selects the current query.
2. Applies a scheduled drift event if one exists. There are none in the current KB-A run.
3. For action `0`, checks whether the current query has cached context.
4. Searches the current corpus only for action `1`.
5. Generates the deterministic answer from cached or retrieved context.
6. Caches retrieved context only when the retrieved answer is correct.
7. Compares the answer with the current query's ground truth.
8. Calculates reward and retrieval cost.
9. Updates recent reward/retrieval state.
10. Returns the next observation and an audit `info` dictionary.

The `info` dictionary includes query ID, action, answer, ground truth, correctness, reward, database version, drift event, drift rate field, retrieval cost, cache hit, and cache size.

### Reward

Default values:

```text
correct_reward = +1.0
incorrect_reward = -1.0
retrieval_cost = 0.10
```

Therefore:

| Result | Reward |
|---|---:|
| Direct and correct | +1.0 |
| Retrieve and correct | +0.9 |
| Direct and incorrect | -1.0 |
| Retrieve and incorrect | -1.1 |

The retrieval cost represents extra retrieval/context/token cost and prevents the objective from becoming simply “always retrieve.” A cached DIRECT answer has zero retrieval cost.

## RL implementation

`src/agents/rl_agent.py` contains a real Monte Carlo REINFORCE agent.

### Policy network

```text
11 input state values
 -> Linear(11, 32)
 -> Tanh
 -> Linear(32, 2)
 -> action logits
```

The two logits define a categorical distribution over DIRECT and RETRIEVE.

### Training path

`experiments/run_all.py` creates `RLAgent(...)` and calls:

```python
for _ in range(config["train_episodes"]):
    infos, loss = policy.train_episode(env)
```

`train_episode()` samples actions, collects log probabilities and rewards, then calls `update()` once at the end of the episode.

Inside `update()`:

```python
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

`optimizer.step()` changes the MLP weights. The loss uses discounted returns:

```text
G_t = r_t + gamma*r_(t+1) + gamma^2*r_(t+2) + ...
loss = -sum(log_probability(action_t) * normalized_return_t)
```

This is genuine policy-gradient learning, but intentionally a simple REINFORCE baseline. It is not PPO and has no critic, replay buffer, or generalized advantage estimation.

## Baseline policies

- `AlwaysDirect`: always action `0`.
- `AlwaysRetrieve`: always action `1`.
- `RandomPolicy`: seeded random action selection.
- `RLAgent`: sampled neural policy during training; weights update from episode returns.

Policy means the decision strategy. Action means one concrete decision produced by that strategy.

## Current experiment runner

`experiments/run_all.py` now:

1. Loads both fixed snapshots with `load_knowledge_base()`.
2. Creates separate documentation queries for A and B.
3. Trains the old RL policy on KB A.
4. Saves `rl_old_policy_kb_a.pt`.
5. Evaluates that frozen policy deterministically on KB A and KB B.
6. Creates an adapted policy from the old policy state dict.
7. Continues RL training on KB B.
8. Saves `rl_adapted_policy_kb_b.pt`.
9. Evaluates the adapted policy deterministically on KB B.
10. Evaluates retained baseline policies on both snapshots.
11. Writes comparison metrics and reward figures.

The old synthetic generator and active drift loops were removed from the runner. The legacy drift module remains unused because KB-A and KB-B are manually prepared snapshots.

The active default run produces nine rows: old policy on A and B, adapted policy on B, and three baselines on both A and B.

## Evaluation

Baseline policies call `run_policy()` and receive summaries from environment `info` records. The RL branch follows the same metric path after training:

```text
train RL policy
 -> save checkpoint
 -> evaluate with explore=False
 -> collect actual info records
 -> summarize actual accuracy/reward/retrieval metrics
```

The summary distinguishes `training_average_reward` from evaluation `average_reward`. The important research comparison is `old_policy / kb_b` versus `adapted_policy / kb_b`.

The latest run produced:

| Policy | KB | Accuracy | Average reward | Retrieval rate | Retrieval cost |
|---|---|---:|---:|---:|---:|
| old_policy | kb_a | 0.625 | 0.150 | 1.000 | 0.100 |
| old_policy | kb_b | 0.625 | 0.150 | 1.000 | 0.100 |
| adapted_policy | kb_b | 0.625 | 0.150 | 1.000 | 0.100 |

In this short run, adaptation did not improve the measured KB-B metrics. That is a valid result, not a failure to hide: the current experiment does not show recovery yet.

## Configuration and commands

`configs/default.yaml` currently contains:

```yaml
seed: 7
corpus_dir: data/documentation
top_k: 3
train_episodes: 20
learning_rate: 0.01
retrieval_cost: 0.10
correct_reward: 1.0
incorrect_reward: -1.0
recovery_threshold: 0.90
rolling_window: 30
```

From the repository root:

```powershell
$env:PYTHONPATH='.'
python -m pytest -q tests
python experiments/run_all.py
```

Use the project `.venv` interpreter if the shell is not already activated.

Outputs:

- `results/metrics/summary.csv`
- `results/metrics/config.json`
- `results/figures/*_kb_*_reward.png`
- `results/checkpoints/rl_old_policy_kb_a.pt`
- `results/checkpoints/rl_adapted_policy_kb_b.pt`

## Tests

The tests cover:

1. Existing synthetic compatibility behavior.
2. Document loading and metadata preservation.
3. Documentation query ground truth and `UNKNOWN` direct answers.
4. Retrieval of a documentation chunk with source metadata and score.
5. Gymnasium environment reset/step behavior.
6. Uncached DIRECT cannot answer documentation queries.
7. Successful RETRIEVE caches context for a later DIRECT action at zero retrieval cost.
8. Deterministic RL evaluation returns computed metrics.
9. Frozen old-policy evaluation does not update weights.
10. KB-B adaptation starts from old weights and updates them.

Run:

```powershell
python -m pytest -q tests
```

## Deferred work

Do not implement these as part of the current documentation-corpus milestone:

- Automatic drift generation beyond the manually prepared KB-A/KB-B snapshots.
- RL algorithm redesign.
- LLM answer generation.
- Large-scale document ingestion.

Recommended later order:

1. Replace hypothetical files with selected real documentation.
2. Improve or validate semantic embeddings.
3. Add per-step logs and repeated random seeds.
4. Add additional manually prepared snapshots or controlled drift only if needed.
5. Measure post-change performance drop and recovery time.

## Handoff rules

- Preserve the separation between policy, action, environment, retrieval, and generation.
- Do not give ground truth to the agent observation.
- Keep cache state in the environment, not inside the policy.
- Keep retrieval cost in the reward.
- Do not silently replace RL with a heuristic.
- Do not report placeholder RL metrics as measured results.
- Keep the deterministic fallback path for offline reproducibility.
- Run focused tests after data, retrieval, environment, or agent changes.
- Treat claims about RL adaptation as unverified until deterministic post-training evaluation supports them.

## Short summary

The project now uses two small hypothetical Markdown documentation snapshots, KB A and manually changed KB B. Documents are loaded and chunked with metadata, queries are grounded to exact chunks, and the existing RL-RAG pipeline trains on A, evaluates the frozen old policy on B, continues training from the old weights on B, and evaluates the adapted policy. The RL algorithm, action space, cache, and reward function are unchanged. The latest short run did not improve KB-B metrics after adaptation, so longer training or better task coverage is needed before claiming recovery.
