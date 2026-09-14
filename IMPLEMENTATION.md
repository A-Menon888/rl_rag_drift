
# RL-RAG under Knowledge Base Drift

## 1. Intro

This project is a research prototype for studying whether a reinforcement-learning agent can decide when retrieval is worth using while its knowledge base changes over time.

The agent receives a synthetic question and chooses one of two actions:

- `0 / DIRECT`: answer from the query's original direct knowledge.
- `1 / RETRIEVE`: search the current knowledge base and answer from the top retrieved fact.

The answer is compared with the ground truth for the current database snapshot. The agent receives reward and can update its policy. The knowledge base can drift through updates, additions, deletions, or contradictions.

The project is intentionally synthetic and deterministic. This makes it useful for testing policy behavior and adaptation before introducing a real dataset or a local language model.

## 2. What has been implemented

### Data and drift

Implemented in `src/data/`:

- `Fact` records with entity, attribute, value, validity interval, version, and document ID.
- Synthetic database generation for 40 entities and five attributes per entity by default.
- Synthetic query generation with deterministic answer mappings.
- Snapshot drift operations:
  - update
  - addition
  - deletion
  - contradiction
- Configurable drift rates: 0%, 10%, 25%, and 50%.
- Ground-truth recalculation for every query and database snapshot.

### Retrieval and generation

Implemented in `src/retrieval/` and `src/generation/`:

- Embedding interface.
- Deterministic hashed-vector embedding fallback, so the experiment can run without downloading model weights.
- Optional `sentence-transformers` support.
- Optional FAISS support.
- NumPy inner-product search is the default because it avoids Windows OpenMP conflicts between FAISS and PyTorch.
- Retrieval results retain the fact metadata and similarity score.
- Deterministic mock answer generator.

### RL environment

Implemented in `src/environment/rl_rag_env.py`:

- `RLRAGEnv(gymnasium.Env)`.
- Two-action discrete action space.
- Observation includes query embedding, recent reward, recent retrieval frequency, and database version signal.
- The environment returns query ID, action, answer, ground truth, correctness, reward, database version, drift event, drift rate field, and retrieval cost in `info`.
- Reward defaults:
  - correct answer: `+1.0`
  - incorrect answer: `-1.0`
  - retrieval cost: `-0.10`

### Policies

Implemented in `src/agents/`:

- Always Direct.
- Always Retrieve.
- Random Policy.
- Trainable neural MLP policy using REINFORCE.
- PyTorch checkpoints are written for each drift condition.

### Experiments and outputs

Implemented in `experiments/run_all.py`:

- Stationary condition at 0% drift.
- Drift-rate sweep over 0%, 10%, 25%, and 50%.
- Drift-type sweep for update, addition, deletion, and contradiction.
- Baseline comparison.
- Reward plots saved in `results/figures/`.
- Metrics saved to `results/metrics/summary.csv`.
- Configuration saved to `results/metrics/config.json`.
- RL checkpoints saved to `results/checkpoints/`.

## 3. Verified run

Commands used:

```powershell
$env:PYTHONPATH='.'
C:/Users/Aayush/AppData/Local/Programs/Python/Python313/python.exe -m pytest -q tests
C:/Users/Aayush/AppData/Local/Programs/Python/Python313/python.exe experiments/run_all.py
```

Observed results:

- Tests: `3 passed`.
- Full experiment: completed successfully.
- Metrics: `52` result rows written.
- Checkpoints: 13 RL checkpoint files written.
- Figures: reward plots written for the policies and drift conditions.

The direct Phase 1 drift check was also verified independently:

```text
Before drift: Alice
After update drift: updated-Alice
```

This confirms that the same query can have different deterministic ground truth in different snapshots.

## 4. Current measured baseline results

These values come from `results/metrics/summary.csv`. Accuracy and average reward are proportions/means over the final evaluation episode for each baseline condition.

| Drift rate | Drift type | Policy | Accuracy | Average reward | Retrieval rate |
|---:|---|---|---:|---:|---:|
| 0% | update | Always Direct | 1.000 | 1.000 | 0.000 |
| 0% | update | Always Retrieve | 0.073 | -0.953 | 1.000 |
| 0% | update | Random | 0.533 | 0.016 | 0.507 |
| 10% | update | Always Direct | 0.953 | 0.907 | 0.000 |
| 25% | update | Always Direct | 0.873 | 0.747 | 0.000 |
| 50% | update | Always Direct | 0.740 | 0.480 | 0.000 |

The baseline trend is clear: as drift rate rises, the stale direct answer becomes less reliable. Always Retrieve performs poorly in this initial synthetic setup because the lightweight hashed retriever is not yet a semantic retriever and often returns the wrong fact. This is a useful diagnostic result, not evidence that retrieval is inherently bad.

For the drift-type sweep, the baseline results show the same broad rate-dependent degradation. The current synthetic changes are generated from the same fact pool and the mock generator uses exact top-result retrieval, so drift type differences are not yet a strong scientific conclusion.

## 5. Important interpretation caveat

The current runner trains the RL policy and records its reward trajectory, but it writes placeholder values of zero for RL accuracy, retrieval rate, and retrieval cost in the final CSV row construction. Therefore:

- Present the baseline rows as measured results.
- Present the RL reward values as training-reward values only.
- Do not claim the CSV currently demonstrates RL accuracy or retrieval-policy adaptation.
- The existence of RL checkpoints proves that a trainable RL component ran, but not that it outperformed a baseline.

This limitation is visible in the implementation and should be fixed before making a strong claim about the research hypothesis. The correct next engineering step is to evaluate the trained RL policy with `explore=False`, collect fresh `info` records, and calculate the same metrics as the baselines.

## 6. Suggested mentor presentation

### Slide/demo order

1. **Research question**
   - Can an RL agent learn when retrieval is useful?
   - What happens when the knowledge base drifts?

2. **System diagram**

   ```text
   Query -> RL action -> Direct or Retrieve -> Mock answer
                                      |
                              Current KB snapshot
                                      |
                     Ground truth -> reward -> policy update
   ```

3. **Show the data model**
   Open `src/data/generator.py` and point out the `Fact` fields: entity, attribute, value, validity, version, document ID.

4. **Show drift changing an answer**
   Run:

   ```powershell
   $env:PYTHONPATH='.'
   C:/Users/Aayush/AppData/Local/Programs/Python/Python313/python.exe -c "from src.data.generator import *; from src.data.drift import *; f=generate_database(seed=1,entities=2); q=generate_queries(f,1,seed=1); f1,_=apply_drift(f,1,1.0,'update',seed=1); x=update_queries(q,{0:f,1:f1})[0]; print('before:',x.answer(0)); print('after:',x.answer(1))"
   ```

5. **Show the environment contract**
   Open `src/environment/rl_rag_env.py`. Explain that ground truth is not part of the observation; it is used only after answer generation to calculate reward.

6. **Show baseline results**
   Open `results/metrics/summary.csv` and compare Always Direct, Always Retrieve, and Random. Emphasize retrieval cost and the degradation from 0% to 50% drift.

7. **Show generated artifacts**
   Open one plot from `results/figures/` and one checkpoint from `results/checkpoints/`. Explain that the files are generated by `experiments/run_all.py`.

8. **State the limitation honestly**
   The RL training loop is present and runs, but RL evaluation metrics need one more implementation pass before claiming adaptation. This demonstrates research discipline and gives a clear next milestone.

## 7. What the code does during one environment step

1. Select the current query.
2. Apply a scheduled database drift event if the step reaches the drift point.
3. Rebuild the retriever for the current snapshot.
4. Use the selected action.
5. Retrieve top-k facts only for action `1`.
6. Generate a deterministic answer.
7. Look up the current snapshot's ground truth.
8. Calculate correctness and reward.
9. Store recent reward and retrieval history.
10. Return the next observation and an audit record in `info`.

## 8. Limitations

- Synthetic facts are easier than real-world knowledge.
- The mock generator is exact-match and does not measure language generation quality.
- The fallback embedding is lexical/hash based, not genuinely semantic.
- FAISS and transformer embeddings are optional rather than the default runtime path.
- The current experiment stores aggregate CSV rows rather than every per-step interaction log.
- Recovery time and the requested richer accuracy/retrieval/adaptation plots are not yet implemented.
- RL evaluation metrics must be corrected before drawing conclusions about policy adaptation.

## 9. Recommended next milestone

Add an explicit evaluation function:

```text
train RL policy -> save checkpoint -> load checkpoint
-> run deterministic evaluation with exploration disabled
-> collect info records -> calculate accuracy, reward, retrieval rate, cost
-> calculate pre-drift drop, post-drift final performance, and recovery time
```

After that, replace the fallback retriever with a fixed local sentence-transformers model and repeat the same experiments over several random seeds. Only then should the project support claims about whether RL adapts better than the baselines.
