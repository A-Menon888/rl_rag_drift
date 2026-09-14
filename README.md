# RL-RAG under Knowledge Base Drift

A small, reproducible research prototype for testing whether a neural reinforcement-learning policy can learn when retrieval is worth its cost in a local documentation knowledge base. It is an independent implementation and does not use AionRAG.

## Run

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH='.'
pytest -q
python experiments/run_all.py
```

Outputs are written to `results/metrics/`, `results/figures/`, and `results/checkpoints/`.

## Design

Knowledge Base A is the local Markdown corpus in `data/documentation/`, covering authentication, users, payments, deployment, and database configuration. The document loader turns paragraphs into metadata-preserving chunks and creates deterministic factual queries from those chunks. Knowledge Base B and drift adaptation are intentionally deferred.

The Gymnasium environment has two actions: `0 = DIRECT` and `1 = RETRIEVE`. The observation contains a compact query embedding, recent reward, recent retrieval frequency, and database version. Ground truth is used only to calculate reward. Correct answers earn +1, incorrect answers earn -1, and retrieval costs 0.10 by default.

Retrieval uses sentence-transformers with FAISS when available. A deterministic hashed-vector fallback keeps the initial experiment runnable without downloading model weights. The mock generator returns the current top retrieved fact or the query's stale direct answer.

Baselines are always-direct, always-retrieve, random, and a trainable REINFORCE MLP. Results intentionally do not assume RL wins. The current runner reports accuracy, reward, retrieval rate, and retrieval cost and produces reward plots; the CSV is the source of truth for analysis.

## Limitations and extensions

The corpus is hypothetical documentation with an exact-match mock generator, so it measures retrieval-policy behavior rather than open-ended language quality. The current run is stationary: it uses Knowledge Base A only and does not claim drift adaptation. To use real data, replace the Markdown files while keeping the loader's chunk metadata and deterministic query answers.
