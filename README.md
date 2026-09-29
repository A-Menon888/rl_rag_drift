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

`data/documentation/` holds three Markdown snapshots: `kb_0` (what the frozen generator saw before its training cutoff, used only as closed-book memory), `kb_a` (the deployed knowledge base) and `kb_b` (the drifted knowledge base). `facts.yaml` names each fact's answer slot with a regex on its owning page, plus hand-written questions. Values, drift types (unchanged, modified, contradicted, added, removed) and memory status (correct, stale, unknown) are extracted from the snapshots, not labelled by hand.

One episode is one question. The Gymnasium environment has three actions: SEARCH_MORE reveals the next unseen chunks of the question's ranking (paged search, with a per-question budget and a cost per search), ANSWER asks the frozen reader to answer from the evidence so far (or from memory if there is none), and GIVE_UP abstains, which is correct for questions the documentation does not answer. The observation is the query embedding plus features computed from the retrieved evidence and the generator's own outputs; ground truth is used only for reward.

Retrieval uses sentence-transformers with FAISS when available. A deterministic hashed-vector fallback keeps the initial experiment runnable without downloading model weights. The mock generator is a deterministic slot reader: closed-book it returns the KB-0 memory value (or abstains), and open-book it returns the first value for the question's slot found in the retrieved chunks, in rank order.

Baselines are answer-directly, search-once, search-all and random, alongside a trainable REINFORCE MLP with action masking. Results intentionally do not assume RL wins. The CSV is the source of truth for analysis.

## Limitations and extensions

The corpus is hypothetical documentation with an exact-match mock generator, so it measures retrieval-policy behavior rather than open-ended language quality. The current run is stationary: it uses Knowledge Base A only and does not claim drift adaptation. To use real data, replace the Markdown files while keeping the loader's chunk metadata and deterministic query answers.
