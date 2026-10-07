---
name: run-eval
description: Run the RAG capstone evaluation and summarise results. Use when asked about current quality, metrics, whether CI will pass, or before and after any retrieval, prompt or data change.
---

# Run the evaluation

1. Install if needed: `pip install -e ".[dev]"`.
2. Run `python -m ragkit eval --out results --site site` (add `--only p03` for one project).
3. Read `results/RESULTS.md`. For each project give: main configuration, PASS or FAIL, the headline metrics, and the failure analysis items.
4. Flag weak evidence: scores of 1.0 on tiny corpora, `Faithful*` being a lexical proxy, offline stand-ins for models.
5. If something fails, switch to the `debug-gate-failure` skill. Never edit thresholds or golden answers to get a pass.

Done when every project is PASS or has a named cause for FAIL.
