---
name: debug-gate-failure
description: Diagnose a failing evaluation gate or CI run in this repo. Use when `ragkit eval` exits 1, pytest project tests fail, or a metric regressed after a change.
---

# Debug a gate failure

1. Run `python -m ragkit eval --only <pNN>`; note which thresholds failed and read the failure analysis.
2. Classify each failing question:
   - retrieval miss: gold doc not in the top 5
   - ranking problem: gold doc present but not first
   - answer problem: right chunk, wrong sentences or missing keyword
   - refusal problem: `min_coverage` too strict (refused an answerable question) or too loose (answered an unanswerable one)
3. Inspect before changing code: print the retrieved chunks and `ragkit.text.coverage(question, chunk.text)` for the failing question.
4. Fix at the right level: chunking in `ragkit/chunk.py`, synonyms or stemming in `ragkit/text.py`, retrieval depth or fusion in `ragkit/index.py`, extraction in `ragkit/answer.py`, or the project's `min_coverage`.
5. Rerun all projects. A fix for one must not regress another.
6. Never lower `THRESHOLDS`, never edit a golden item to pass. If the question is genuinely wrong, say so and ask before changing it.

Done when the gate passes, other projects are unchanged, and the cause is explained in one or two sentences.
