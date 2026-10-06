# Evaluation

## Golden sets

`projects/*/data/golden.jsonl`, one JSON object per line:

```json
{"question": "...", "source_ids": ["doc-or-part-id"], "answer_keywords": ["25"]}
```

An empty `source_ids` marks an unanswerable question: the pipeline should refuse. For 12.1 items carry `gold_sql` instead, or nothing for questions that must be refused.

## Metrics

| Metric | Definition |
|---|---|
| recall@5 | A gold document appears in the top 5 retrieved |
| precision@1 | The first retrieved document is a gold document |
| MRR | Mean of 1 / rank of the first gold document |
| answer_correctness | Answer is not a refusal and contains every keyword |
| citation_accuracy | Cited document is a gold document (non-refused answerable questions) |
| refusal_accuracy | Refuses exactly when the question is unanswerable |
| faithfulness_lexical | Share of answer terms present in the grounding text. A proxy only |
| p95_latency_s | 95th percentile end-to-end time per question |
| execution_accuracy (12.1) | Result set equals the result set of the gold SQL |
| safety_block_rate (12.1) | Hostile SQL statements rejected by the validator and read-only authorizer |

Retrieval metrics are computed over answerable questions only.

## Adding an LLM judge (recommended next step)

1. Label 40 to 60 answers by hand as supported or unsupported.
2. Write a judge prompt that lists the answer's claims and checks each against the context.
3. Run the judge on the same answers, compute agreement with your labels, and tighten the prompt until agreement is at least 0.7.
4. Report the judged faithfulness next to the lexical proxy. Pin the judge model and prompt.

## Regression gate

Thresholds live in each `pipeline.py` (`THRESHOLDS`). CI fails when the main configuration drops below any minimum. Do not edit expected answers or thresholds to make a run pass.
