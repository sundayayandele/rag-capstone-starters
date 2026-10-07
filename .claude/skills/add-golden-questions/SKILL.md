---
name: add-golden-questions
description: Add questions to a project's golden set and measure the effect. Use when the user supplies real queries, wants harder questions, or asks to improve evaluation coverage for p01, p02, p03, p09 or p12.
---

# Add golden questions

1. Confirm the project and the source of the questions (real user queries beat invented ones).
2. Append JSON lines to `projects/<project>/data/golden.jsonl`:
   `{"question": "...", "source_ids": ["doc-id"], "answer_keywords": ["key fact"]}`.
   Unanswerable questions use `"source_ids": []`. For p12 use `{"question": "...", "gold_sql": "SELECT ..."}`; refusals have no `gold_sql`. Keep about 20 percent unanswerable.
3. Run `python -m ragkit eval --only <pNN>` before and after. Never change existing items.
4. If a new question fails, analyse why (retrieval, ranking, extraction, refusal). Fix the system, not the question.
5. Report a before and after table and list the new failures.

Done when the table is shown and every failure has an explanation.
