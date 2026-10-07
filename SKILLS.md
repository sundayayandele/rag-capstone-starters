# SKILLS.md

Playbooks for working in this repo with Claude Code (or by hand). The first five are also installed as Claude Code skills in `.claude/skills/<name>/SKILL.md`, so Claude loads them automatically when the task matches. The rest are written here as checklists.

| Skill | Use it when | Output |
|---|---|---|
| [run-eval](#run-eval) | You want to know where quality stands | Results table and a short summary |
| [add-golden-questions](#add-golden-questions) | You have new questions or real user queries | More golden items, metrics before and after |
| [add-project](#add-project) | You want a new capstone MVP (for example 01.2 or 08.1) | A new `projects/pNN_*` package wired into CLI, tests and CI |
| [debug-gate-failure](#debug-gate-failure) | CI or `ragkit eval` fails | Root cause and a fix, not a threshold change |
| [swap-real-models](#swap-real-models) | You want neural embeddings, a cross-encoder or an LLM | Comparison table: offline defaults against real models |
| [add-llm-judge](#add-llm-judge) | You want a better faithfulness score | Calibrated judge and agreement figure |
| [publish-results](#publish-results) | You want the results page live | Pages enabled, CI green, link in README |

## run-eval

1. `pip install -e ".[dev]"` if needed, then `python -m ragkit eval --out results --site site`.
2. Read `results/RESULTS.md`. For each project report: main config, headline metrics, any failures listed in the failure analysis.
3. Say which numbers are weak evidence (score 1.0 on a tiny corpus, lexical faithfulness proxy).
4. Done when: every project has PASS or a named cause for FAIL.

## add-golden-questions

1. Ask which project and where the questions come from (real queries are best).
2. Append items to `projects/<p>/data/golden.jsonl`: `question`, `source_ids` (empty for unanswerable), `answer_keywords`. For p12 use `gold_sql` instead. Include about 20 percent unanswerable questions.
3. Run `python -m ragkit eval --only <p>` before and after; do not change existing items.
4. If a new question fails, analyse it; fix the system, do not edit the question to pass.
5. Done when: the table shows before and after, and failures are explained.

## add-project

1. Read the capstone docs for the project in RAG-Designs (`01_architecture.md`, `04_build_guide.md`, `06_testing_plan.md`).
2. Create `projects/pNN_name/` with `__init__.py`, `pipeline.py` (`PROJECT`, `TITLE`, `THRESHOLDS`, a pipeline class with `ask`, `golden()`, `run()`, `make()`), and `data/` with a small corpus and a golden set of at least 20 questions.
3. Register it in `PROJECTS` in `ragkit/cli.py`.
4. Add tests: gate passes, results deterministic, one pattern-specific behaviour test.
5. Update the table in `README.md` and the layout in `CLAUDE.md`.
6. Done when: `pytest -q` and `python -m ragkit eval` pass and the results table shows at least two configurations to compare.

## debug-gate-failure

1. Run `python -m ragkit eval --only <p>` and read the failed checks and the failure analysis.
2. Classify each failing question: retrieval miss (gold doc not in top 5), ranking problem (gold doc present but not first), answer problem (right chunk, wrong sentences or keywords), refusal problem (support threshold too strict or too loose).
3. Check the chunk text and the `coverage` score for that question before changing code.
4. Fix at the right level (chunking, synonyms, retrieval depth, `min_coverage`, extractor) and rerun all projects.
5. Never lower `THRESHOLDS` or edit the golden item. Done when: gate passes and no other project regressed.

## swap-real-models

1. `pip install -e ".[ml]"` (needs a machine with enough memory; Termux may not cope, use a Codespace or CI).
2. Run `RAG_EMBEDDER=st python -m ragkit eval --out results-st`, then `RAG_RERANKER=ce` for p02, then `RAG_LLM=github` or `anthropic` with the relevant token.
3. Compare against the offline results in one table. Report cost, latency and any rate limits hit.
4. State clearly that these paths were not exercised when the repo was generated; fix any breakage you find.
5. Done when: a comparison table is committed under `docs/` with the model names and versions.

## add-llm-judge

1. Label 40 to 60 answers by hand as supported or unsupported.
2. Add a judge function in `ragkit/evaluate.py` that lists claims and checks each against the context; call it only when `RAG_JUDGE=1`.
3. Compute agreement with your labels; iterate on the prompt until it is at least 0.7. Pin model and prompt.
4. Report judged faithfulness beside `faithfulness_lexical`.
5. Done when: agreement figure and prompt are committed and documented in `docs/EVALUATION.md`.

## publish-results

1. Repo Settings, Pages, Source: GitHub Actions.
2. Run the CI workflow on `main`; open the Pages URL.
3. Add the link to `README.md`.
4. Done when: the page shows all five projects with PASS and the README links to it.
