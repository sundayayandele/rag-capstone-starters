# RAG Capstone Starters

Runnable starters for five RAG capstone MVPs from the [RAG-Designs](https://github.com/sundayayandele/RAG-Designs) programme. Each has a sample corpus, a golden set, source code, an evaluation script and tests. CI runs the evaluation, applies a regression gate and publishes a results table. A Claude Code workflow lets you drive the build from issues.

| ID | Project | Pattern | What the evaluation shows |
|---|---|---|---|
| 01.1 | Handbook Q&A bot | Naive RAG | Baseline retrieval, citations, refusal on unanswerable questions |
| 02.1 | IT ticket search | Retrieve-and-Rerank | First-stage only versus first stage plus reranker |
| 03.1 | Part-code finder | Hybrid RAG | BM25 only, dense only, and RRF fusion |
| 09.1 | Compliance Q&A | Corrective RAG | No correction versus grade, rewrite and trusted fallback |
| 12.1 | Natural-language analytics | Structured RAG | Execution accuracy, read-only SQL safety, refusal of destructive requests |

Design, rationale, plan, MVP spec, testing plan and rollout guide for each project live in the capstone folders of RAG-Designs (`RAG-Capstones/<pattern>/level-1-...`).

## Quick start

```bash
pip install -e ".[dev]"
python -m ragkit eval --out results --site site   # run all five, write results/RESULTS.md and site/index.html
pytest -q                                         # 48 tests
python -m ragkit ask p01 "How many days of annual leave do I get?"
python -m ragkit ask p12 "Top 3 customers by revenue"
```

Runs fully offline in about a second. Phone setup: [docs/TERMUX.md](docs/TERMUX.md).

## What runs offline, and what is a stand-in

The defaults need no downloads and no API keys so CI is fast and deterministic. They are stand-ins, and the numbers they produce prove the **harness and the pipeline logic**, not production quality.

| Component | Offline default | Real option |
|---|---|---|
| Embeddings | Hashing embedder over words and character trigrams, with a small synonym lexicon | `RAG_EMBEDDER=st` (sentence-transformers, e.g. bge-small), `pip install -e ".[ml]"` |
| Reranker | Lexical coverage and bigram scorer | `RAG_RERANKER=ce` (cross-encoder) |
| Answer generation | Extractive: best-supported sentences with a citation, refuses when support is weak | `RAG_LLM=github` (GitHub Models) or `RAG_LLM=anthropic` |
| Faithfulness | Lexical overlap proxy (`Faithful*`) | Add an LLM judge (see [docs/EVALUATION.md](docs/EVALUATION.md)) |
| 12.1 SQL generator | Rule-based, supports about a dozen question shapes | LLM writes SQL from retrieved schema and metric cards |

The neural embedder, cross-encoder and both hosted LLM providers are wired in but were **not exercised** when this repo was generated (the build environment had no access to model hosts). Test them in your own environment before trusting them.

### Honest limits

- The corpora are tiny and synthetic. Several scores are 1.0 because the tasks are easy; that is expected and not evidence of production quality.
- 02.1 shows only a small reranking effect with the lexical reranker. A real cross-encoder on a larger, noisier corpus is where the uplift appears.
- 09.1 questions were written around acronym and out-of-corpus failures, so the correction uplift is by construction. Replace the golden set with questions from your own users.
- 12.1 in offline mode tests the harness and safety layer, not an LLM's SQL ability.

## Layout

```
ragkit/                 shared library: text, chunk, embed, index (BM25, dense, RRF), rerank, llm, answer, evaluate, cli, site
projects/
  p01_handbook_qa/      pipeline.py, data/corpus/*.md, data/golden.jsonl
  p02_ticket_rerank/
  p03_part_finder/      data/parts.jsonl
  p09_compliance_crag/  data/corpus, data/fallback, data/glossary.json
  p12_nl_analytics/     db.py (sample DB + validated read-only executor), generator.py, pipeline.py
tests/                  48 tests: library, answers, project gates, CRAG, SQL safety, workflow files
.github/workflows/      ci.yml, claude.yml
docs/                   TERMUX.md, EVALUATION.md
CLAUDE.md               instructions for Claude Code
SKILLS.md               playbooks; the first five are installed as Claude Code skills in .claude/skills/
.claude/skills/         run-eval, add-golden-questions, add-project, debug-gate-failure, swap-real-models
```

## Evaluation and the regression gate

`python -m ragkit eval` runs each project over its golden set, compares configurations side by side, checks the main configuration against the thresholds in the project's `pipeline.py`, and exits non-zero on failure. Metrics: recall@5, precision@1, MRR, answer correctness (keyword match), citation accuracy, refusal accuracy, a lexical faithfulness proxy and p95 latency. Details in [docs/EVALUATION.md](docs/EVALUATION.md).

## CI

`.github/workflows/ci.yml` on every push and pull request: install, run pytest, run the evaluation and gate, write the results table to the job summary, upload `results/` and `site/` as artifacts. On `main` it deploys the results page to GitHub Pages (enable **Settings, Pages, Source: GitHub Actions** once). A manual run with **run_ml** ticked also starts the `evaluate-ml` job: CPU-only torch, sentence-transformers embeddings and a cross-encoder reranker, with models cached between runs. It is informational (it does not block merges) and its results appear in the job summary and the `rag-results-ml` artifact. The manual run also accepts `llm=github` to evaluate with GitHub Models using the built-in token (rate-limited; the job has `models: read`) or `llm=anthropic` with the `ANTHROPIC_API_KEY` secret.

## Claude Code on GitHub

`.github/workflows/claude.yml` runs Claude Code when someone comments `@claude ...` on an issue or pull request, or opens an issue that mentions it.

1. Add the repository secret `ANTHROPIC_API_KEY` (Settings, Secrets and variables, Actions), or run `/install-github-app` in Claude Code and follow the prompts.
2. Open an issue, for example: `@claude add 20 harder questions to projects/p02_ticket_rerank/data/golden.jsonl, run python -m ragkit eval and open a PR with the results table`.
3. Review and merge the pull request it opens; CI runs the gate on it.

Check the [claude-code-action](https://github.com/anthropics/claude-code-action) README for current inputs. Claude API usage is billed separately. Restrict who can trigger the workflow and review every PR it opens.

## Ideas for Claude Code tasks

- Replace the golden sets with questions and documents from your own domain.
- Turn on `RAG_EMBEDDER=st` and `RAG_RERANKER=ce` and compare the tables.
- Add an LLM judge for faithfulness and calibrate it on 40 hand-labelled answers.
- Add a second corpus type (PDF parsing) to 01.1.
- Build the L2 versions from the capstone docs, starting with 01.2 (multilingual FAQ bot).

## Next steps beyond free tier

L2 and L3 projects need hosting, real models and security work described in each project's `07_enterprise_rollout.md`. This repo covers only the MVP evidence for the five L1 projects.
