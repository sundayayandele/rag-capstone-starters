---
name: add-project
description: Add a new capstone MVP package under projects/ (for example 08.1 adaptive RAG or an L2 version) wired into the CLI, tests and CI. Use when asked to build another pattern or level in this repo.
---

# Add a project

1. Read the project's capstone docs in the RAG-Designs repo (`01_architecture.md`, `04_build_guide.md`, `06_testing_plan.md`). If they are not available, ask the user.
2. Create `projects/pNN_name/` with `__init__.py`, `pipeline.py` and `data/`.
   `pipeline.py` must define `PROJECT`, `TITLE`, `THRESHOLDS`, a pipeline class with `ask(question)` returning `ragkit.answer.Answer`, `golden()`, `run()` (use `ragkit.evaluate.evaluate_configs` with at least two configurations to compare) and `make()`. Copy `projects/p03_part_finder/pipeline.py` as the template.
3. Data: a small public or synthetic corpus and a golden set of at least 20 questions, about 20 percent unanswerable.
4. Register the project in `PROJECTS` in `ragkit/cli.py`.
5. Tests: the gate passes, results are deterministic, and one test for the pattern's key behaviour.
6. Update the table in `README.md` and the map in `CLAUDE.md`.
7. Keep defaults offline and deterministic; real models only behind environment variables.

Done when `pytest -q` and `python -m ragkit eval` pass and the results show a meaningful comparison.
