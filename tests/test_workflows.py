from pathlib import Path

import yaml

WF = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def _load(name):
    d = yaml.safe_load((WF / name).read_text())
    d["on"] = d.get("on", d.get(True))  # PyYAML parses the bare key `on` as True
    return d


def test_ci_workflow_shape():
    d = _load("ci.yml")
    assert {"push", "pull_request", "workflow_dispatch"} <= set(d["on"])
    assert "evaluate" in d["jobs"]
    steps = " ".join(str(s) for s in d["jobs"]["evaluate"]["steps"])
    assert "pytest" in steps and "ragkit eval" in steps and "upload-artifact" in steps


def test_ci_has_manual_ml_job():
    d = _load("ci.yml")
    job = d["jobs"]["evaluate-ml"]
    assert "workflow_dispatch" in job["if"] and "run_ml" in job["if"]
    assert d["on"]["workflow_dispatch"]["inputs"]["run_ml"]["type"] == "boolean"
    steps = " ".join(str(s) for s in job["steps"])
    assert ".[dev,ml]" in steps and "download.pytorch.org/whl/cpu" in steps and "results-ml" in steps
    assert job["env"]["RAG_EMBEDDER"] == "st" and job["env"]["RAG_RERANKER"] == "ce"


def test_claude_workflow_shape():
    d = _load("claude.yml")
    steps = " ".join(str(s) for s in d["jobs"]["claude"]["steps"])
    assert "anthropics/claude-code-action" in steps and "ANTHROPIC_API_KEY" in steps


def test_claude_skills_are_well_formed():
    skills = Path(__file__).resolve().parent.parent / ".claude" / "skills"
    dirs = [d for d in skills.iterdir() if d.is_dir()]
    assert len(dirs) >= 5
    for d in dirs:
        text = (d / "SKILL.md").read_text()
        assert text.startswith("---\n")
        front = yaml.safe_load(text.split("---")[1])
        assert front["name"] == d.name and len(front["description"]) > 40
