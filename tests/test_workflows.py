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


def test_claude_workflow_shape():
    d = _load("claude.yml")
    steps = " ".join(str(s) for s in d["jobs"]["claude"]["steps"])
    assert "anthropics/claude-code-action" in steps and "ANTHROPIC_API_KEY" in steps
