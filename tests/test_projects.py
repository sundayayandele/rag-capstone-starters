import pytest

from ragkit.cli import PROJECTS, load


@pytest.mark.parametrize("key", list(PROJECTS))
def test_project_meets_its_gate(key):
    r = load(key).run()
    assert r["passed"], r["failed"]


@pytest.mark.parametrize("key", ["p01", "p02", "p03", "p09"])
def test_results_are_deterministic(key):
    a, b = load(key).run(), load(key).run()
    for cfg in a["configs"]:
        for metric, v in a["configs"][cfg].items():
            if metric in ("p95_latency_s", "failures"):
                continue
            assert v == b["configs"][cfg][metric], (cfg, metric)


def test_hybrid_is_not_worse_than_either_leg():
    r = load("p03").run()["configs"]
    best_leg = max(r["bm25"]["answer_correctness"], r["dense"]["answer_correctness"])
    assert r["hybrid"]["answer_correctness"] >= best_leg - 0.05


def test_crag_improves_on_naive():
    r = load("p09").run()["configs"]
    assert r["crag + trusted fallback"]["answer_correctness"] > r["naive (no correction)"]["answer_correctness"]
