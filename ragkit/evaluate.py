"""Evaluation: metrics, gate and result writing. See docs/EVALUATION.md."""
from __future__ import annotations
import json
import time
from collections import Counter
from pathlib import Path

from .answer import Answer
from .text import terms


def p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def faithfulness_lexical(answer: str, context: str) -> float:
    """Offline proxy: share of the answer's content terms that appear in the grounding context.
    Replace with an LLM judge for real evaluation (see EVALUATION.md)."""
    a = [t for t in terms(answer) if not t.startswith("[")]
    if not a:
        return 1.0
    c = set(terms(context))
    return sum(1 for t in a if t in c) / len(a)


def mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def evaluate(pipe, golden: list[dict], k: int = 5) -> dict:
    rec, rr, p1, cit, corr, ref, faith, lat = [], [], [], [], [], [], [], []
    grades: Counter = Counter()
    failures = []
    for g in golden:
        t0 = time.perf_counter()
        ans: Answer = pipe.ask(g["question"])
        lat.append(time.perf_counter() - t0)
        gold = set(g.get("source_ids", []))
        answerable = bool(gold)
        ok_ref = ans.refused == (not answerable)
        ref.append(1.0 if ok_ref else 0.0)
        if "grade" in ans.meta:
            grades[ans.meta["grade"]] += 1
        if answerable:
            rec.append(1.0 if gold & set(ans.retrieved[:k]) else 0.0)
            rank = next((i + 1 for i, d in enumerate(ans.retrieved) if d in gold), None)
            rr.append(1.0 / rank if rank else 0.0)
            p1.append(1.0 if ans.retrieved and ans.retrieved[0] in gold else 0.0)
            if not ans.refused:
                cit.append(1.0 if gold & set(ans.sources) else 0.0)
                faith.append(faithfulness_lexical(ans.text, ans.context))
            kws = g.get("answer_keywords", [])
            good = (not ans.refused) and all(kw.lower() in ans.text.lower() for kw in kws)
            corr.append(1.0 if good else 0.0)
            if not good:
                failures.append({"q": g["question"], "gold": sorted(gold), "got": ans.sources, "refused": ans.refused})
        elif not ok_ref:
            failures.append({"q": g["question"], "gold": [], "got": ans.sources, "refused": ans.refused})
    m = {
        "n_questions": len(golden),
        "recall@5": mean(rec), "precision@1": mean(p1), "mrr": mean(rr),
        "answer_correctness": mean(corr), "citation_accuracy": mean(cit),
        "faithfulness_lexical": mean(faith), "refusal_accuracy": mean(ref),
        "p95_latency_s": round(p95(lat), 4),
    }
    if grades:
        m["retrieval_grades"] = dict(grades)
    m["failures"] = failures[:8]
    return m


def gate(metrics: dict, thresholds: dict) -> list[str]:
    """Return a list of failed checks. thresholds: metric -> minimum, or metric -> ('max', value)."""
    bad = []
    for k, t in thresholds.items():
        v = metrics.get(k)
        if isinstance(t, (list, tuple)) and t[0] == "max":
            if v is None or v > t[1]:
                bad.append(f"{k} = {v} > {t[1]}")
        elif v is None or v < t:
            bad.append(f"{k} = {v} < {t}")
    return bad


def evaluate_configs(project: str, title: str, pipelines: dict, golden: list[dict], main: str, thresholds: dict, notes: str = "") -> dict:
    res = {"project": project, "title": title, "main": main, "thresholds": thresholds, "notes": notes, "configs": {}}
    for name, pipe in pipelines.items():
        res["configs"][name] = evaluate(pipe, golden)
    res["failed"] = gate(res["configs"][main], thresholds)
    res["passed"] = not res["failed"]
    return res


def write_results(results: list[dict], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for r in results:
        (out / f"{r['project']}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    md = results_markdown(results)
    (out / "RESULTS.md").write_text(md, encoding="utf-8")
    return out / "RESULTS.md"


COLS = [("recall@5", "Recall@5"), ("precision@1", "P@1"), ("mrr", "MRR"), ("answer_correctness", "Correct"),
        ("citation_accuracy", "Citation"), ("faithfulness_lexical", "Faithful*"), ("refusal_accuracy", "Refusal"), ("p95_latency_s", "p95 s")]


def results_markdown(results: list[dict]) -> str:
    out = ["# Evaluation results", "",
           "Offline defaults: hashing embedder, lexical reranker, extractive answers. `Faithful*` is a lexical proxy, not an LLM judge.", ""]
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        out += [f"## {r['title']}  ({status})", ""]
        if r.get("notes"):
            out += [r["notes"], ""]
        cols = r.get("columns") or COLS
        out.append("| Config | " + " | ".join(c[1] for c in cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for name, m in r["configs"].items():
            tag = f"**{name}**" if name == r["main"] else name
            out.append(f"| {tag} | " + " | ".join(str(m.get(c[0], "-")) for c in cols) + " |")
        out.append("")
        out.append("Gate on **" + r["main"] + "**: " + ", ".join(f"{k} {'<=' if isinstance(v, (list, tuple)) else '>='} {v[1] if isinstance(v, (list, tuple)) else v}" for k, v in r["thresholds"].items()))
        if r["failed"]:
            out.append("")
            out.append("Failed: " + "; ".join(r["failed"]))
        main_m = r["configs"][r["main"]]
        if main_m.get("failures"):
            out += ["", "<details><summary>Failure analysis (main config)</summary>", ""]
            for f in main_m["failures"]:
                out.append(f"- {f['q']} (gold {f['gold']}, got {f['got']}, refused={f['refused']})")
            out += ["", "</details>"]
        out.append("")
    return "\n".join(out)
