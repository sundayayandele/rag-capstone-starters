"""Command line: ragkit list | eval [--only p01,p03] [--out results] | ask p01 "question" | site."""
from __future__ import annotations
import argparse
import importlib
import json
import sys
from pathlib import Path

from .evaluate import write_results

PROJECTS = {
    "p01": "projects.p01_handbook_qa.pipeline",
    "p02": "projects.p02_ticket_rerank.pipeline",
    "p03": "projects.p03_part_finder.pipeline",
    "p09": "projects.p09_compliance_crag.pipeline",
    "p12": "projects.p12_nl_analytics.pipeline",
}


def load(key: str):
    return importlib.import_module(PROJECTS[key])


def cmd_eval(args) -> int:
    keys = args.only.split(",") if args.only else list(PROJECTS)
    results = []
    for k in keys:
        r = load(k).run()
        results.append(r)
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['title']}" + ("" if r["passed"] else "  -> " + "; ".join(r["failed"])))
    out = write_results(results, Path(args.out))
    print(f"wrote {out}")
    if args.site:
        from .site import build_site
        print(f"wrote {build_site(Path(args.out), Path(args.site))}")
    return 0 if all(r["passed"] for r in results) else 1


def cmd_ask(args) -> int:
    pipe = load(args.project).make()
    out = pipe.ask(args.question)
    if isinstance(out, dict):  # text-to-SQL
        print("SQL :", out.get("sql"))
        print("ROWS:", out.get("rows") if not out["refused"] else f"refused ({out.get('error') or 'unsupported'})")
    else:
        print(out.text)
        print("sources:", out.sources, "| refused:", out.refused)
    return 0


def cmd_site(args) -> int:
    from .site import build_site
    print(build_site(Path(args.results), Path(args.out)))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ragkit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=lambda a: print("\n".join(f"{k}  {v}" for k, v in PROJECTS.items())) or 0)
    e = sub.add_parser("eval")
    e.add_argument("--only")
    e.add_argument("--out", default="results")
    e.add_argument("--site", help="also build the results site into this directory")
    e.set_defaults(fn=cmd_eval)
    a = sub.add_parser("ask")
    a.add_argument("project", choices=list(PROJECTS))
    a.add_argument("question")
    a.set_defaults(fn=cmd_ask)
    s = sub.add_parser("site")
    s.add_argument("--results", default="results")
    s.add_argument("--out", default="site")
    s.set_defaults(fn=cmd_site)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
