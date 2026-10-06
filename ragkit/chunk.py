"""Corpus loading and section-aware chunking."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    id: str            # unique chunk id, e.g. leave-policy#2
    doc: str           # document id used for citations and for the golden set
    text: str
    title: str = ""
    meta: dict = field(default_factory=dict)


def chunk_markdown(doc_id: str, text: str, max_words: int = 140) -> list[Chunk]:
    """Split on level-2 headings; further split long sections on blank lines. Each chunk keeps the doc title and heading."""
    lines = text.strip().splitlines()
    title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else doc_id
    sections: list[tuple[str, list[str]]] = []
    head, buf = "", []
    for ln in lines[1:] if lines and lines[0].startswith("#") and not lines[0].startswith("##") else lines:
        if ln.startswith("## "):
            if buf:
                sections.append((head, buf))
            head, buf = ln[3:].strip(), []
        else:
            buf.append(ln)
    if buf or head:
        sections.append((head, buf))
    chunks: list[Chunk] = []
    for head, body in sections:
        paras = [p.strip() for p in re.split(r"\n\s*\n", "\n".join(body)) if p.strip()]
        cur: list[str] = []
        for p in paras:
            if cur and sum(len(x.split()) for x in cur) + len(p.split()) > max_words:
                chunks.append(_mk(doc_id, len(chunks), title, head, cur))
                cur = []
            cur.append(p)
        if cur:
            chunks.append(_mk(doc_id, len(chunks), title, head, cur))
    return chunks


def _mk(doc_id, n, title, head, paras):
    body = " ".join(paras)
    text = f"{title}. {head}. {body}" if head else f"{title}. {body}"
    return Chunk(id=f"{doc_id}#{n}", doc=doc_id, text=text, title=title, meta={"heading": head, "body": body})


def load_markdown_dir(path: Path, prefix: str = "") -> list[Chunk]:
    chunks: list[Chunk] = []
    for f in sorted(Path(path).glob("*.md")):
        chunks.extend(chunk_markdown(prefix + f.stem, f.read_text(encoding="utf-8")))
    return chunks


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
