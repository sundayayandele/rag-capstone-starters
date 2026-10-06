"""Tokenisation, light stemming and a tiny synonym lexicon.

The synonym lexicon is a stand-in for what a neural embedding model learns.
It keeps the offline demo deterministic and download-free. For real semantic
quality use the optional neural embedder (see embed.py, RAG_EMBEDDER=st).
"""
from __future__ import annotations
import re

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
PART_SPLIT = re.compile(r"[-_.]")

STOP = set(
    "a an the of to in on for and or is are was were be been do does did how what when where who which why can "
    "i we you it its my our your their this that with from at by as if not no than then there any all per about "
    "after before into me us should would could will shall must may might have has had get got".split()
)
# generic words that carry no topical signal; ignored when measuring how well a chunk covers a question
GENERIC = set("company policy employee employees many much long often soon far need needs tell give list year yearly spec specs details detail info information show find".split())

SYNONYM_GROUPS = [
    ("leave", "vacation", "holiday", "pto"),
    ("laptop", "notebook", "computer", "pc"),
    ("audio", "sound"),
    ("fuel", "gasoline", "petrol"),
    ("expense", "reimbursement"),
    ("manager", "supervisor"),
    ("car", "vehicle", "automobile"),
    ("cost", "price", "fee"),
    ("buy", "purchase", "procure"),
    ("fix", "repair", "mend"),
    ("broken", "faulty", "defective"),
    ("client", "customer"),
    ("password", "passcode"),
    ("sick", "ill", "illness"),
    ("mac", "macos", "osx"),
    ("remote", "wfh"),
    ("headlight", "headlamp"),
    ("battery", "accumulator"),
    ("slow", "sluggish"),
    ("licence", "license"),
    ("email", "mail"),
]
def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def stem(t: str) -> str:
    """Tiny suffix stripper, consistent rather than linguistically perfect (leave/leaves/leaving -> leav)."""
    if any(c.isdigit() for c in t) or any(c in t for c in "-_."):
        return t
    n = len(t)
    if n > 4 and t.endswith("ies"):
        t = t[:-3] + "y"
    elif n > 5 and t.endswith("ing"):
        t = t[:-3]
    elif n > 4 and t.endswith("ed"):
        t = t[:-2]
    elif n > 4 and t.endswith("es") and t[:-2].endswith(("s", "x", "z", "ch", "sh")):
        t = t[:-2]
    elif n > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    if len(t) > 4 and t.endswith("e"):
        t = t[:-1]
    return t


CANON = {stem(w): stem(g[0]) for g in SYNONYM_GROUPS for w in g}


def canon(t: str) -> str:
    s = stem(t)
    return CANON.get(s, s)


def terms(text: str, drop_generic: bool = False) -> list[str]:
    """Content terms: lowercase, stopwords removed, stemmed, synonyms folded. Hyphenated codes stay whole."""
    out = []
    for t in tokenize(text):
        if t in STOP or (drop_generic and t in GENERIC):
            continue
        out.append(canon(t))
    return out


def with_parts(ts: list[str]) -> list[str]:
    """Add the pieces of hyphenated codes so FP-2291-B also matches '2291'."""
    out = []
    for t in ts:
        out.append(t)
        if PART_SPLIT.search(t):
            out.extend(p for p in PART_SPLIT.split(t) if p)
    return out


def coverage(question: str, text: str) -> float:
    """Share of the question's distinct content terms that appear in the text (0..1)."""
    q = set(terms(question, drop_generic=True))
    if not q:
        return 0.0
    c = set(with_parts(terms(text)))
    return len(q & c) / len(q)


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]
