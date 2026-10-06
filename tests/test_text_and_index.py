from ragkit.chunk import chunk_markdown
from ragkit.index import BM25Index, DenseIndex, HybridIndex, rrf
from ragkit.text import coverage, terms, with_parts


def test_stemming_and_synonyms_fold():
    assert terms("Leaves") == terms("leave") == terms("vacation")
    assert terms("contributes") == terms("contribute")


def test_codes_stay_whole_and_gain_parts():
    ts = terms("Part FP-2291-B")
    assert "fp-2291-b" in ts
    assert {"fp", "2291", "b"} <= set(with_parts(ts))


def test_coverage_ignores_generic_words():
    assert coverage("What is the company policy on pets?", "Pets are allowed on Fridays.") == 1.0
    assert coverage("turbocharger", "Fuel pump for petrol engines.") == 0.0


def test_chunker_keeps_title_and_heading():
    chunks = chunk_markdown("doc", "# Title\n\n## Sec A\nOne sentence here.\n\n## Sec B\nAnother one.")
    assert [c.meta["heading"] for c in chunks] == ["Sec A", "Sec B"]
    assert chunks[0].text.startswith("Title. Sec A.")
    assert chunks[0].id == "doc#0" and chunks[1].id == "doc#1"


def _toy():
    return chunk_markdown("a", "# Alpha\n\n## X\nThe fuel pump FP-2291-B is rated 4.5 bar.") + \
           chunk_markdown("b", "# Beta\n\n## Y\nBrake pads are ceramic and quiet.")


def test_bm25_finds_exact_code():
    top = BM25Index(_toy()).search("FP-2291-B", 1)[0]
    assert top.chunk.doc == "a"


def test_dense_is_deterministic():
    a = DenseIndex(_toy()).search("ceramic brake", 1)[0]
    b = DenseIndex(_toy()).search("ceramic brake", 1)[0]
    assert a.chunk.doc == b.chunk.doc == "b" and abs(a.score - b.score) < 1e-6


def test_rrf_rewards_agreement():
    chunks = _toy()
    l1 = BM25Index(chunks).search("pump", 2)
    l2 = DenseIndex(chunks).search("pump", 2)
    fused = rrf([l1, l2])
    assert fused[0].chunk.doc == "a"


def test_hybrid_returns_k():
    assert len(HybridIndex(_toy()).search("brake", 1)) == 1
