import pytest

from ragkit.answer import REFUSAL, answer_from_hits
from ragkit.chunk import chunk_markdown
from ragkit.index import DenseIndex
from ragkit.llm import Anthropic, GitHubModels, LLMError, get_llm


def _idx():
    ch = chunk_markdown("hr", "# HR\n\n## Leave\nEmployees get 25 days of annual leave.\n\n## Pets\nPets are not allowed in the office.")
    return DenseIndex(ch)


def test_answers_with_citation():
    a = answer_from_hits("How many days of annual leave?", _idx().search("How many days of annual leave?", 3))
    assert not a.refused and "25 days" in a.text and a.sources == ["hr"]


def test_refuses_when_unsupported():
    a = answer_from_hits("What is the sabbatical policy?", _idx().search("sabbatical policy", 3))
    assert a.refused and a.text == REFUSAL and a.sources == []


def test_no_llm_by_default(monkeypatch):
    monkeypatch.delenv("RAG_LLM", raising=False)
    assert get_llm() is None


def test_providers_need_credentials(monkeypatch):
    for v in ("GITHUB_TOKEN", "GH_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(LLMError):
        GitHubModels()
    with pytest.raises(LLMError):
        Anthropic()


def test_llm_path_parses_citations_and_refusal():
    class Fake:
        def __init__(self, out):
            self.out = out

        def complete(self, system, user, max_tokens=400):
            assert "ONLY the provided context" in system
            return self.out

    hits = _idx().search("How many days of annual leave?", 3)
    ok = answer_from_hits("How many days of annual leave?", hits, Fake("25 days [hr#0]"))
    assert ok.sources == ["hr"] and not ok.refused
    no = answer_from_hits("How many days of annual leave?", hits, Fake("I do not know based on the available documents."))
    assert no.refused and no.sources == []
