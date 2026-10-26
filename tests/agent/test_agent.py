from __future__ import annotations

import json
from pathlib import Path

from graphmem.agent.agent import LocalizationAgent
from graphmem.graph.builder import GraphBuilder
from graphmem.ingestion.scanner import RepositoryScanner
from graphmem.llm.providers import MockProvider


def _agent(tmp_path: Path, response: str):
    (tmp_path / "sample.py").write_text(
        "def parse_value(value):\n"
        "    return int(value)\n\n"
        "def format_value(value):\n"
        "    return str(value)\n",
        encoding="utf-8",
    )

    scanner = RepositoryScanner(str(tmp_path))
    parser = scanner.get_parser()
    parsed = (
        parser.parse_repository(str(tmp_path))
        if hasattr(parser, "parse_repository")
        else None
    )

    if parsed is None:
        from graphmem.ingestion.cache import load_or_parse
        parsed = load_or_parse(scanner.repository_path, parser)

    from graphmem.retrieval.localizer import Localizer

    localizer = Localizer(GraphBuilder().build(parsed))

    issue_signals_response = json.dumps({
        "keywords": ["parse_value", "converting input"],
        "symbols": ["parse_value"],
        "error_terms": [],
        "search_queries": ["parse_value input conversion"],
    })

    return LocalizationAgent(
        localizer,
        str(tmp_path),
        MockProvider([issue_signals_response, response]),
    )


def test_agent_rejects_hallucinated_location_and_falls_back(tmp_path):
    agent = _agent(tmp_path, json.dumps({"locations": [
        {"location_id": "not-in-candidates::fake:1-2", "reason": "invented"}
    ]}))
    result = agent.localize("parse_value fails converting input")
    assert result["locations"]
    assert "hallucinated location rejected" in result["llm"]["fallback_reason"]
    assert result["tool_trace"][-1]["ok"] is False


def test_agent_accepts_only_known_candidates(tmp_path):
    # First run deterministically to discover the exact candidate ID.
    agent = _agent(tmp_path, '{"locations": []}')
    deterministic = agent.localizer.localize("parse_value fails converting input", top_k=10)
    hit = deterministic.scopes[0]
    entity = hit.entity
    key = f"{entity.path}::{entity.qualified_name or entity.name}:{entity.start_line or 1}-{entity.end_line or entity.start_line or 1}"
    issue_signals_response = json.dumps({
        "keywords": ["parse_value", "converting input"],
        "symbols": ["parse_value"],
        "error_terms": [],
        "search_queries": ["parse_value input conversion"],
    })

    rerank_response = json.dumps({
        "locations": [{
            "location_id": key,
            "reason": "Matches the parser failure.",
        }]
    })

    agent.provider = MockProvider([
        issue_signals_response,
        rerank_response,
    ])
    result = agent.localize("parse_value fails converting input")
    assert result["locations"][0]["reason"] == "Matches the parser failure."
    assert result["llm"]["fallback_reason"] is None


def test_agent_bad_json_falls_back_to_full_deterministic_ranking(tmp_path):
    agent = _agent(tmp_path, "not json")
    expected = agent.localizer.localize("parse_value fails converting input", top_k=8)
    result = agent.localize("parse_value fails converting input", top_k=8)
    assert len(result["locations"]) == len(expected.scopes)
    assert "JSONDecodeError" in result["llm"]["fallback_reason"]
    assert result["tool_trace"][-1]["ok"] is False


def test_agent_rejects_more_than_five_llm_locations(tmp_path):
    agent = _agent(tmp_path, json.dumps({"locations": [
        {"location_id": "x", "reason": "reason"} for _ in range(6)
    ]}))
    result = agent.localize("parse_value fails converting input")
    assert "at most five" in result["llm"]["fallback_reason"]
    assert result["locations"][0]["reason"] == "Deterministic retrieval ranking"


def test_agent_provider_failure_falls_back(tmp_path):
    class BrokenProvider:
        def generate(self, messages, tools=None):
            raise RuntimeError("provider unavailable")

    agent = _agent(tmp_path, "unused")
    agent.provider = BrokenProvider()
    result = agent.localize("parse_value fails converting input")
    assert "provider unavailable" in result["llm"]["fallback_reason"]
    assert result["locations"]
    assert result["locations"][0]["reason"] == "Deterministic retrieval ranking"
