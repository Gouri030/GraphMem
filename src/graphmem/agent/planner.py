
"""Issue understanding and query planning for the GraphMem agent."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from graphmem.indexing.code_index import extract_paths, extract_symbols, tokenize
from graphmem.retrieval.issue_signals import IssueSignals
from graphmem.llm.base import LLMProvider, Message


@dataclass(frozen=True)
class QueryPlan:
    issue: str
    tokens: list[str]
    symbols: list[str]
    paths: list[str]
    error_terms: list[str]
    variables: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_query_plan(issue: str) -> QueryPlan:
    """Build the existing deterministic query plan."""
    signals = IssueSignals.parse(issue)
    return QueryPlan(
        issue=issue,
        tokens=list(dict.fromkeys(tokenize(signals.prose))),
        symbols=sorted(set(signals.symbols) | extract_symbols(issue)),
        paths=sorted(set(signals.paths) | extract_paths(issue)),
        error_terms=sorted(signals.error_terms),
        variables=sorted(
            {
                symbol
                for symbol in extract_symbols(issue)
                if symbol.islower() and "_" in symbol
            }
        ),
    )


_ISSUE_SIGNAL_SYSTEM = """
You extract search signals from bug reports for repository fault localization.

Return ONLY valid JSON with this schema:
{
  "keywords": [],
  "symbols": [],
  "error_terms": [],
  "search_queries": []
}

Rules:
- Extract useful repository search terms from the bug report.
- Preserve exact identifiers, exception names, and error messages.
- Do not diagnose the bug or explain the report.
- Do not invent filenames, symbols, or stack-trace frames.
- Use short alternative search queries.
- Limit keywords to 15, symbols to 10, error_terms to 10,
  and search_queries to 5.
"""


def extract_issue_signals_with_llm(
    issue: str,
    provider: LLMProvider,
) -> dict[str, list[str]]:
    """Extract structured repository-search signals using the LLM."""
    response = provider.generate(
        [
            Message("system", _ISSUE_SIGNAL_SYSTEM),
            Message("user", json.dumps({"bug_report": issue})),
        ]
    )

    raw = response.text.strip()

    # Accept JSON enclosed in Markdown code fences.
    if raw.startswith("```"):
        raw = re.sub(r"\A```(?:json)?\s*", "", raw, count=1)
        raw = re.sub(r"\s*```\s*\Z", "", raw)

    result = json.loads(raw)

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object from the issue-signal LLM")

    limits = {
        "keywords": 15,
        "symbols": 10,
        "error_terms": 10,
        "search_queries": 5,
    }

    signals: dict[str, list[str]] = {}

    for field, limit in limits.items():
        values = result.get(field, [])

        if not isinstance(values, list):
            raise ValueError(f"LLM field {field!r} must be a list")

        signals[field] = list(
            dict.fromkeys(
                value.strip()
                for value in values
                if isinstance(value, str) and value.strip()
            )
        )[:limit]

    return signals
