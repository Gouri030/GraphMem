"""Deterministic, token-budgeted evidence bundles for localization results."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _estimate(text: str) -> int:
    """Conservative, deterministic token estimate required by the context contract."""
    return (len(text) + 3) // 4


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _entity(hit: Any) -> Any:
    return _value(hit, "entity", {})


def _evidence_text(evidence: Any) -> str:
    kind = _value(evidence, "type", "evidence")
    source = _value(evidence, "source", "")
    weight = _value(evidence, "weight", None)
    detail = _value(evidence, "detail", {}) or {}
    explanation = detail.get("reason") or detail.get("description") or detail.get("matched") or source or "supporting retrieval signal"
    suffix = f" (weight {weight:.3f})" if isinstance(weight, (int, float)) else ""
    return f"Why: {kind}: {explanation}{suffix}"


def _candidate(hit: Any, repo_root: Path, issue: str) -> dict[str, Any]:
    entity = _entity(hit)
    path = _value(entity, "path") or _value(hit, "file") or "<unknown>"
    name = _value(entity, "qualified_name") or _value(entity, "name") or _value(hit, "scope") or "<unknown>"
    score = float(_value(hit, "score", 0.0) or 0.0)
    start = _value(hit, "start_line") or _value(entity, "start_line") or 1
    end = _value(hit, "end_line") or _value(entity, "end_line") or start
    evidence = list(_value(hit, "evidence", []) or [])
    statements = list(_value(hit, "statements", []) or [])
    if statements:
        # Statement hits provide a more precise source location than the enclosing scope.
        best_statement = max(statements, key=lambda st: float(_value(st, "score", 0.0) or 0.0))
        start = _value(best_statement, "start_line", start) or start
        end = _value(best_statement, "end_line", end) or end
        evidence = list(_value(best_statement, "evidence", []) or []) + evidence
    try:
        source_path = (repo_root / str(path)).resolve()
        source_path.relative_to(repo_root.resolve())
        lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
        lo, hi = max(1, int(start) - 3), min(len(lines), int(end) + 3)
        code = "\n".join(f"{n:>5}: {lines[n-1]}" for n in range(lo, hi + 1))
    except (OSError, ValueError, TypeError):
        code = "(source span unavailable)"
    bullets = [_evidence_text(ev) for ev in evidence[:8]]
    if not bullets:
        bullets = [f"Why: ranked candidate for the issue; score={score:.4f}."]
    # Graph and data-flow details are carried by evidence when available. Never fabricate edges.
    neighbours: list[str] = []
    dataflow: list[str] = []
    for ev in evidence:
        detail = _value(ev, "detail", {}) or {}
        for n in detail.get("neighbours", detail.get("neighbors", [])) or []:
            label = str(n.get("name", n) if isinstance(n, dict) else n)
            if label not in neighbours:
                neighbours.append(label)
        for item in detail.get("dataflow", detail.get("slice", [])) or []:
            if isinstance(item, dict):
                ent = item.get("entity", item)
                p = ent.get("path", "") if isinstance(ent, dict) else ""
                ln = ent.get("start_line", ent.get("line", "?")) if isinstance(ent, dict) else "?"
                nm = ent.get("qualified_name", ent.get("name", "statement")) if isinstance(ent, dict) else str(ent)
                dataflow.append(f"{p}:{ln} {nm}".strip())
            else:
                dataflow.append(str(item))
    return {"location": {"file": str(path), "scope": str(name), "start_line": int(start), "end_line": int(end)},
            "score": score, "evidence": bullets, "code_span": code,
            "graph_neighbours": neighbours[:3], "data_flow_slice": dataflow}


def _render(issue: str, items: list[dict[str, Any]]) -> str:
    chunks = [f"Issue: {issue}", "Retrieved localization evidence"]
    for i, item in enumerate(items, 1):
        loc = item["location"]
        chunks.append(f"\n{i}. {loc['file']}::{loc['scope']}:{loc['start_line']}-{loc['end_line']} (score: {item['score']:.4f})")
        chunks.extend(f"  - {line}" for line in item["evidence"])
        chunks.append("  Code span:\n" + "\n".join("    " + line for line in item["code_span"].splitlines()))
        if item["graph_neighbours"]:
            chunks.append("  Graph neighbours: " + ", ".join(item["graph_neighbours"]))
        if item["data_flow_slice"]:
            chunks.append("  Data-flow slice:\n" + "\n".join("    - " + line for line in item["data_flow_slice"]))
    return "\n".join(chunks)


def build_context(localization_result: Any, issue: str, repo_root: str | Path, token_budget: int = 3000) -> dict[str, Any]:
    """Build readable evidence context without LLM calls, respecting ``token_budget``.

    Candidates are considered in descending score order. To preserve the highest-ranked
    evidence, code spans are shortened before lower-ranked candidates are removed.
    ``tokens_estimate`` is always ``ceil(len(text) / 4)`` and never exceeds the budget.
    """
    budget = max(0, int(token_budget))
    root = Path(repo_root).resolve()
    hits = list(_value(localization_result, "scopes", []) or [])
    hits.sort(key=lambda hit: float(_value(hit, "score", 0.0) or 0.0), reverse=True)
    candidates = [_candidate(hit, root, issue) for hit in hits]
    # Start with all ranked candidates, trimming code first, then dropping the tail.
    while candidates:
        text = _render(issue, candidates)
        if _estimate(text) <= budget:
            return {"text": text, "items": candidates, "tokens_estimate": _estimate(text)}
        code_items = [item for item in candidates if item["code_span"] and item["code_span"] != "(source span unavailable)" and len(item["code_span"].splitlines()) > 1]
        if code_items:
            # Trim the largest code span by one line, retaining the location and evidence.
            target = max(code_items, key=lambda item: len(item["code_span"]))
            lines = target["code_span"].splitlines()
            if len(lines) > 2:
                target["code_span"] = "\n".join(lines[:max(1, len(lines)//2)] + ["    ... (code span truncated)"])
            else:
                target["code_span"] = "(code span truncated)"
            continue
        candidates.pop()  # lowest-ranked candidate is always last
    text = f"Issue: {issue}\nRetrieved localization evidence"
    # Extremely small budgets cannot fit even the header; return empty text rather than exceed.
    if _estimate(text) > budget:
        text = ""
    return {"text": text, "items": [], "tokens_estimate": _estimate(text)}


# Friendly alias for callers that prefer a noun phrase.
build_context_bundle = build_context
