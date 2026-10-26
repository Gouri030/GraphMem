"""Repository localization agent: deterministic retrieval, optional safe LLM reranking."""
from __future__ import annotations

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

from graphmem.agent.context_builder import build_context
from graphmem.agent.planner import build_query_plan
from graphmem.agent.tools import ToolRegistry
from graphmem.llm.base import LLMProvider, Message


_SYSTEM = """You rerank repository fault-localization candidates. Return ONLY JSON of shape
{"locations":[{"location_id":"candidate ID","reason":"one concise sentence"}]}
Use only candidate IDs supplied in the input. Return at most five unique IDs. Never invent paths, IDs, lines, or symbols."""


def _location_id(candidate: dict[str, Any]) -> str:
    loc = candidate["location"]
    return f"{loc['file']}::{loc['scope']}:{loc['start_line']}-{loc['end_line']}"


class LocalizationAgent:
    def __init__(self, localizer: Any, repo_root: str, provider: LLMProvider | None = None,
                 token_budget: int = 3000, timeout_seconds: float = 1000.0) -> None:
        self.localizer = localizer
        self.repo_root = repo_root
        self.provider = provider
        self.token_budget = token_budget
        self.timeout_seconds = timeout_seconds
        self.tools = ToolRegistry(localizer, repo_root)

    def localize(self, issue: str, top_k: int = 10, config: Any = None,
                 use_llm: bool = True) -> dict[str, Any]:
        started = time.perf_counter()
        trace: list[dict[str, Any]] = []

        plan = build_query_plan(issue)
        search_issue = issue
        llm_issue_signals = None

        if use_llm and self.provider is not None:
            try:
                from graphmem.agent.planner import (
                    extract_issue_signals_with_llm,
                )

                llm_issue_signals = extract_issue_signals_with_llm(
                    issue, self.provider
                )

                search_terms = (
                    llm_issue_signals["keywords"]
                    + llm_issue_signals["symbols"]
                    + llm_issue_signals["error_terms"]
                    + llm_issue_signals["search_queries"]
                )

                if search_terms:
                    search_issue = (
                        issue
                        + "\n\nRepository search signals:\n"
                        + "\n".join(dict.fromkeys(search_terms))
                    )

                if os.getenv("GRAPHMEM_AGENT_DEBUG") == "1":
                    print(
                        "[LLM] Issue signals:",
                        json.dumps(llm_issue_signals, indent=2),
                        flush=True,
                    )

                trace.append({
                    "tool": "llm_issue_understanding",
                    "ok": True,
                    "signals": llm_issue_signals,
                })

            except Exception as exc:
                if os.getenv("GRAPHMEM_AGENT_DEBUG") == "1":
                    print(
                        f"[LLM] Issue understanding fallback: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )

                trace.append({
                    "tool": "llm_issue_understanding",
                    "ok": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                })

        # Deterministic extraction remains available as fallback.
        deterministic = self.localizer.localize(
            search_issue,
            top_k=top_k,
            config=config,
        )

        def call(name: str, **kwargs: Any) -> dict[str, Any]:
            t0 = time.perf_counter()
            result = self.tools.call(name, **kwargs)
            trace.append({"tool": name, "arguments": kwargs, "ok": result.get("ok", False),
                          "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                          "result_count": len(result.get("results", result.get("nodes", [])))})
            return result

        # Deterministic localization is always the source of truth and fallback.
        deterministic = self.localizer.localize(issue, top_k=top_k, config=config)
        ranked: list[dict[str, Any]] = []
        for hit in deterministic.scopes:
            entity = hit.entity
            path = entity.path or "<unknown>"
            scope = entity.qualified_name or entity.name or "<unknown>"
            start = entity.start_line or 1
            end = entity.end_line or start
            evidence = []
            for ev in (getattr(hit, "evidence", []) or [])[:8]:
                evidence.append({"type": getattr(ev, "type", "evidence"),
                                 "source": getattr(ev, "source", ""),
                                 "weight": getattr(ev, "weight", None)})
            ranked.append({"location": {"file": path, "scope": scope,
                                         "start_line": int(start), "end_line": int(end)},
                           "score": float(hit.score), "reason": "Deterministic retrieval ranking",
                           "evidence": evidence})
        candidate_by_id = {_location_id(c): c for c in ranked}
        fallback_reason = "LLM disabled" if not use_llm else None
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        llm_latency_ms = 0.0
        final = ranked[:max(1, top_k)]

        # Execute bounded graph/dataflow retrieval for the top three deterministic scopes.
        for candidate in ranked[:3]:
            path = candidate["location"]["file"]
            scope = candidate["location"]["scope"]
            search = call("search_entities", query=scope, limit=5)
            for entity in search.get("results", [])[:2]:
                call("traverse_graph", entity_id=entity["id"], direction="both", hops=2)
            # Slice only when a variable was explicitly identified in the issue.
            if plan.variables:
                call("get_dataflow_slice", path=path, line=candidate["location"]["start_line"],
                     variable=plan.variables[0], direction="both")
        # Record the context-bundle tool in the trace, while using the concrete
        # C07 builder as a reliable fallback when no bundle provider is registered.
        bundle_result = call("build_context_bundle", issue=issue,
                             seed_entities=[c["location"]["scope"] for c in ranked[:3]],
                             token_budget=self.token_budget)
        context = build_context(deterministic, issue, self.repo_root, token_budget=self.token_budget)
        if bundle_result.get("ok") and bundle_result.get("bundle") is not None:
            context = bundle_result["bundle"]
        # Historical/artifact evidence is optional; failures are captured in the trace.
        if ranked:
            call("search_history", query=issue, limit=5)
            call("search_artifacts", query=issue, limit=5)
        if use_llm and self.provider is not None and ranked:
            try:
                payload = {"issue": issue, "query_plan": plan.to_dict(),
                           "candidates": [{"location_id": key, **value} for key, value in candidate_by_id.items()],
                           "evidence_context": context.get("text", context)}
                t0 = time.perf_counter()
                messages = [
                    Message("system", _SYSTEM),
                    Message("user", json.dumps(payload, ensure_ascii=False, default=str)),
                ]
                # Bound waiting even for providers that do not implement their own timeout.
                pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="graphmem-llm")
                if os.getenv("GRAPHMEM_AGENT_DEBUG") == "1":
                    print(
                        f"[LLM] Calling provider={type(self.provider).__name__}; "
                        f"candidates={len(candidate_by_id)}; "
                        f"timeout={self.timeout_seconds}s",
                        flush=True,
                    )
                future = pool.submit(self.provider.generate, messages)
                try:
                    response = future.result(timeout=self.timeout_seconds)
                except FutureTimeoutError as exc:
                    future.cancel()
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise TimeoutError(f"LLM exceeded {self.timeout_seconds:.1f}s timeout") from exc
                else:
                    pool.shutdown(wait=True)
                llm_latency_ms = float(response.latency_ms or ((time.perf_counter() - t0) * 1000))
                usage = dict(response.usage or usage)
                raw_text = response.text.strip()

                # Qwen may wrap JSON in Markdown fences.
                if raw_text.startswith("```"):
                    lines = raw_text.splitlines()
                    if lines and lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    raw_text = "\n".join(lines).strip()

                if os.getenv("GRAPHMEM_AGENT_DEBUG") == "1":
                    print(f"[LLM] Raw response:\n{response.text}", flush=True)

                decoded = json.loads(raw_text)

                items = decoded.get("locations") if isinstance(decoded, dict) else None
                if not isinstance(items, list):
                    raise ValueError("bad JSON schema: expected locations list")
                if len(items) > 5:
                    raise ValueError("bad JSON schema: expected at most five locations")
                reranked, seen = [], set()
                for item in items:
                    if not isinstance(item, dict) or not isinstance(item.get("location_id"), str):
                        raise ValueError("bad location entry")
                    key = item["location_id"]
                    if key not in candidate_by_id:
                        raise ValueError(f"hallucinated location rejected: {key}")
                    if key in seen:
                        continue
                    seen.add(key)
                    chosen = dict(candidate_by_id[key])
                    reason = item.get("reason", "")
                    if not isinstance(reason, str) or not reason.strip():
                        raise ValueError(f"missing reason for candidate: {key}")
                    chosen["reason"] = reason.strip()[:500]
                    reranked.append(chosen)
                if not reranked:
                    raise ValueError("LLM returned no valid candidate locations")
                # Preserve the deterministic tail without duplicates.
                final = (reranked + [c for c in ranked if _location_id(c) not in seen])[:max(1, top_k)]
                if os.getenv("GRAPHMEM_AGENT_DEBUG") == "1":
                    print("[LLM] Reranking succeeded", flush=True)
                    print("[LLM] Before:", [_location_id(c) for c in ranked[:5]], flush=True)
                    print("[LLM] After: ", [_location_id(c) for c in final[:5]], flush=True)
                trace.append({"tool": "llm_rerank", "ok": True, "latency_ms": round(llm_latency_ms, 3),
                              "prompt_tokens": usage.get("prompt_tokens", 0),
                              "completion_tokens": usage.get("completion_tokens", 0)})
            except Exception as exc:
                fallback_reason = f"{type(exc).__name__}: {exc}"
                if os.getenv("GRAPHMEM_AGENT_DEBUG") == "1":
                    print(f"[LLM] FALLBACK: {fallback_reason}", flush=True)
                final = ranked[:max(1, top_k)]
                trace.append({"tool": "llm_rerank", "ok": False, "reason": fallback_reason,
                              "latency_ms": round(llm_latency_ms, 3)})
        elif use_llm and self.provider is None:
            fallback_reason = "No LLM provider configured"

        return {"locations": final[:max(1, top_k)], "query_plan": plan.to_dict(), "tool_trace": trace,
                "context": context, "llm": {"used": bool(self.provider and use_llm and fallback_reason is None),
                "fallback_reason": fallback_reason, "token_usage": usage,
                "latency_ms": round(llm_latency_ms, 3)},
                "latency_ms": round((time.perf_counter() - started) * 1000, 3)}


def localize(localizer: Any, repo_root: str, issue: str, provider: LLMProvider | None = None,
             **kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper matching the task's ``agent.localize(issue)`` API."""
    return LocalizationAgent(localizer, repo_root, provider).localize(issue, **kwargs)
