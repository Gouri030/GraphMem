from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from graphmem.dataflow.analyzer import DataFlowSlicer, SliceDirection
from graphmem.graph.store import GraphStore
from graphmem.indexing.code_index import (
    CodeIndex, statement_header, tokenize,
)
from graphmem.models.entities import CodeEntity, EntityType
from graphmem.ranking.config import RankConfig
from graphmem.retrieval.evidence import Evidence, EvidenceProvider
from graphmem.retrieval.issue_signals import IssueSignals
from graphmem.retrieval.scope_graph import ScopeGraph

BUILTIN_SIGNALS = {"lexical", "graph", "dataflow"}
CONTEXT_SIGNALS = {"graph", "history", "artifact", "semantic"}
TEST_TERMS = {"test", "tests", "testing", "pytest", "unittest", "fixture", "assertion"}


def _is_test_scope(entity: CodeEntity) -> bool:
    path_parts = (entity.path or "").replace("\\", "/").lower().split("/")
    filename = path_parts[-1] if path_parts else ""
    name = (entity.name or "").lower()
    return (
        any(part in {"test", "tests", "testing"} for part in path_parts[:-1])
        or filename.startswith("test_")
        or filename.endswith("_test.py")
        or name.startswith(("test_", "test"))
    )


@dataclass
class StatementHit:
    entity: CodeEntity
    score: float
    evidence: list[Evidence]
    start_line: int
    end_line: int


@dataclass
class ScopeHit:
    entity: CodeEntity
    score: float
    signals: dict[str, float]
    evidence: list[Evidence]
    statements: list[StatementHit] = field(default_factory=list)


@dataclass
class LocalizationResult:
    issue: str
    config: str
    scopes: list[ScopeHit]
    timings_ms: dict[str, float]

    @property
    def files(self) -> list[tuple[str, float]]:
        best: dict[str, float] = {}
        for hit in self.scopes:
            if hit.entity.path:
                best[hit.entity.path] = max(best.get(hit.entity.path, 0.0), hit.score)
        return sorted(best.items(), key=lambda kv: -kv[1])

    def to_dict(self, max_candidates: int = 20) -> dict:
        candidates = []
        for hit in self.scopes:
            name = hit.entity.qualified_name or hit.entity.name
            context = [e for e in hit.evidence if e.type in CONTEXT_SIGNALS]
            if hit.statements:
                for st in hit.statements:
                    candidates.append({
                        "level": "statement",
                        "file": hit.entity.path,
                        "scope": name,
                        "start_line": st.start_line,
                        "end_line": st.end_line,
                        "score": round(st.score, 4),
                        "evidence": [e.to_dict() for e in st.evidence + context],
                    })
            else:
                candidates.append({
                    "level": "scope",
                    "file": hit.entity.path,
                    "scope": name,
                    "start_line": hit.entity.start_line,
                    "end_line": hit.entity.end_line,
                    "score": round(hit.score, 4),
                    "evidence": [e.to_dict() for e in hit.evidence],
                })
        candidates.sort(key=lambda c: -c["score"])
        return {
            "issue": self.issue,
            "config": self.config,
            "files": [{"file": f, "score": round(s, 4)} for f, s in self.files],
            "scopes": [
                {
                    "file": h.entity.path,
                    "scope": h.entity.qualified_name or h.entity.name,
                    "type": h.entity.type.value,
                    "start_line": h.entity.start_line,
                    "end_line": h.entity.end_line,
                    "score": round(h.score, 4),
                    "signals": {k: round(v, 4) for k, v in h.signals.items()},
                }
                for h in self.scopes
            ],
            "candidates": candidates[:max_candidates],
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
        }


class Localizer:
    """Build once per repository, then call localize() with any RankConfig."""

    def __init__(
        self,
        graph: GraphStore,
        config: RankConfig | None = None,
        providers: list[EvidenceProvider] | None = None,
    ) -> None:
        self.graph = graph
        self.config = config or RankConfig()
        self.providers = list(providers or [])
        self.index = CodeIndex(graph)
        self.scope_graph = ScopeGraph(graph)
        self.slicer = DataFlowSlicer(graph)

    # ------------------------------------------------------------------
    def localize(
        self, issue: str, top_k: int = 10, config: RankConfig | None = None
    ) -> LocalizationResult:
        cfg = config or self.config
        w = cfg.weights
        timings: dict[str, float] = {}
        t_start = time.perf_counter()

        signals = IssueSignals.parse(issue)
        tokens = signals.query_tokens
        symbols = signals.symbols
        paths = signals.paths
        # Error terms are searched separately, not allowed to swamp the prose query.
        error_tokens = sorted(signals.error_terms)
        structured_tokens = list(dict.fromkeys(tokens + error_tokens + [t for sym in symbols for t in tokenize(sym)]))
        test_related = bool(TEST_TERMS & set(tokens)) or any(
            any(part in {"test", "tests", "testing"} for part in p.lower().replace("\\", "/").split("/"))
            for p in paths
        ) or any(sym.lower().startswith("test") for sym in symbols)

        def test_penalty(entity: CodeEntity) -> float:
            return (
                1.0 if test_related or not _is_test_scope(entity)
                else cfg.test_scope_penalty
            )

        entities: dict[str, CodeEntity] = {}
        evidence: dict[str, list[Evidence]] = defaultdict(list)

        # ---------------- Stage 1: lexical ----------------
        t = time.perf_counter()
        lex: dict[str, float] = {}
        if w.get("lexical", 0.0) > 0:
            for e, s in self.index.search_scopes(tokens, cfg.lexical_pool):
                entities[e.id] = e
                lex[e.id] = s
                evidence[e.id].append(
                    Evidence("lexical", "BM25 over names, paths and statement text", s)
                )
            # Exact identifiers (including one-word names) are high-value evidence,
            # but only boost explicitly mentioned names verified by CodeIndex.
            for sym in symbols:
                matches = self.index.resolve_symbol(sym)
                for e in matches:
                    entities[e.id] = e
                    bonus = cfg.explicit_symbol_bonus
                    lex[e.id] = min(1.0, lex.get(e.id, 0.0) + bonus)
                    evidence[e.id].append(
                        Evidence("lexical", f"verified explicit symbol: {sym}", 1.0, {"symbol": sym})
                    )
            # Traceback frames provide direct file/function/line evidence.
            for frame in signals.frames:
                for e in self.index.scopes_for_path(frame.path):
                    entities[e.id] = e
                    bonus = 0.85 if e.type == EntityType.FILE else 0.25
                    lex[e.id] = min(1.0, lex.get(e.id, 0.0) + bonus)
                    evidence[e.id].append(Evidence(
                        "lexical", f"traceback frame: {frame.path}:{frame.line}"
                        + (f" in {frame.function}" if frame.function else ""),
                        1.0, {"path": frame.path, "line": frame.line, "function": frame.function},
                    ))
                    if frame.function:
                        for candidate in self.index.by_name.get(frame.function, []):
                            if candidate.path == e.path:
                                entities[candidate.id] = candidate
                                lex[candidate.id] = min(1.0, lex.get(candidate.id, 0.0) + 0.9)
                                evidence[candidate.id].append(Evidence(
                                    "lexical", f"traceback function at line {frame.line}",
                                    1.0, {"line": frame.line, "function": frame.function},
                                ))
            for typ, message in signals.exceptions:
                evidence_label = f"exception: {typ}" + (f": {message}" if message else "")
                # Exception names/messages are included in lexical retrieval, but
                # only repository candidates matching their tokens are boosted.
                for e, score in self.index.search_scopes(tokenize(typ + " " + message), cfg.lexical_pool):
                    entities[e.id] = e
                    lex[e.id] = min(1.0, lex.get(e.id, 0.0) + 0.2 * score)
                    evidence[e.id].append(Evidence("lexical", evidence_label, score))
            for p in paths:
                for e in self.index.scopes_for_path(p):
                    bonus = 0.5 if e.type == EntityType.FILE else 0.25
                    entities[e.id] = e
                    lex[e.id] = min(1.0, lex.get(e.id, 0.0) + bonus)
                    evidence[e.id].append(
                        Evidence("lexical", f"issue mentions path: {p}", 1.0, {"path": p})
                    )
        timings["lexical"] = (time.perf_counter() - t) * 1000

        # ---------------- Stage 2: graph expansion ----------------
        t = time.perf_counter()
        if w.get("graph", 0.0) > 0 and lex:
            seeds = [
                i for i, _ in sorted(
                    lex.items(),
                    key=lambda kv: (-(kv[1] * test_penalty(entities[kv[0]])), kv[0]),
                )[: cfg.seed_limit]
            ]
            best: dict[str, tuple[float, list[str], list[str]]] = {}
            for sid in seeds:
                seed_score = lex[sid] * test_penalty(entities[sid])
                for nid, val in self._expand(sid, seed_score, cfg).items():
                    if val[0] > best.get(nid, (0.0,))[0]:
                        best[nid] = val
            for nid, (score, path, rels) in best.items():
                ent = self.graph.get_entity(nid)
                if ent is None:
                    continue
                entities[nid] = ent
                seed_name = self._name(path[0])
                evidence[nid].append(Evidence(
                    "graph",
                    f"{' > '.join(rels)} from {seed_name}",
                    score,
                    {"seed": path[0], "relations": rels, "hops": len(rels)},
                ))
        timings["graph"] = (time.perf_counter() - t) * 1000

        # ---------------- Pool selection (pre-dataflow score) ----------------
        active_all = BUILTIN_SIGNALS | {p.signal for p in self.providers}
        prelim_active = active_all - {"dataflow"}
        prelim = {
            i: self._combine(self._strengths(evidence[i], lex.get(i, 0.0)), w, prelim_active)
            * test_penalty(entities[i])
            for i in entities
        }
        pool = [i for i, s in sorted(prelim.items(), key=lambda kv: -kv[1]) if s > 0][: cfg.pool_size]

        # ---------------- Providers (history / artifacts / ...) ----------------
        t = time.perf_counter()
        for provider in self.providers:
            if w.get(provider.signal, 0.0) <= 0 or not pool:
                continue
            pool_set = set(pool)
            for eid, evs in provider.score([entities[i] for i in pool], issue).items():
                if eid in pool_set:
                    evidence[eid].extend(evs)
        timings["providers"] = (time.perf_counter() - t) * 1000

        # ---------------- Stage 3: statements + data flow ----------------
        t = time.perf_counter()
        q_set = set(structured_tokens)
        idf_q = sorted((self.index.idf[x] for x in q_set if x in self.index.idf), reverse=True)
        qmax = sum(idf_q[:5])

        hits: list[ScopeHit] = []
        for sid in pool:
            scope = entities[sid]
            frame_lines = {f.line for f in signals.frames if f.path.lstrip("./") == (scope.path or "").lstrip("./")}
            stmts, df_strength = self._statements(scope, q_set, qmax, cfg, prelim[sid], frame_lines)
            if df_strength > 0:
                evidence[sid].append(Evidence(
                    "dataflow", "data flow links statements that match the issue", df_strength
                ))
            strengths = self._strengths(evidence[sid], lex.get(sid, 0.0))
            strengths["dataflow"] = df_strength
            score = self._combine(strengths, w, active_all) * test_penalty(scope)
            hits.append(ScopeHit(scope, score, strengths, evidence[sid], stmts))
        timings["statements"] = (time.perf_counter() - t) * 1000

        hits.sort(key=lambda h: -h.score)
        timings["total"] = (time.perf_counter() - t_start) * 1000
        return LocalizationResult(issue, cfg.name, hits[:top_k], timings)

    # ------------------------------------------------------------------
    def _name(self, entity_id: str) -> str:
        e = self.graph.get_entity(entity_id)
        return (e.qualified_name or e.name) if e else entity_id

    @staticmethod
    def _strengths(evs: list[Evidence], lexical: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for ev in evs:
            out[ev.type] = max(out.get(ev.type, 0.0), ev.weight)
        out["lexical"] = lexical
        return out

    @staticmethod
    def _combine(strengths: dict[str, float], weights: dict[str, float], active: set[str]) -> float:
        num = den = 0.0
        for sig in active:
            wt = weights.get(sig, 0.0)
            if wt <= 0:
                continue
            den += wt
            num += wt * strengths.get(sig, 0.0)
        return num / den if den else 0.0

    def _expand(self, seed_id: str, seed_score: float, cfg: RankConfig):
        """Typed, directional BFS. Returns {scope_id: (score, path, relations)}."""
        best: dict[str, tuple[float, list[str], list[str]]] = {}
        frontier = {seed_id: (seed_score, [seed_id], [])}
        for _ in range(cfg.max_depth):
            nxt: dict[str, tuple[float, list[str], list[str]]] = {}
            for nid, (sc, path, rels) in frontier.items():
                for m, rel, direction in self.scope_graph.neighbors(
                    nid, cfg.edge_weights, cfg.max_neighbors
                ):
                    if m in path:
                        continue
                    s = sc * cfg.edge_weights[rel] * cfg.hop_decay
                    if s > nxt.get(m, (0.0,))[0]:
                        nxt[m] = (s, path + [m], rels + [f"{rel.value}:{direction}"])
            for m, v in nxt.items():
                if v[0] > best.get(m, (0.0,))[0]:
                    best[m] = v
            frontier = nxt
        return best

    def _statements(self, scope, q_set, qmax, cfg, scope_score, traceback_lines=None):
        """Rank statements inside one scope. Returns (hits, dataflow strength)."""
        w = cfg.weights
        stmts = self.slicer.statements_in_scope(scope.id)
        if not stmts:
            return [], 0.0

        traceback_lines = traceback_lines or set()
        spans: dict[str, tuple[int, int]] = {}
        lex_s: dict[str, float] = {}
        matched: dict[str, list[str]] = {}
        for s in stmts:
            text, start, end = statement_header(s)
            spans[s.id] = (start, end)
            hit = q_set & set(tokenize(text))
            if hit:
                raw = sum(self.index.idf.get(tok, 0.0) for tok in hit)
                lex_s[s.id] = min(1.0, raw / qmax)
                matched[s.id] = sorted(hit)
            if start in traceback_lines:
                lex_s[s.id] = max(lex_s.get(s.id, 0.0), 1.0)
                matched.setdefault(s.id, []).append("traceback line")

        df: dict[str, tuple[float, Evidence]] = {}
        if w.get("dataflow", 0.0) > 0 and lex_s:
            by_id = {s.id: s for s in stmts}
            seeds = sorted(lex_s, key=lambda k: -lex_s[k])[: cfg.dataflow_seeds]
            for sid in seeds:
                seed = by_id[sid]
                for node in self.slicer.slice(
                    seed, SliceDirection.BOTH, max_depth=cfg.slice_depth
                ):
                    if node.depth == 0:
                        continue
                    strength = lex_s[sid] / (1 + node.depth)
                    if strength > df.get(node.entity.id, (0.0,))[0]:
                        df[node.entity.id] = (strength, Evidence(
                            "dataflow",
                            f"{node.direction} slice from line {seed.start_line}"
                            f" via '{node.variable}'",
                            strength,
                            {"seed_line": seed.start_line, "depth": node.depth,
                             "variable": node.variable, "direction": node.direction},
                        ))

        active = {"lexical", "dataflow"}
        mix = cfg.statement_scope_mix
        out: list[StatementHit] = []
        for s in stmts:
            if s.id not in lex_s and s.id not in df:
                continue
            strengths = {"lexical": lex_s.get(s.id, 0.0), "dataflow": df.get(s.id, (0.0,))[0]}
            local = self._combine(strengths, w, active)
            score = (1 - mix) * local + mix * scope_score
            evs: list[Evidence] = []
            if s.id in lex_s:
                evs.append(Evidence(
                    "lexical", "statement matches: " + ", ".join(matched[s.id]), lex_s[s.id]
                ))
            if s.id in df:
                evs.append(df[s.id][1])
            start, end = spans[s.id]
            out.append(StatementHit(s, score, evs, start, end))

        out.sort(key=lambda h: -h.score)
        df_strength = max((v[0] for v in df.values()), default=0.0)
        return out[: cfg.statements_per_scope], df_strength
