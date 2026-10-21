"""Identifier-aware BM25 over scopes (file/class/function/method)."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from graphmem.graph.store import GraphStore
from graphmem.models.entities import CodeEntity, EntityType

STOP = {
    "the", "a", "an", "to", "is", "are", "and", "or", "of", "in", "on", "when",
    "with", "for", "it", "this", "that", "be", "as", "at", "by", "not", "but",
    "if", "then", "else", "we", "i", "you", "can", "should", "does", "do",
    "has", "have", "was", "were", "from", "into", "so", "my", "its", "use",
    "used", "using", "which", "what", "there", "self", "cls", "return", "def",
    "class", "import", "none", "true", "false", "pass", "elif", "try", "except",
    "raise", "lambda", "yield", "while", "assert", "del", "global", "nonlocal",
    "async", "await", "py",
}

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_QUOTED = re.compile(r"(?:`([^`\n]+)`|['\"]([^'\"\n]+)['\"])")
_SYMBOL = re.compile(r"\b(?:[a-z]+_[a-z0-9_]+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+)\b")
_QUALIFIED = re.compile(r"\b(?:[A-Za-z_]\w*\.)+[A-Za-z_]\w*\b")
_EXPLICIT_NAME = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_PATH = re.compile(r"[\w./-]+\.py")

SCOPE_TYPES = {EntityType.FILE, EntityType.CLASS, EntityType.FUNCTION, EntityType.METHOD}
COMPOUND = {
    "if_statement", "for_statement", "while_statement",
    "with_statement", "try_statement", "match_statement",
}


def tokenize(text: str) -> list[str]:
    """Lowercased tokens: the full identifier plus its snake/camel parts."""
    tokens: list[str] = []
    for word in _WORD.findall(text):
        parts = [p.lower() for p in _PART.findall(word.replace("_", " "))]
        candidates = parts + ([word.lower()] if len(parts) != 1 or word.lower() != parts[0] else [])
        for tok in candidates:
            if len(tok) >= 2 and tok not in STOP:
                tokens.append(tok)
    return tokens


def extract_symbols(text: str) -> set[str]:
    """Extract explicitly written code symbols, never every prose identifier."""
    found: set[str] = set()
    without_paths = _PATH.sub(" ", text)
    for ticked, quoted in _QUOTED.findall(without_paths):
        value = (ticked or quoted).strip()
        # Inline code may include a call suffix; the symbol itself is the name.
        value = value[:-2] if value.endswith("()") else value
        if _EXPLICIT_NAME.fullmatch(value):
            found.add(value)
    found.update(_SYMBOL.findall(without_paths))
    found.update(_QUALIFIED.findall(without_paths))
    return {s for s in found if len(s) > 1 and s.lower() not in STOP}


def extract_paths(text: str) -> set[str]:
    return {p.lstrip("./") for p in _PATH.findall(text)}


def statement_header(stmt: CodeEntity) -> tuple[str, int, int]:
    """(text, start_line, end_line). Compound statements use only their header
    line, so an `if` does not 'contain' every token of its body."""
    src = stmt.metadata.get("source", "")
    start = stmt.start_line or 0
    end = stmt.end_line or start
    if stmt.metadata.get("ast_type") in COMPOUND:
        return src.split("\n", 1)[0], start, start
    return src, start, end


class BM25:
    """Small inverted-index BM25 (Lucene-style idf, always positive)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 1.0
        self.avgdl = self.avgdl or 1.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, count in Counter(doc).items():
                self.postings[term].append((i, count))
        self.idf = {
            t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self.postings.items()
        }

    def score(self, query_tokens: list[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for i, tf in posting:
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / norm
        return scores


class CodeIndex:
    def __init__(self, graph: GraphStore) -> None:
        self.graph = graph
        self.scopes: list[CodeEntity] = []
        for t in (EntityType.FILE, EntityType.CLASS, EntityType.FUNCTION, EntityType.METHOD):
            self.scopes.extend(graph.get_entities_by_type(t))

        self.statements_by_scope: dict[str, list[CodeEntity]] = defaultdict(list)
        for stmt in graph.get_entities_by_type(EntityType.STATEMENT):
            if stmt.parent_id:
                self.statements_by_scope[stmt.parent_id].append(stmt)

        self.children_names: dict[str, list[str]] = defaultdict(list)
        self.by_name: dict[str, list[CodeEntity]] = defaultdict(list)
        self.by_qualified_name: dict[str, list[CodeEntity]] = defaultdict(list)
        self.by_path: dict[str, list[CodeEntity]] = defaultdict(list)
        # Maps every path suffix (e.g. "pkg/module.py") to scopes in that file.
        # This preserves suffix matching without scanning every known path.
        self.by_path_suffix: dict[str, list[CodeEntity]] = defaultdict(list)
        for e in self.scopes:
            if e.parent_id:
                self.children_names[e.parent_id].append(e.name)
            if e.name:
                self.by_name[e.name].append(e)
            if e.qualified_name:
                self.by_qualified_name[e.qualified_name].append(e)
            if e.path:
                self.by_path[e.path].append(e)
                normalized_path = e.path.replace("\\", "/").lstrip("./")
                parts = normalized_path.split("/")
                for i in range(len(parts)):
                    suffix = "/".join(parts[i:])
                    self.by_path_suffix[suffix].append(e)

        self._bm25 = BM25([self._doc(e) for e in self.scopes])
        self.idf = self._bm25.idf

    def _doc(self, e: CodeEntity) -> list[str]:
        tokens = tokenize(e.name or "") * 3
        tokens += tokenize(e.qualified_name or "")
        tokens += tokenize(e.path or "")
        if e.type == EntityType.CLASS:
            tokens += tokenize(" ".join(self.children_names.get(e.id, [])))
        if e.type != EntityType.FILE:
            for stmt in self.statements_by_scope.get(e.id, []):
                tokens += tokenize(statement_header(stmt)[0])
        return tokens

    def search_scopes(self, query_tokens: list[str], limit: int = 100):
        raw = self._bm25.score(query_tokens)
        if not raw:
            return []
        top = sorted(raw.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        best = top[0][1] or 1.0
        return [(self.scopes[i], s / best) for i, s in top]

    def scopes_for_path(self, path: str) -> list[CodeEntity]:
        path = path.lstrip("./")
        out: list[CodeEntity] = []
        for key, items in self.by_path.items():
            if key == path or key.endswith("/" + path):
                out.extend(items)
        return out

    def resolve_symbol(self, symbol: str) -> list[CodeEntity]:
        """Resolve an explicit symbol against exact names in the repository."""
        exact_qualified = self.by_qualified_name.get(symbol)
        if exact_qualified:
            return list(exact_qualified)
        if "." not in symbol:
            return list(self.by_name.get(symbol, []))

        # Permit a qualified issue name whose module prefix differs from the
        # repository's package prefix only when its qualified suffix is unique.
        suffix_matches = [
            entity
            for entity in self.by_name.get(symbol.rsplit(".", 1)[-1], [])
            if entity.qualified_name
            and entity.qualified_name.endswith("." + symbol)
        ]
        if len(suffix_matches) == 1:
            return suffix_matches
        if suffix_matches:
            return []
        short_matches = self.by_name.get(symbol.rsplit(".", 1)[-1], [])
        return list(short_matches) if len(short_matches) == 1 else []
