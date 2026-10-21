"""Structured signals extracted from issue descriptions and Python tracebacks."""
from __future__ import annotations

from dataclasses import dataclass, field
import re

from graphmem.indexing.code_index import extract_paths, extract_symbols, tokenize

_FRAME = re.compile(
    r"""File ["'](?P<path>[^"']+)["'],\s*line\s+(?P<line>\d+)"""
    r"""(?:,\s*in\s+(?P<function>[^\s]+))?"""
)
_EXCEPTION = re.compile(
    r"(?m)^\s*(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt|Exit))"
    r"(?::\s*(?P<message>[^\n]+))?\s*$"
)
_FENCE = re.compile(r"```[^\n]*\n.*?```", re.S)


@dataclass(frozen=True)
class TracebackFrame:
    path: str
    line: int
    function: str | None = None


@dataclass
class IssueSignals:
    prose: str
    code: str
    symbols: set[str] = field(default_factory=set)
    paths: set[str] = field(default_factory=set)
    frames: list[TracebackFrame] = field(default_factory=list)
    exceptions: list[tuple[str, str]] = field(default_factory=list)
    error_terms: set[str] = field(default_factory=set)

    @property
    def query_tokens(self) -> list[str]:
        return tokenize(self.prose)

    @classmethod
    def parse(cls, issue: str) -> "IssueSignals":
        code_chunks = _FENCE.findall(issue)
        code = "\n".join(code_chunks)
        without_fences = _FENCE.sub(" ", issue)
        frames = []
        for match in _FRAME.finditer(issue):
            path = match.group("path").replace("\\", "/")
            frames.append(TracebackFrame(path, int(match.group("line")), match.group("function")))
        exceptions = [
            (m.group("type"), (m.group("message") or "").strip())
            for m in _EXCEPTION.finditer(issue)
        ]
        paths = extract_paths(issue)
        for frame in frames:
            paths.add(frame.path.lstrip("./"))
        symbols = extract_symbols(issue)
        for frame in frames:
            if frame.function:
                symbols.add(frame.function)
        for typ, _ in exceptions:
            symbols.add(typ.rsplit(".", 1)[-1])
        error_terms = set()
        for typ, message in exceptions:
            error_terms.update(tokenize(typ + " " + message))
        prose_lines = []
        for line in without_fences.splitlines():
            if _FRAME.search(line) or _EXCEPTION.match(line) or line.strip().startswith("Traceback"):
                continue
            prose_lines.append(line)
        return cls("\n".join(prose_lines), code, symbols, paths, frames, exceptions, error_terms)
