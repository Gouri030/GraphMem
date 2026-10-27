"""History evidence provider (ADR-12/13/16): git churn as ranking evidence.

Code that changed often and recently is more likely to hold a bug, and
commit subjects such as "fix tokenizer crash" are lexical clues.

v1 - file level
    file score    = min(1, sum(recency weight of each commit touching it) / 5)
    recency weight = exp(-age_days / 180)
    A commit whose subject shares >= 2 tokens with the issue adds 0.2 to its
    own contribution before summing and capping.

v2 - function level (G02)
    A second `git log -p -U0` reads the *hunk headers*. With git's built-in
    Python diff driver a header names the nearest preceding `def`/`class`
    line, e.g. `@@ -16 +16 @@ def m2(self):`, so a commit can be attributed
    to the function it changed without parsing any source. For FUNCTION,
    METHOD and CLASS scopes:

        weight = 0.4 * file score + 0.6 * function score

    where the function score uses only the commits that touched that
    function (same recency weights and message bonus). Untouched siblings in a
    hot file keep the damped file-level share, so they still rank above cold
    files but below the function that actually changed. A commit that
    *creates* a file counts for every function in it.

    Known limits of header-based attribution (it is a heuristic, not a parse):
      * a hunk spanning several functions is credited to the first only;
      * module-level code placed after a function is credited to that function;
      * functions are matched by bare name, so same-named methods in one file
        share credit.

Anti-leakage (ADR-13): both commands run `git log <as_of>`, so only commits
reachable from `as_of` are seen. Age is measured from the newest commit in
that window, never from today's date.
"""
from __future__ import annotations

import logging
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from graphmem.indexing.code_index import tokenize
from graphmem.models.entities import CodeEntity, EntityType
from graphmem.retrieval.evidence import Evidence

log = logging.getLogger(__name__)

DECAY_DAYS = 180.0        # recency weight: exp(-age_days / 180)
SATURATION = 5.0          # summed recency weight at which a score reaches 1.0
MESSAGE_BONUS = 0.2       # added to a commit's contribution on a subject match
MIN_SHARED_TOKENS = 2     # subject/issue tokens needed to earn the bonus
MAX_LISTED_COMMITS = 3    # commits quoted in Evidence.detail
FILE_SHARE = 0.4          # v2: share of a function's weight inherited from its file
FUNCTION_SHARE = 1.0 - FILE_SHARE
SECONDS_PER_DAY = 86400.0

_COMMIT_PREFIX = "@@C|"
_KIND_BY_TYPE = {
    EntityType.FUNCTION: "def",
    EntityType.METHOD: "def",
    EntityType.CLASS: "class",
}
_HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@(?: (.*))?$")
_SCOPE_HEADER_RE = re.compile(r"^\s*(?:async\s+)?(def|class)\s+(\w+)")


@dataclass(frozen=True)
class Commit:
    sha: str
    unix_time: int
    subject: str
    files: tuple[str, ...]


@dataclass
class FileTouches:
    """What one commit did to one file, as far as hunk headers can tell."""

    created: bool = False
    scopes: set[tuple[str, str]] = field(default_factory=set)  # {("def", "f"), ("class", "K")}


def _parse_header(line: str) -> tuple[str, int, str] | None:
    """'@@C|sha|unix|subject' -> (sha, unix, subject). Subject may contain '|'."""
    parts = line.split("|", 3)
    if len(parts) != 4:
        return None
    try:
        return parts[1], int(parts[2]), parts[3]
    except ValueError:
        return None


def parse_git_log(output: str) -> list[Commit]:
    """Parse `--name-only --format=@@C|%H|%ct|%s` output, in log order.

    A line starting with '@@C|' opens a commit; every following non-empty
    line is a file path. Malformed header lines are skipped along with their
    files.
    """
    commits: list[Commit] = []
    header: tuple[str, int, str] | None = None
    files: list[str] = []

    def flush() -> None:
        if header is not None:
            sha, unix_time, subject = header
            commits.append(Commit(sha, unix_time, subject, tuple(dict.fromkeys(files))))

    for raw in output.split("\n"):
        line = raw.rstrip("\r")
        if line.startswith(_COMMIT_PREFIX):
            flush()
            files = []
            header = _parse_header(line)
        elif line and header is not None:
            files.append(line)
    flush()
    return commits


def parse_git_patch_log(output: str) -> dict[str, dict[str, FileTouches]]:
    """Parse `git log -p -U0 --format=@@C|%H|%ct|%s` into {sha: {path: FileTouches}}.

    Hunk bodies are consumed by their declared line counts, so an added line
    that happens to read `+++ b/x.py` is never mistaken for a file header.
    """
    result: dict[str, dict[str, FileTouches]] = {}
    touches: dict[str, FileTouches] | None = None
    current: FileTouches | None = None
    pending_created = False
    old_left = new_left = 0

    for raw in output.split("\n"):
        line = raw.rstrip("\r")

        if old_left > 0 or new_left > 0:  # inside a hunk body
            if line.startswith("-"):
                old_left -= 1
                continue
            if line.startswith("+"):
                new_left -= 1
                continue
            if line.startswith("\\"):  # "\ No newline at end of file"
                continue
            old_left = new_left = 0  # malformed: resynchronise on this line

        if line.startswith(_COMMIT_PREFIX):
            header = _parse_header(line)
            touches = result.setdefault(header[0], {}) if header else None
            current, pending_created = None, False
        elif line.startswith("diff --git "):
            current, pending_created = None, False
        elif line.startswith("new file mode"):
            pending_created = True
        elif line.startswith("+++ "):
            current = None
            target = line[4:].rstrip("\t")  # git appends a tab to names containing spaces
            if touches is not None and target.startswith("b/") and not target.startswith('b/"'):
                current = touches.setdefault(target[2:], FileTouches())
                current.created = current.created or pending_created
        elif current is not None and line.startswith("@@"):
            match = _HUNK_RE.match(line)
            if match is None:
                continue
            old_left = int(match.group(1)) if match.group(1) is not None else 1
            new_left = int(match.group(2)) if match.group(2) is not None else 1
            header_text = match.group(3)
            found = _SCOPE_HEADER_RE.match(header_text) if header_text else None
            if found:
                current.scopes.add((found.group(1), found.group(2)))
    return result


def _normalize(path: str | None) -> str | None:
    """Entity paths are repo-relative; compare them in git's '/' form."""
    if not path:
        return None
    path = path.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path or None


class HistoryProvider:
    """EvidenceProvider (signal='history') backed by `git log`.

    `function_level=False` reproduces v1 (file-level evidence only) and skips
    the second git command.

    If `repo_path` is not a git repository, or has no commits, the provider
    is created empty (`available` is False, `disabled_reason` says why) and
    `score()` returns `{}`; a warning is logged. A missing directory or an
    unknown `as_of` in a repository that does have history raises
    RuntimeError, because those are configuration mistakes that would
    otherwise silently switch the history signal off.
    """

    signal = "history"

    def __init__(
        self,
        repo_path: str,
        as_of: str = "HEAD",
        max_commits: int = 300,
        function_level: bool = True,
    ) -> None:
        # as_of may come from benchmark data; never let it parse as a git option.
        if not as_of or as_of.startswith("-"):
            raise ValueError(f"invalid as_of revision: {as_of!r}")
        if max_commits < 1:
            raise ValueError("max_commits must be >= 1")

        self.repo_path = str(repo_path)
        self.as_of = as_of
        self.max_commits = max_commits
        self.function_level = function_level
        self.disabled_reason: str | None = None

        self.commits: list[Commit] = []
        self._by_file: dict[str, list[int]] = {}
        self._patched_paths: set[str] = set()
        self._created: dict[str, list[int]] = {}
        self._func_hits: dict[str, dict[tuple[str, str], list[int]]] = {}
        self._weights: list[float] = []
        self._subject_tokens: list[frozenset[str]] = []
        self.reference_time = 0

        output = self._run_file_log()
        if output is None:  # disabled: not a repo / no commits
            return

        self.commits = parse_git_log(output)

        # Age is relative to the newest commit in the window (the as_of
        # commit), not to the wall clock.
        self.reference_time = max((c.unix_time for c in self.commits), default=0)
        self._weights = [
            math.exp(-max(0.0, (self.reference_time - c.unix_time) / SECONDS_PER_DAY) / DECAY_DAYS)
            for c in self.commits
        ]
        self._subject_tokens = [frozenset(tokenize(c.subject)) for c in self.commits]
        for i, commit in enumerate(self.commits):
            for path in commit.files:
                self._by_file.setdefault(path, []).append(i)

        if self.function_level and self.commits:
            self._build_function_index()

    @property
    def available(self) -> bool:
        return self.disabled_reason is None

    # ------------------------------------------------------------------ git
    def _git(self, *args: str) -> subprocess.CompletedProcess:
        cmd = ["git", "-C", self.repo_path, "-c", "core.quotepath=off", *args]
        try:
            return subprocess.run(
                cmd, capture_output=True, encoding="utf-8", errors="replace", check=False
            )
        except OSError as exc:  # git not installed
            raise RuntimeError(f"could not run git: {exc}") from exc

    def _run_file_log(self) -> str | None:
        """v1 command. Returns None when the provider is disabled."""
        # `--relative` makes paths relative to repo_path rather than the git
        # top-level, so they match CodeEntity.path even when the parsed folder
        # is a subdirectory of a larger repository.
        proc = self._git(
            "log", self.as_of, "--no-merges", "-n", str(self.max_commits),
            "--relative", "--name-only", f"--format={_COMMIT_PREFIX}%H|%ct|%s",
            "--", "*.py",
        )
        if proc.returncode == 0:
            return proc.stdout
        self._disable_or_raise(proc.stderr.strip())
        return None

    def _disable_or_raise(self, stderr: str) -> None:
        if not Path(self.repo_path).is_dir():
            raise RuntimeError(f"repository path is not a directory: {self.repo_path!r}")
        if self._git("rev-parse", "--git-dir").returncode != 0:
            reason = "not a git repository"
        elif self._git("rev-parse", "--verify", "--quiet", "HEAD^{commit}").returncode != 0:
            reason = "repository has no commits"
        else:
            raise RuntimeError(
                f"git log failed for {self.repo_path!r} at {self.as_of!r}: {stderr}"
            )
        self.disabled_reason = f"{reason}: {stderr}" if stderr else reason
        log.warning("HistoryProvider disabled for %s (%s)", self.repo_path, self.disabled_reason)

    def _build_function_index(self) -> None:
        """v2 command: read hunk headers to learn which functions each commit touched."""
        with tempfile.TemporaryDirectory(prefix="graphmem-attrs-") as tmp:
            attrs = Path(tmp) / "attributes"
            # Git's default driver reports `class K:` for a change inside a method;
            # the built-in python driver reports the enclosing `def`.
            attrs.write_text("*.py diff=python\n", encoding="utf-8")
            proc = self._git(
                "-c", f"core.attributesFile={attrs.as_posix()}",
                "log", self.as_of, "--no-merges", "-n", str(self.max_commits),
                "--relative", "-p", "-U0", "--no-color", "--no-ext-diff", "--no-textconv",
                "--src-prefix=a/", "--dst-prefix=b/",  # immune to diff.noprefix etc.
                f"--format={_COMMIT_PREFIX}%H|%ct|%s", "--", "*.py",
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"git log -p failed for {self.repo_path!r} at {self.as_of!r}: {proc.stderr.strip()}"
            )

        sha_index = {c.sha: i for i, c in enumerate(self.commits)}
        for sha, per_file in parse_git_patch_log(proc.stdout).items():
            i = sha_index.get(sha)
            if i is None:  # history moved between the two git calls
                continue
            for path, touched in per_file.items():
                self._patched_paths.add(path)
                if touched.created:
                    self._created.setdefault(path, []).append(i)
                for key in touched.scopes:
                    self._func_hits.setdefault(path, {}).setdefault(key, []).append(i)

    # ---------------------------------------------------------------- scoring
    def score(self, scopes: list[CodeEntity], issue: str) -> dict[str, list[Evidence]]:
        """Evidence for every scope whose file was touched in the window."""
        if not scopes or not self._by_file:
            return {}

        issue_tokens = set(tokenize(issue))
        per_file: dict[str, tuple[float, str, list[str]] | None] = {}
        out: dict[str, list[Evidence]] = {}

        for scope in scopes:
            path = _normalize(scope.path)
            if path is None:
                continue
            if path not in per_file:
                per_file[path] = self._file_evidence(path, issue_tokens)
            found = per_file[path]
            if found is None:
                continue

            kind = _KIND_BY_TYPE.get(scope.type)
            if kind is not None and path in self._patched_paths:
                evidence = self._scope_evidence(scope, kind, path, found, issue_tokens)
            else:
                weight, source, listed = found
                evidence = Evidence(
                    "history", source, weight, {"commits": list(listed), "level": "file"}
                )
            out[scope.id] = [evidence]
        return out

    def _contribution(self, i: int, issue_tokens: set[str]) -> float:
        """One commit's recency weight, plus the message bonus on a subject match."""
        contribution = self._weights[i]
        if len(self._subject_tokens[i] & issue_tokens) >= MIN_SHARED_TOKENS:
            contribution += MESSAGE_BONUS
        return contribution

    def _newest_first(self, indices) -> list[int]:
        # Log order breaks timestamp ties.
        return sorted(indices, key=lambda i: (-self.commits[i].unix_time, i))

    def _listed(self, newest: list[int]) -> list[str]:
        return [f"{self.commits[i].sha[:7]} {self.commits[i].subject}" for i in newest[:MAX_LISTED_COMMITS]]

    def _file_evidence(
        self, path: str, issue_tokens: set[str]
    ) -> tuple[float, str, list[str]] | None:
        indices = self._by_file.get(path)
        if not indices:
            return None

        total = sum(self._contribution(i, issue_tokens) for i in indices)
        weight = min(1.0, total / SATURATION)
        if weight <= 0.0:
            return None

        newest = self._newest_first(indices)
        latest = self.commits[newest[0]]
        noun = "commit" if len(indices) == 1 else "commits"
        source = (
            f"{len(indices)} {noun} touched {PurePosixPath(path).name} "
            f"(latest {latest.sha[:7]}: '{latest.subject}')"
        )
        return weight, source, self._listed(newest)

    def _scope_evidence(
        self,
        scope: CodeEntity,
        kind: str,
        path: str,
        file_found: tuple[float, str, list[str]],
        issue_tokens: set[str],
    ) -> Evidence:
        file_weight, _, file_listed = file_found
        file_name = PurePosixPath(path).name
        label = f"class {scope.name}" if kind == "class" else f"{scope.name}()"

        # A commit that created the file authored every function in it.
        touching = set(self._created.get(path, ()))
        touching.update(self._func_hits.get(path, {}).get((kind, scope.name), ()))
        function_total = sum(self._contribution(i, issue_tokens) for i in touching)
        function_weight = min(1.0, function_total / SATURATION)
        weight = min(1.0, FILE_SHARE * file_weight + FUNCTION_SHARE * function_weight)

        detail = {
            "function": scope.qualified_name or scope.name,
            "file_weight": round(file_weight, 4),
            "function_weight": round(function_weight, 4),
        }
        if touching:
            newest = self._newest_first(touching)
            latest = self.commits[newest[0]]
            noun = "commit" if len(newest) == 1 else "commits"
            source = (
                f"{len(newest)} {noun} touched {label} in {file_name} "
                f"(latest {latest.sha[:7]}: '{latest.subject}')"
            )
            detail.update(commits=self._listed(newest), level="function")
        else:
            file_indices = self._by_file[path]
            latest = self.commits[self._newest_first(file_indices)[0]]
            noun = "commit" if len(file_indices) == 1 else "commits"
            source = (
                f"{file_name} changed in {len(file_indices)} {noun}, none touching {label} "
                f"(latest {latest.sha[:7]}: '{latest.subject}')"
            )
            detail.update(commits=list(file_listed), level="file")
        return Evidence("history", source, weight, detail)
