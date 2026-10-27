"""HistoryProvider tests (G01 file level, G02 function level, G03 fixture).

Every test builds its own throwaway git repo in tmp_path with fixed commit
timestamps, so the suite needs no network, no existing checkout, and nothing
about "today". The repos are created by `make_repo`.
"""
from __future__ import annotations

import atexit
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pytest

from graphmem.graph.builder import GraphBuilder
from graphmem.models.entities import CodeEntity, EntityType
from graphmem.parsing.python_parser import PythonParser
from graphmem.ranking.config import RankConfig
from graphmem.retrieval.localizer import Localizer
from graphmem.retrieval.providers.history import (
    HistoryProvider,
    parse_git_log,
    parse_git_patch_log,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

T0 = 1_600_000_000  # a fixed epoch far in the past: nothing here may depend on "today"
DAY = 86400

# An explicitly empty git config, so fixture repos ignore the machine's own settings
# (an empty real file is safer than relying on os.devnull semantics on Windows).
_EMPTY_CONFIG_DIR = tempfile.mkdtemp(prefix="graphmem-gitcfg-")
atexit.register(shutil.rmtree, _EMPTY_CONFIG_DIR, ignore_errors=True)
EMPTY_GITCONFIG = Path(_EMPTY_CONFIG_DIR) / "gitconfig"
EMPTY_GITCONFIG.write_text("", encoding="utf-8")


# ====================================================================== fixture
@dataclass
class Spec:
    """One commit to create: files to write ({path: text}), message, unix time."""

    files: dict[str, str]
    message: str
    when: int


class Repo(NamedTuple):
    path: Path
    shas: list[str]  # one per Spec, oldest first


def _git(path: Path, *args: str, env_extra: dict[str, str] | None = None) -> str:
    env = {
        **os.environ,
        # Hermetic: ignore the machine's own git configuration.
        "GIT_CONFIG_GLOBAL": str(EMPTY_GITCONFIG),
        "GIT_CONFIG_NOSYSTEM": "1",
        **(env_extra or {}),
    }
    done = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True, encoding="utf-8", errors="replace", env=env, check=True,
    )
    return done.stdout


def make_repo(tmp_path: Path, commits: list[Spec], name: str = "repo") -> Repo:
    """`git init`, set user.name/user.email locally, then make each commit with
    GIT_AUTHOR_DATE / GIT_COMMITTER_DATE fixed to Spec.when."""
    path = tmp_path / name
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Test User")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "commit.gpgsign", "false")

    shas = []
    for spec in commits:
        for rel, text in spec.files.items():
            target = path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))  # bytes: no CRLF translation on Windows
        _git(path, "add", "-A")
        stamp = f"{spec.when} +0000"
        _git(
            path, "commit", "-q", "--no-verify", "-m", spec.message,
            env_extra={"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp},
        )
        shas.append(_git(path, "rev-parse", "HEAD").strip())
    return Repo(path, shas)


def scope(path, kind=EntityType.FILE, name=None, qualified=None):
    name = name or path
    return CodeEntity(
        id=f"{kind.value}:{path}:{qualified or name}", type=kind, name=name,
        path=path, qualified_name=qualified or name,
    )


def only(out, entity):
    (ev,) = out[entity.id]
    return ev


def body(n: int) -> str:
    return f"x = {n}\n"


TWO_FUNCS = "def f(x):\n    a = x + 1\n    return a\n\ndef g(y):\n    b = y * 2\n    return b\n"
CLASS_SRC = (
    "class K:\n    size = 1\n\n    def m1(self):\n        return 1\n\n"
    "    def m2(self):\n        return 2\n"
)


# ============================================================ G03 required tests
def test_file_changed_in_three_recent_commits_beats_one_old_commit(tmp_path):
    repo = make_repo(tmp_path, [
        Spec({"cold.py": body(0)}, "add cold", T0),
        Spec({"hot.py": body(1)}, "add hot", T0 + 400 * DAY),
        Spec({"hot.py": body(2)}, "change hot", T0 + 401 * DAY),
        Spec({"hot.py": body(3)}, "change hot again", T0 + 402 * DAY),
    ])
    hot, cold = scope("hot.py"), scope("cold.py")
    out = HistoryProvider(str(repo.path)).score([hot, cold], "unrelated words")

    assert only(out, hot).weight > only(out, cold).weight
    assert only(out, hot).weight == pytest.approx(sum(math.exp(-d / 180) for d in (0, 1, 2)) / 5)
    assert only(out, cold).weight == pytest.approx(math.exp(-402 / 180) / 5)


def test_commits_after_as_of_are_excluded(tmp_path):
    repo = make_repo(tmp_path, [
        Spec({"a.py": body(1)}, "add a", T0),
        Spec({"a.py": body(2), "b.py": body(1)}, "Fix tokenizer crash in b", T0 + DAY),
        Spec({"a.py": body(3), "c.py": body(1)}, "later change", T0 + 2 * DAY),
    ])
    a, b, c = repo.shas
    scopes = [scope("a.py"), scope("b.py"), scope("c.py")]

    past = HistoryProvider(str(repo.path), as_of=a)
    head = HistoryProvider(str(repo.path))

    assert [cm.sha for cm in past.commits] == [a]          # b and c are provably absent
    past_out = past.score(scopes, "tokenizer crash")
    assert set(past_out) == {scopes[0].id}                 # b.py / c.py did not exist yet
    ev = only(past_out, scopes[0])
    assert ev.detail["commits"] == [f"{a[:7]} add a"]
    assert b[:7] not in ev.source and c[:7] not in ev.source

    # Without as_of the same repo does show them, so the exclusion above is real.
    assert [cm.sha for cm in head.commits] == [c, b, a]
    assert set(head.score(scopes, "x")) == {s.id for s in scopes}
    assert [cm.sha for cm in HistoryProvider(str(repo.path), as_of=b).commits] == [b, a]


def test_commit_subject_sharing_two_tokens_adds_message_bonus(tmp_path):
    repo = make_repo(tmp_path, [Spec({"parser.py": body(1)}, "Fix tokenizer crash", T0)])
    provider = HistoryProvider(str(repo.path))
    target = scope("parser.py")

    none_shared = only(provider.score([target], "banana smoothie"), target)
    one_shared = only(provider.score([target], "tokenizer fails sometimes"), target)
    two_shared = only(provider.score([target], "tokenizer crash on empty input"), target)

    assert none_shared.weight == pytest.approx(1 / 5)
    assert one_shared.weight == pytest.approx(1 / 5)             # one token is not enough
    assert two_shared.weight == pytest.approx((1 + 0.2) / 5)     # bonus joins the sum, then /5
    assert two_shared.weight > one_shared.weight


def test_commit_touching_only_f_scores_f_above_g(tmp_path):
    f_edited = TWO_FUNCS.replace("x + 1", "x + 100")
    repo = make_repo(tmp_path, [
        Spec({"m.py": TWO_FUNCS}, "add m", T0),
        Spec({"m.py": f_edited}, "tweak f", T0 + DAY),
    ])
    f = scope("m.py", EntityType.FUNCTION, "f")
    g = scope("m.py", EntityType.FUNCTION, "g")
    file_scope = scope("m.py")
    out = HistoryProvider(str(repo.path)).score([f, g, file_scope], "unrelated")

    ev_f, ev_g, ev_file = only(out, f), only(out, g), only(out, file_scope)
    assert ev_f.weight > ev_g.weight > 0

    # Numbers: creation counts for both functions; the edit counts for f only.
    w_new, w_old = 1.0, math.exp(-1 / 180)
    file_w = (w_new + w_old) / 5
    assert ev_file.weight == pytest.approx(file_w)                          # file level unchanged
    assert ev_f.weight == pytest.approx(0.4 * file_w + 0.6 * file_w)
    assert ev_g.weight == pytest.approx(0.4 * file_w + 0.6 * w_old / 5)
    assert ev_f.detail["level"] == ev_g.detail["level"] == "function"
    assert ev_f.source.startswith("2 commits touched f() in m.py (latest ")
    assert ev_g.source.startswith("1 commit touched g() in m.py (latest ")


def test_zero_commit_repo_and_non_git_directory_return_empty(tmp_path, caplog, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))  # never discover a parent repo
    empty_repo = tmp_path / "empty-repo"
    empty_repo.mkdir()
    _git(empty_repo, "init", "-q")
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    (plain_dir / "a.py").write_text("x = 1\n", encoding="utf-8")

    for directory, reason in ((empty_repo, "no commits"), (plain_dir, "not a git repository")):
        provider = HistoryProvider(str(directory))          # must not raise
        assert provider.score([scope("a.py")], "anything") == {}
        assert provider.score([], "anything") == {}
        assert provider.commits == []
        assert provider.available is False
        assert reason in provider.disabled_reason
    assert "HistoryProvider disabled" in caplog.text        # not silent: it is logged


def test_weights_in_unit_interval_and_every_evidence_is_history_with_detail(tmp_path):
    specs = [Spec({"m.py": TWO_FUNCS, "k.py": CLASS_SRC, "u.py": body(0)}, "add all", T0)]
    for i in range(1, 8):  # enough churn to hit the cap on some scopes
        specs.append(Spec({"m.py": TWO_FUNCS.replace("x + 1", f"x + {i + 1}")},
                          "Fix tokenizer crash again", T0 + i * DAY))
    repo = make_repo(tmp_path, specs)

    scopes = [
        scope("m.py"), scope("m.py", EntityType.FUNCTION, "f"), scope("m.py", EntityType.FUNCTION, "g"),
        scope("k.py"), scope("k.py", EntityType.CLASS, "K"),
        scope("k.py", EntityType.METHOD, "m1", "K.m1"), scope("u.py"),
    ]
    for issue in ("", "tokenizer crash", "completely unrelated"):
        for flag in (True, False):
            out = HistoryProvider(str(repo.path), function_level=flag).score(scopes, issue)
            assert len(out) == len(scopes)
            for evidence_list in out.values():
                for ev in evidence_list:
                    assert ev.type == "history"
                    assert isinstance(ev.detail, dict) and ev.detail
                    assert 0.0 <= ev.weight <= 1.0
                    assert ev.detail["level"] in {"file", "function"}
                    assert isinstance(ev.detail["commits"], list) and len(ev.detail["commits"]) <= 3
                    json.dumps(ev.to_dict())                 # JSON-safe for the API / CLI output


# ============================================================ v1 behaviours kept
def test_evidence_format_matches_spec(tmp_path):
    subjects = ["add parser", "tweak parser", "Fix tokenizer crash", "refactor parser"]
    repo = make_repo(tmp_path, [
        Spec({"parser.py": body(i)}, s, T0 + i * DAY) for i, s in enumerate(subjects)
    ])
    provider = HistoryProvider(str(repo.path))
    assert provider.signal == "history"

    target = scope("parser.py")
    ev = only(provider.score([target], "unrelated words"), target)
    assert ev.weight == pytest.approx(sum(math.exp(-d / 180) for d in (0, 1, 2, 3)) / 5)
    assert ev.detail["level"] == "file"
    assert ev.detail["commits"] == [   # newest first, at most 3
        f"{repo.shas[3][:7]} refactor parser",
        f"{repo.shas[2][:7]} Fix tokenizer crash",
        f"{repo.shas[1][:7]} tweak parser",
    ]
    assert ev.source == (
        f"4 commits touched parser.py (latest {repo.shas[3][:7]}: 'refactor parser')"
    )


def test_singular_wording_for_one_commit(tmp_path):
    repo = make_repo(tmp_path, [Spec({"solo.py": body(1)}, "add solo", T0)])
    target = scope("solo.py")
    ev = only(HistoryProvider(str(repo.path)).score([target], "x"), target)
    assert ev.source == f"1 commit touched solo.py (latest {repo.shas[0][:7]}: 'add solo')"


def test_age_is_measured_from_as_of_commit_not_today(tmp_path):
    repo = make_repo(tmp_path, [
        Spec({"old.py": body(1)}, "add old", T0),
        Spec({"new.py": body(1)}, "add new", T0 + 180 * DAY),   # exactly 180 days newer
    ])
    old, new = scope("old.py"), scope("new.py")
    out = HistoryProvider(str(repo.path)).score([old, new], "banana")
    # Relative to the newest commit: age 0 -> w=1, age 180d -> w=e^-1. (Measured
    # from today both weights would be ~0, since this repo is from 2020.)
    assert only(out, new).weight == pytest.approx(1 / 5)
    assert only(out, old).weight == pytest.approx(math.exp(-1) / 5)


def test_weight_is_capped_at_one_even_with_bonus(tmp_path):
    repo = make_repo(tmp_path, [
        Spec({"hot.py": body(i)}, "Fix tokenizer crash", T0) for i in range(7)
    ])
    target = scope("hot.py")
    provider = HistoryProvider(str(repo.path))
    assert only(provider.score([target], "x"), target).weight == 1.0
    assert only(provider.score([target], "tokenizer crash"), target).weight == 1.0


def test_untouched_nonpython_and_pathless_scopes_get_nothing(tmp_path):
    repo = make_repo(tmp_path, [Spec({"a.py": body(1), "README.md": "docs\n"}, "add files", T0)])
    pathless = CodeEntity(id="e:none", type=EntityType.FUNCTION, name="f", path=None)
    scopes = [scope("a.py"), scope("README.md"), scope("missing.py"), pathless]
    out = HistoryProvider(str(repo.path)).score(scopes, "x")
    assert set(out) == {scopes[0].id}


def test_windows_style_entity_paths_match_git_paths(tmp_path):
    repo = make_repo(tmp_path, [Spec({"pkg/mod.py": body(1)}, "add mod", T0)])
    target = scope("pkg\\mod.py")
    assert only(HistoryProvider(str(repo.path)).score([target], "x"), target).type == "history"


def test_parsed_folder_inside_larger_repo_uses_relative_paths(tmp_path):
    repo = make_repo(tmp_path, [Spec({"proj/a.py": body(1), "other/c.py": body(1)}, "add", T0)])
    provider = HistoryProvider(str(repo.path / "proj"))
    assert provider.commits[0].files == ("a.py",)      # not 'proj/a.py' or 'other/c.py'
    a, c = scope("a.py"), scope("c.py")
    assert set(provider.score([a, c], "x")) == {a.id}


def test_max_commits_limits_the_window(tmp_path):
    repo = make_repo(tmp_path, [Spec({"a.py": body(i)}, f"change {i}", T0 + i * DAY) for i in range(4)])
    provider = HistoryProvider(str(repo.path), max_commits=2)
    assert len(provider.commits) == 2
    target = scope("a.py")
    assert only(provider.score([target], "x"), target).source.startswith("2 commits touched")


def test_history_free_repo_gives_no_evidence(tmp_path):
    repo = make_repo(tmp_path, [Spec({"README.md": "docs\n"}, "docs only", T0)])
    provider = HistoryProvider(str(repo.path))
    assert provider.available and provider.commits == []   # a repo, just no .py history
    assert provider.score([scope("a.py")], "x") == {}


def test_scopes_in_a_touched_file_get_independent_evidence(tmp_path):
    repo = make_repo(tmp_path, [Spec({"pkg/mod.py": body(1)}, "add mod", T0)])
    scopes = [
        scope("pkg/mod.py"),
        scope("pkg/mod.py", EntityType.CLASS, "Mod"),
        scope("pkg/mod.py", EntityType.METHOD, "run", "Mod.run"),
    ]
    out = HistoryProvider(str(repo.path)).score(scopes, "x")
    evs = [only(out, s) for s in scopes]
    assert len({e.weight for e in evs}) == 1               # file created in one commit: all equal
    assert len({id(e) for e in evs}) == 3                  # no shared instances
    assert len({id(e.detail["commits"]) for e in evs}) == 3


# ============================================================== error handling
def test_rejects_bad_arguments(tmp_path):
    repo = make_repo(tmp_path, [Spec({"a.py": body(1)}, "add a", T0)])
    for bad in ("", "--output=leak.txt", "-n1"):
        with pytest.raises(ValueError):
            HistoryProvider(str(repo.path), as_of=bad)
    with pytest.raises(ValueError):
        HistoryProvider(str(repo.path), max_commits=0)


def test_misconfiguration_still_raises(tmp_path):
    repo = make_repo(tmp_path, [Spec({"a.py": body(1)}, "add a", T0)])
    with pytest.raises(RuntimeError):                      # real history, unknown revision
        HistoryProvider(str(repo.path), as_of="no-such-revision")
    with pytest.raises(RuntimeError):                      # directory does not exist at all
        HistoryProvider(str(tmp_path / "does-not-exist"))


# ============================================================ v2: function level
def test_method_level_attribution_uses_python_diff_driver(tmp_path):
    """Git's default driver would report `class K:` for a change inside m2()."""
    repo = make_repo(tmp_path, [
        Spec({"k.py": CLASS_SRC}, "add k", T0),
        Spec({"k.py": CLASS_SRC.replace("return 2", "return 22")}, "change m2", T0 + DAY),
    ])
    provider = HistoryProvider(str(repo.path))
    m1 = scope("k.py", EntityType.METHOD, "m1", "K.m1")
    m2 = scope("k.py", EntityType.METHOD, "m2", "K.m2")
    klass = scope("k.py", EntityType.CLASS, "K")
    out = provider.score([m1, m2, klass], "x")

    assert only(out, m2).weight > only(out, m1).weight
    assert only(out, m2).detail["level"] == "function"
    assert only(out, m2).detail["function"] == "K.m2"
    assert only(out, m1).weight == pytest.approx(only(out, klass).weight)  # m2's edit is not K's


def test_class_body_change_credits_the_class_not_its_methods(tmp_path):
    repo = make_repo(tmp_path, [
        Spec({"k.py": CLASS_SRC}, "add k", T0),
        Spec({"k.py": CLASS_SRC.replace("size = 1", "size = 2")}, "change attribute", T0 + DAY),
    ])
    klass = scope("k.py", EntityType.CLASS, "K")
    m1 = scope("k.py", EntityType.METHOD, "m1", "K.m1")
    out = HistoryProvider(str(repo.path)).score([klass, m1], "x")
    assert only(out, klass).weight > only(out, m1).weight
    assert only(out, klass).source.startswith("2 commits touched class K in k.py")


def test_commit_that_creates_a_file_credits_every_function_in_it(tmp_path):
    repo = make_repo(tmp_path, [Spec({"m.py": TWO_FUNCS}, "add m", T0)])
    f = scope("m.py", EntityType.FUNCTION, "f")
    g = scope("m.py", EntityType.FUNCTION, "g")
    file_scope = scope("m.py")
    out = HistoryProvider(str(repo.path)).score([f, g, file_scope], "x")
    for s in (f, g):
        assert only(out, s).detail["level"] == "function"
        assert only(out, s).weight == pytest.approx(only(out, file_scope).weight)


def test_untouched_function_keeps_only_the_damped_file_share(tmp_path):
    src = "def f():\n    return 1\n\ndef g():\n    return 2\n"
    repo = make_repo(tmp_path, [
        Spec({"m.py": src}, "add m", T0),
        Spec({"m.py": src.replace("return 1", "return 11")}, "edit f", T0 + DAY),
        Spec({"m.py": src.replace("return 1", "return 12")}, "edit f again", T0 + 2 * DAY),
    ])
    # A 2-commit window sees only the two edits of f(): g() was never touched in it.
    provider = HistoryProvider(str(repo.path), max_commits=2)
    f = scope("m.py", EntityType.FUNCTION, "f")
    g = scope("m.py", EntityType.FUNCTION, "g")
    file_scope = scope("m.py")
    out = provider.score([f, g, file_scope], "x")

    file_w = (1.0 + math.exp(-1 / 180)) / 5
    assert only(out, file_scope).weight == pytest.approx(file_w)
    assert only(out, f).weight == pytest.approx(file_w)               # f owns every commit in the window
    assert only(out, g).weight == pytest.approx(0.4 * file_w)         # g: file share only
    assert 0 < only(out, g).weight < only(out, f).weight

    ev = only(out, g)
    assert ev.detail["level"] == "file" and ev.detail["function_weight"] == 0.0
    assert ev.source.startswith("m.py changed in 2 commits, none touching g() (latest ")
    assert [c.split(" ", 1)[1] for c in ev.detail["commits"]] == ["edit f again", "edit f"]


def test_message_bonus_applies_at_function_level(tmp_path):
    repo = make_repo(tmp_path, [
        Spec({"m.py": TWO_FUNCS}, "add m", T0),
        Spec({"m.py": TWO_FUNCS.replace("y * 2", "y * 3")}, "Fix tokenizer crash in g", T0 + DAY),
    ])
    provider = HistoryProvider(str(repo.path))
    f = scope("m.py", EntityType.FUNCTION, "f")
    g = scope("m.py", EntityType.FUNCTION, "g")
    plain = provider.score([f, g], "banana smoothie")
    matched = provider.score([f, g], "tokenizer crash")

    w0 = math.exp(-1 / 180)  # the creation commit; the fix commit has weight 1
    # g owns the matching commit, so it gets the full bonus ...
    assert only(plain, g).weight == pytest.approx((w0 + 1.0) / 5)
    assert only(matched, g).weight == pytest.approx((w0 + 1.2) / 5)
    # ... f only gets it through its 40% file share.
    assert only(plain, f).weight == pytest.approx(0.4 * (w0 + 1.0) / 5 + 0.6 * w0 / 5)
    assert only(matched, f).weight == pytest.approx(0.4 * (w0 + 1.2) / 5 + 0.6 * w0 / 5)


def test_file_scope_is_unchanged_by_function_level(tmp_path):
    repo = make_repo(tmp_path, [
        Spec({"m.py": TWO_FUNCS}, "add m", T0),
        Spec({"m.py": TWO_FUNCS.replace("x + 1", "x + 2")}, "tweak f", T0 + DAY),
    ])
    file_scope = scope("m.py")
    v1 = HistoryProvider(str(repo.path), function_level=False).score([file_scope], "x")
    v2 = HistoryProvider(str(repo.path)).score([file_scope], "x")
    assert only(v1, file_scope).to_dict() == only(v2, file_scope).to_dict()


def test_function_level_off_reproduces_v1(tmp_path):
    repo = make_repo(tmp_path, [
        Spec({"m.py": TWO_FUNCS}, "add m", T0),
        Spec({"m.py": TWO_FUNCS.replace("x + 1", "x + 2")}, "tweak f", T0 + DAY),
    ])
    f = scope("m.py", EntityType.FUNCTION, "f")
    g = scope("m.py", EntityType.FUNCTION, "g")
    out = HistoryProvider(str(repo.path), function_level=False).score([f, g], "x")
    assert only(out, f).weight == only(out, g).weight
    assert only(out, f).detail["level"] == "file"


def test_user_diff_config_cannot_break_function_level(tmp_path, monkeypatch):
    """A global `diff.noprefix` / `diff.mnemonicPrefix` changes the `+++ b/` prefix git prints."""
    user_config = tmp_path / "gitconfig"
    user_config.write_text("[diff]\n\tnoprefix = true\n\tmnemonicPrefix = true\n", encoding="utf-8")
    repo = make_repo(tmp_path, [
        Spec({"m.py": TWO_FUNCS}, "add m", T0),
        Spec({"m.py": TWO_FUNCS.replace("x + 1", "x + 2")}, "tweak f", T0 + DAY),
    ])
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(user_config))
    f = scope("m.py", EntityType.FUNCTION, "f")
    g = scope("m.py", EntityType.FUNCTION, "g")
    out = HistoryProvider(str(repo.path)).score([f, g], "x")
    assert only(out, f).weight > only(out, g).weight


def test_file_with_spaces_in_its_path(tmp_path):
    repo = make_repo(tmp_path, [
        Spec({"my dir/my file.py": TWO_FUNCS}, "add", T0),
        Spec({"my dir/my file.py": TWO_FUNCS.replace("x + 1", "x + 2")}, "tweak f", T0 + DAY),
    ])
    f = scope("my dir/my file.py", EntityType.FUNCTION, "f")
    g = scope("my dir/my file.py", EntityType.FUNCTION, "g")
    out = HistoryProvider(str(repo.path)).score([f, g], "x")
    assert only(out, f).weight > only(out, g).weight


def test_function_history_respects_as_of(tmp_path):
    f_edit = TWO_FUNCS.replace("x + 1", "x + 2")
    repo = make_repo(tmp_path, [
        Spec({"m.py": TWO_FUNCS}, "add m", T0),
        Spec({"m.py": f_edit}, "tweak f", T0 + DAY),
        Spec({"m.py": f_edit.replace("y * 2", "y * 9")}, "tweak g later", T0 + 2 * DAY),
    ])
    f = scope("m.py", EntityType.FUNCTION, "f")
    g = scope("m.py", EntityType.FUNCTION, "g")
    before = HistoryProvider(str(repo.path), as_of=repo.shas[1]).score([f, g], "x")
    after = HistoryProvider(str(repo.path)).score([f, g], "x")
    assert only(before, f).weight > only(before, g).weight      # g's later edit is invisible
    assert only(after, g).detail["commits"][0].endswith("tweak g later")
    assert all("tweak g later" not in c for c in only(before, g).detail["commits"])


# ================================================================== parser units
def test_parse_git_log_handles_pipes_crlf_and_bad_headers():
    raw = (
        "@@C|aaaa|100|fix | pipe\r\n\r\na.py\r\nb/c.py\r\n"
        "@@C|bbbb|50|no files\r\n"
        "@@C|bad-header\r\nlost.py\r\n"
        "@@C|cccc|notanint|x\r\nlost2.py\r\n"
        "@@C|dddd|10|last\r\n\r\nz.py\r\n"
    )
    got = [(c.sha, c.unix_time, c.subject, c.files) for c in parse_git_log(raw)]
    assert got == [
        ("aaaa", 100, "fix | pipe", ("a.py", "b/c.py")),
        ("bbbb", 50, "no files", ()),
        ("dddd", 10, "last", ("z.py",)),
    ]


PATCH = "\n".join([
    "@@C|aaaa|100|edit things",
    "",
    "diff --git a/m.py b/m.py",
    "index 111..222 100644",
    "--- a/m.py",
    "+++ b/m.py",
    "@@ -4 +4 @@ def f(x):",
    "-    a = x + 1",
    "+    a = x + 100",
    "@@ -9,0 +10,2 @@ class K:",
    "+++ b/fake.py",            # an *added line* that looks like a file header
    "+@@ -1 +1 @@ def trap():",  # ...and one that looks like a hunk header
    "@@ -20,2 +22 @@ async def run(self):",
    "-a",
    "-b",
    "+c",
    "\\ No newline at end of file",
    "diff --git a/new.py b/new.py",
    "new file mode 100644",
    "--- /dev/null",
    "+++ b/new.py",
    "@@ -0,0 +1,2 @@",
    "+def made():",
    "+    pass",
    "diff --git a/gone.py b/gone.py",
    "deleted file mode 100644",
    "--- a/gone.py",
    "+++ /dev/null",
    "@@ -1 +0,0 @@ def dead():",
    "-x",
    "diff --git a/my file.py b/my file.py",
    "--- a/my file.py",
    "+++ b/my file.py\t",
    "@@ -1 +1 @@ def spaced():",
    "-x",
    "+y",
    "@@C|bbbb|50|second",
    "",
    "diff --git a/m.py b/m.py",
    "--- a/m.py",
    "+++ b/m.py",
    "@@ -1 +1 @@",
    "-x",
    "+y",
])


def test_parse_git_patch_log_reads_headers_and_ignores_lookalike_content():
    got = parse_git_patch_log(PATCH)
    first = got["aaaa"]

    assert first["m.py"].scopes == {("def", "f"), ("class", "K"), ("def", "run")}
    assert "trap" not in {name for _, name in first["m.py"].scopes}   # content lines are not hunks
    assert "fake.py" not in first                                     # nor file headers
    assert first["new.py"].created is True and first["new.py"].scopes == set()
    assert first["m.py"].created is False
    assert "gone.py" not in first and "dead" not in {n for t in first.values() for _, n in t.scopes}
    assert first["my file.py"].scopes == {("def", "spaced")}
    assert got["bbbb"]["m.py"].scopes == set()                        # hunk with no header text


# ================================================================== integration
def test_plugs_into_localizer_and_function_evidence_reaches_the_result(tmp_path):
    """entity.path / entity.name from the real parser must line up with git's output."""
    auth = (
        "def hash_password(raw):\n    return raw\n\n"
        "def login(user, raw):\n    return hash_password(raw) == user.password\n"
    )
    billing = "def charge(amount):\n    return amount * 2\n"
    repo = make_repo(tmp_path, [
        Spec({"auth.py": auth, "billing.py": billing}, "add modules", T0),
        Spec({"auth.py": auth.replace("== user.password", "== user.pw")}, "Fix login crash", T0 + DAY),
    ])

    graph = GraphBuilder().build(PythonParser().parse_repository(str(repo.path)))
    localizer = Localizer(
        graph,
        config=RankConfig().with_weights(history=0.3),
        providers=[HistoryProvider(str(repo.path))],
    )
    result = localizer.localize("login is broken")

    hits = {h.entity.qualified_name: h for h in result.scopes}
    history = [e for e in hits["login"].evidence if e.type == "history"]
    assert history and history[0].detail["level"] == "function"
    assert history[0].detail["function"] == "login"
    assert "login()" in history[0].source
    assert hits["login"].signals["history"] > 0

    if "hash_password" in hits:  # same file, but its code was not touched by the fix
        assert hits["login"].signals["history"] > hits["hash_password"].signals["history"]
    json.dumps(result.to_dict())  # evidence detail stays JSON-serialisable end to end
