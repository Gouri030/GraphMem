"""SWE-bench Lite subset selection and repository checkout helpers (ADR-18)."""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable

PREFERRED_REPOS = (
    "pallets/flask",
    "psf/requests",
    "mwaskom/seaborn",
    "pylint-dev/pylint",
    "pytest-dev/pytest",
)


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(command)}\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def load_swebench_lite():
    """Load the canonical SWE-bench Lite dataset using Hugging Face datasets."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies first (pip install -e .); datasets is required.") from exc
    # Current canonical dataset configuration is `default`; try the explicit
    # split first, then support dataset versions that expose only a default config.
    try:
        return load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    except (ValueError, RuntimeError):
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite")
        return dataset["test"] if "test" in dataset else dataset[next(iter(dataset.keys()))]


def select_subset(records: Iterable[dict[str, Any]], total: int = 40, dev_size: int = 15, seed: int = 18) -> dict:
    """Select a deterministic, repo-balanced subset and split it into dev/test."""
    if total < 1 or not 0 <= dev_size <= total:
        raise ValueError("Require total >= 1 and 0 <= dev_size <= total")
    records = list(records)
    preferred = {name: i for i, name in enumerate(PREFERRED_REPOS)}
    eligible = [
        row for row in records
        if all(row.get(key) for key in ("instance_id", "repo", "base_commit", "problem_statement", "patch"))
    ]
    rng = random.Random(seed)
    # Shuffle each repository's candidates independently, then round-robin the
    # preferred repositories before falling back to the remaining repositories.
    by_repo: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        by_repo.setdefault(str(row["repo"]), []).append(row)
    for rows in by_repo.values():
        rng.shuffle(rows)
    repo_order = sorted(
        by_repo,
        key=lambda repo: (PREFERRED_REPOS.index(repo) if repo in PREFERRED_REPOS else len(PREFERRED_REPOS), repo),
    )
    chosen: list[dict[str, Any]] = []
    while len(chosen) < total:
        made_progress = False
        for repo in repo_order:
            if by_repo[repo]:
                chosen.append(by_repo[repo].pop())
                made_progress = True
                if len(chosen) == total:
                    break
        if not made_progress:
            break
    rng.shuffle(chosen)

    def compact(row: dict[str, Any]) -> dict[str, str]:
        return {key: str(row[key]) for key in ("instance_id", "repo", "base_commit", "problem_statement", "patch")}
    selected = [compact(row) for row in chosen]
    return {
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "seed": seed,
        "requested_total": total,
        "dev_size": min(dev_size, len(selected)),
        "test_size": max(0, len(selected) - dev_size),
        "dev": selected[:dev_size],
        "test": selected[dev_size:],
    }


def save_subset(subset: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(subset, indent=2) + "\n", encoding="utf-8")
    return path


def prepare_repo(repo: str, base_commit: str, repos_dir: str | Path) -> Path:
    """Clone each repository once, then force-checkout the requested snapshot."""
    repos_dir = Path(repos_dir)
    repos_dir.mkdir(parents=True, exist_ok=True)
    safe_name = repo.replace("/", "__")
    path = repos_dir / safe_name
    if not (path / ".git").exists():
        if path.exists():
            import shutil
            shutil.rmtree(path)
        _run(["git", "clone", "--filter=blob:none", "--no-checkout", f"https://github.com/{repo}.git", str(path)])
    _run(["git", "fetch", "--quiet", "origin", base_commit], cwd=path)
    _run(["git", "checkout", "--quiet", "-f", base_commit], cwd=path)
    _run(["git", "clean", "-fdx"], cwd=path)
    return path
