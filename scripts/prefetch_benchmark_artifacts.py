#!/usr/bin/env python3
"""Prefetch GitHub artifacts that predate SWE-bench base commits.

This script:
1. Loads a sample of princeton-nlp/SWE-bench train instances.
2. Resolves each base commit timestamp from the local repo checkout.
3. Calls scripts/fetch_artifacts.py once per repository using that repository's
   earliest selected cutoff. The fetcher must support --before ISO_TIMESTAMP
   and filter both issue creation and PR creation/merge dates accordingly.

Important: the fetcher currently shown in the conversation does NOT support
--before. Update it first; this script intentionally fails rather than silently
fetching today's newest artifacts.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_DIR = ROOT / ".cache" / "swebench-repos"


def commit_timestamp(repo_path: Path, commit: str) -> datetime:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "show", "-s", "--format=%cI", commit],
        check=True,
        capture_output=True,
        text=True,
    )
    value = datetime.fromisoformat(result.stdout.strip().replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-instances", type=int, default=100)
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "dev"),
        help="SWE-bench split to sample (default: train).",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Missing dependency 'datasets'. Install project dependencies first.", file=sys.stderr)
        return 2

    fetcher = ROOT / "scripts" / "fetch_artifacts.py"
    if not fetcher.exists():
        print(f"Fetcher not found: {fetcher}", file=sys.stderr)
        return 2

    # The existing fetcher provided in the conversation accepts only owner/repo.
    # Refuse to run until --before is implemented, to avoid fetching the wrong era.
    source = fetcher.read_text(encoding="utf-8")
    if "--before" not in source or "merged_at" not in source:
        print(
            "ERROR: scripts/fetch_artifacts.py has not been updated for historical "
            "cutoffs. Add --before ISO_TIMESTAMP support and save/filter merged_at "
            "for PRs first. No fetches were started.",
            file=sys.stderr,
        )
        return 2

    print("Loading princeton-nlp/SWE-bench train split...")
    dataset = load_dataset("princeton-nlp/SWE-bench", split=args.split)
    instances = list(dataset)
    if args.max_instances > 0:
        import random
        random.Random(args.seed).shuffle(instances)
        instances = instances[:args.max_instances]

    cutoffs: dict[str, list[datetime]] = defaultdict(list)
    missing: list[str] = []

    for index, instance in enumerate(instances, 1):
        repo = instance["repo"]
        base_commit = instance["base_commit"]
        repo_path = args.repo_dir / repo.replace("/", "__")

        if not repo_path.exists():
            missing.append(f"{repo}@{base_commit[:12]}: repo checkout missing at {repo_path}")
            continue

        try:
            cutoff = commit_timestamp(repo_path, base_commit)
        except (subprocess.CalledProcessError, ValueError) as exc:
            missing.append(f"{repo}@{base_commit[:12]}: cannot resolve commit date ({exc})")
            continue

        cutoffs[repo].append(cutoff)
        print(f"[{index}/{len(instances)}] {repo} {base_commit[:12]} cutoff={cutoff.isoformat()}")

    if missing:
        print(f"\nWARNING: {len(missing)} instances could not be resolved:")
        for item in missing[:20]:
            print(f"  - {item}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    if not cutoffs:
        print("No repositories had resolvable local base commits; nothing fetched.", file=sys.stderr)
        return 1

    failures = 0
    for repo, dates in sorted(cutoffs.items()):
        # Fetch a common historical pool as of the earliest selected base commit.
        # Per-instance ArtifactProvider filtering then applies later cutoffs safely.
        cutoff = min(dates)
        owner, name = repo.split("/", 1)
        cutoff_arg = cutoff.isoformat().replace("+00:00", "Z")
        command = [
            sys.executable,
            str(fetcher),
            owner,
            name,
            "--before",
            cutoff_arg,
        ]
        print(f"\nFetching {repo}: up to 200 issues + 200 merged PRs before {cutoff_arg}")
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            failures += 1
            print(f"ERROR: fetch failed for {repo} (exit {exc.returncode})", file=sys.stderr)

    print(f"\nFinished. Repositories: {len(cutoffs)}; fetch failures: {failures}.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
