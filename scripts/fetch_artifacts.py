from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

from github import Auth, Github, GithubException, RateLimitExceededException

T = TypeVar("T")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "artifacts"
LIMIT = 200
MAX_RETRIES = 6


def retry_request(operation: Callable[[], T], github: Github) -> T:
    """Retry GitHub API operations after rate limits or transient failures."""
    for attempt in range(MAX_RETRIES):
        try:
            return operation()
        except RateLimitExceededException:
            try:
                reset = github.get_rate_limit().core.reset.timestamp()
                delay = max(1, reset - time.time() + 2)
            except Exception:
                delay = min(60, 2 ** (attempt + 1))
        except GithubException as exc:
            if exc.status not in {403, 429, 500, 502, 503, 504}:
                raise
            headers = getattr(exc, "headers", {}) or {}
            retry_after = headers.get("Retry-After")
            if retry_after is not None:
                try:
                    delay = max(1, int(retry_after))
                except (ValueError, TypeError):
                    delay = min(60, 2 ** (attempt + 1))
            elif exc.status in {403, 429}:
                try:
                    reset = github.get_rate_limit().core.reset.timestamp()
                    delay = max(1, reset - time.time() + 2)
                except Exception:
                    delay = min(60, 2 ** (attempt + 1))
            else:
                delay = min(60, 2 ** (attempt + 1))

        if attempt == MAX_RETRIES - 1:
            raise RuntimeError("GitHub request failed after repeated retries")
        print(f"GitHub request failed; retrying in {delay:.0f}s...")
        time.sleep(delay)

    raise RuntimeError("GitHub request failed")


def iso_datetime(value: datetime) -> str:
    """Serialize timestamps consistently in UTC."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def parse_cutoff(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp. Naive timestamps are rejected."""
    if value is None:
        return None
    cutoff = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if cutoff.tzinfo is None:
        raise ValueError(
            "--before must include a timezone, e.g. 2021-06-15T23:59:59Z"
        )
    return cutoff.astimezone(timezone.utc)


def is_before_cutoff(value: datetime | None, cutoff: datetime | None) -> bool:
    if cutoff is None:
        return True
    if value is None:
        return False
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc) <= cutoff


def load_cache(path: Path) -> tuple[list[dict], set[tuple[str, int]]]:
    """Load cached artifacts and index them by kind and number."""
    if not path.exists():
        return [], set()
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")
    existing = {(record["kind"], int(record["number"])) for record in records}
    return records, existing


def fetch_artifacts(owner: str, name: str, cutoff: datetime | None = None) -> Path:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN is missing. Export a GitHub personal access token."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{owner}__{name}.json"
    github = Github(auth=Auth.Token(token))

    try:
        repo = retry_request(lambda: github.get_repo(f"{owner}/{name}"), github)
        records, existing = load_cache(output_path)
        record_by_key = {(r["kind"], int(r["number"])): r for r in records}
        print(f"Cached artifacts: {len(existing)}")
        if cutoff:
            print(f"Historical cutoff (UTC): {cutoff.isoformat()}")
        else:
            print("No cutoff supplied; fetching the latest eligible records.")

        # GitHub includes pull requests in the issues endpoint. Because results
        # are newest-first, records newer than the cutoff are skipped, not counted.
        issues = retry_request(
            lambda: repo.get_issues(
                state="closed", sort="created", direction="desc"
            ),
            github,
        )
        issue_count = 0
        processed_issues = 0
        page = 0
        while processed_issues < LIMIT:
            batch = retry_request(lambda: issues.get_page(page), github)
            if not batch:
                break
            for item in batch:
                if item.pull_request is not None:
                    continue
                if not is_before_cutoff(item.created_at, cutoff):
                    continue

                processed_issues += 1
                key = ("issue", item.number)
                if key not in existing:
                    record = {
                        "kind": "issue",
                        "number": item.number,
                        "title": item.title or "",
                        "body": item.body or "",
                        "created_at": iso_datetime(item.created_at),
                        "files": [],
                    }
                    records.append(record)
                    record_by_key[key] = record
                    existing.add(key)
                    issue_count += 1
                if processed_issues >= LIMIT:
                    break
            page += 1

        print(
            f"New issues cached: {issue_count}; "
            f"eligible historical issues examined: {processed_issues}"
        )

        pulls = retry_request(
            lambda: repo.get_pulls(
                state="closed", sort="created", direction="desc"
            ),
            github,
        )
        pr_count = 0
        processed_prs = 0
        page = 0
        while processed_prs < LIMIT:
            batch = retry_request(lambda: pulls.get_page(page), github)
            if not batch:
                break
            for pr in batch:
                # Require creation AND merge to be at/before the cutoff.
                if not is_before_cutoff(pr.created_at, cutoff):
                    continue
                if pr.merged_at is None or not is_before_cutoff(pr.merged_at, cutoff):
                    continue

                processed_prs += 1
                key = ("pr", pr.number)
                if key in existing:
                    # Upgrade old cache records with merge timestamp. Don't
                    # redownload files just because the cache predates this field.
                    cached = record_by_key[key]
                    if pr.merged_at is not None:
                        cached["merged_at"] = iso_datetime(pr.merged_at)
                    if not cached.get("files"):
                        # Preserve cache behavior for existing records; file list
                        # can be empty legitimately and is not refetched here.
                        pass
                else:
                    files: list[str] = []
                    pr_files = retry_request(lambda: pr.get_files(), github)
                    file_page = 0
                    while True:
                        file_batch = retry_request(
                            lambda: pr_files.get_page(file_page), github
                        )
                        if not file_batch:
                            break
                        files.extend(item.filename for item in file_batch)
                        file_page += 1

                    record = {
                        "kind": "pr",
                        "number": pr.number,
                        "title": pr.title or "",
                        "body": pr.body or "",
                        "created_at": iso_datetime(pr.created_at),
                        "merged_at": iso_datetime(pr.merged_at),
                        "files": sorted(set(files)),
                    }
                    records.append(record)
                    record_by_key[key] = record
                    existing.add(key)
                    pr_count += 1

                if processed_prs >= LIMIT:
                    break
            page += 1

        print(
            f"New merged PRs cached: {pr_count}; "
            f"eligible historical merged PRs examined: {processed_prs}"
        )

        temp_path = output_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(output_path)
        print(f"Saved {len(records)} artifacts to {output_path}")
        return output_path
    finally:
        github.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch GitHub issues and merged PRs, optionally as of a historical cutoff."
    )
    parser.add_argument("owner")
    parser.add_argument("repo")
    parser.add_argument(
        "--before",
        help="Only include artifacts created (and PRs merged) by this ISO-8601 timestamp, e.g. 2021-06-15T23:59:59Z.",
    )
    args = parser.parse_args()

    try:
        cutoff = parse_cutoff(args.before)
        fetch_artifacts(args.owner, args.repo, cutoff)
    except (RuntimeError, GithubException, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
