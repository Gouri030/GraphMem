"""End-to-end localization evaluation for ADR-18."""
from __future__ import annotations

import json
import time
from pathlib import Path
import traceback
from typing import Any

from graphmem.evaluation.datasets import prepare_repo
from graphmem.evaluation.gold import parse_patch, resolve_functions
from graphmem.evaluation.metrics import (
    candidate_coverage, hit_at_k, mean, reciprocal_rank, recall_at_k,
)
from datetime import datetime
from graphmem.artifacts.models import Artifact
from graphmem.graph.builder import GraphBuilder
from graphmem.ingestion.cache import load_or_parse
from graphmem.ingestion.scanner import RepositoryScanner
from graphmem.ranking.config import RankConfig, ablation_configs
from graphmem.retrieval.localizer import Localizer
from graphmem.retrieval.providers.history import HistoryProvider


def _artifact_provider(repo_path: Path, repo: str, base_commit: str):
    """Load cached repository artifacts and create an evidence provider."""
    try:
        from graphmem.retrieval.providers.artifact import ArtifactProvider
    except ImportError:
        return None

    artifact_path = (
        Path(__file__).resolve().parents[3]
        / "data"
        / "artifacts"
        / f"{repo.replace('/', '__')}.json"
    )

    if not artifact_path.exists():
        return None

    try:
        records = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifacts = [
            Artifact(
                kind=record["kind"],
                number=int(record["number"]),
                title=record.get("title", ""),
                body=record.get("body", ""),
                created_at=datetime.fromisoformat(
                    record["created_at"].replace("Z", "+00:00")
                ),
                files=record.get("files", []),
            )
            for record in records
        ]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Could not load artifacts from {artifact_path}: {exc}")
        return None

    return ArtifactProvider(artifacts)


def _ranked(result) -> dict[str, list]:
    files = [path for path, _ in result.files]
    functions: list[str] = []
    statement_hits: dict[tuple[str, int, int], float] = {}
    # Scope hits and statements are ranked separately; flatten statements by
    # their own scores so statement metrics reflect statement-level ranking.
    for hit in result.scopes:
        entity = hit.entity
        if entity.path and entity.type.value in {"function", "method"}:
            name = entity.qualified_name or entity.name
            value = f"{entity.path}::{name}"
            if value not in functions:
                functions.append(value)
        for statement in hit.statements:
            span = (entity.path or "", statement.start_line, statement.end_line)
            statement_hits[span] = max(statement_hits.get(span, float("-inf")), statement.score)
    statements = [span for span, _ in sorted(
        statement_hits.items(), key=lambda item: (-item[1], item[0])
    )]
    return {"files": files, "functions": functions, "statements": statements}


def _score_level(ranked: list, gold: set, level: str) -> dict[str, float]:
    if not gold:
        return {}
    return {
        "acc@1": hit_at_k(ranked, gold, 1),
        "acc@3": hit_at_k(ranked, gold, 3),
        "acc@5": hit_at_k(ranked, gold, 5),
        "recall@5": recall_at_k(ranked, gold, 5),
        "mrr": reciprocal_rank(ranked, gold),
        "coverage": candidate_coverage(ranked, gold),
    }


def _statement_gold(gold) -> set[tuple[str, int, int]]:
    # Use one-line gold spans so exact source lines can be compared directly.
    return {(path, line, line) for path, lines in gold.lines.items() for line in lines}


def _statement_scores(ranked: list[tuple[str, int, int]], gold) -> dict[str, float]:
    if not gold.lines or not any(gold.lines.values()):
        return {}
    def hit(k: int) -> float:
        for path, start, end in ranked[:k]:
            if any(start <= line <= end for line in gold.lines.get(path, set())):
                return 1.0
        return 0.0
    total_gold = sum(len(lines) for lines in gold.lines.values())
    found_top5 = sum(
        1 for path, lines in gold.lines.items()
        for line in lines
        if any(p == path and start <= line <= end for p, start, end in ranked[:5])
    )
    found_any = sum(
        1 for path, lines in gold.lines.items()
        for line in lines
        if any(p == path and start <= line <= end for p, start, end in ranked)
    )
    rr = 0.0
    for i, (path, start, end) in enumerate(ranked, 1):
        if any(start <= line <= end for line in gold.lines.get(path, set())):
            rr = 1.0 / i
            break
    return {
        "acc@1": hit(1), "acc@3": hit(3), "acc@5": hit(5),
        "recall@5": found_top5 / total_gold,
        "mrr": rr,
        "coverage": found_any / total_gold,
    }


def run_evaluation(
    *,
    subset_path: str | Path,
    output_dir: str | Path,
    repos_dir: str | Path,
    split: str = "dev",
    limit: int | None = None,
) -> dict[str, Any]:
    """Run all configured ablations for each eligible instance and save JSON."""
    subset = json.loads(Path(subset_path).read_text(encoding="utf-8"))
    instances = list(subset.get(split, []))
    if limit is not None:
        instances = instances[:limit]
    output_dir = Path(output_dir)
    instance_dir = output_dir / "instances"
    instance_dir.mkdir(parents=True, exist_ok=True)
    configs = ablation_configs()
    summary: dict[str, Any] = {
        "split": split, "requested_instances": len(instances), "completed": 0,
        "completed_instance_ids": [], "skipped": [], "config_results": {},
    }

    for instance in instances:
        instance_id = instance["instance_id"]
        started = time.perf_counter()
        record: dict[str, Any] = {
            "instance_id": instance_id, "repo": instance["repo"],
            "base_commit": instance["base_commit"], "ranked": {}, "metrics": {},
        }
        try:
            repo_path = prepare_repo(instance["repo"], instance["base_commit"], repos_dir)
            scanner = RepositoryScanner(str(repo_path))
            scanner.validate()
            parser = scanner.get_parser()
            # Tests contain real fault locations in SWE-bench patches and must
            # be present in the indexed repository for fair localization.
            parser.excluded_dirs.discard("tests")
            parsed = load_or_parse(scanner.repository_path, parser)
            gold = resolve_functions(parse_patch(instance["patch"]), parsed)
            if not gold.files or (not gold.lines and gold.files <= gold.new_files):
                summary["skipped"].append({"instance_id": instance_id, "reason": "empty gold or only new files"})
                continue
            graph = GraphBuilder().build(parsed)
            providers = [HistoryProvider(str(repo_path), as_of=instance["base_commit"])]
            artifact_provider = _artifact_provider(
                repo_path,
                instance["repo"],
                instance["base_commit"],
            )
            if artifact_provider is not None:
                providers.append(artifact_provider)
            localizer = Localizer(graph, providers=providers)

            gold_by_level = {
                "file": gold.files - gold.new_files,
                "function": gold.functions,
                "statement": _statement_gold(gold),
            }
            # A config requiring unavailable artifact evidence is still run, but
            # this is recorded in metadata and should not be interpreted as an active artifact ablation.
            record["artifact_provider_available"] = artifact_provider is not None
            for config_name, config in configs.items():
                t0 = time.perf_counter()
                result = localizer.localize(instance["problem_statement"], top_k=100, config=config)
                ranked = _ranked(result)
                record["ranked"][config_name] = {
                    "files": ranked["files"],
                    "functions": ranked["functions"],
                    "statements": [list(span) for span in ranked["statements"]],
                }
                metrics = {
                    "file": _score_level(ranked["files"], gold_by_level["file"], "file"),
                    "function": _score_level(ranked["functions"], gold_by_level["function"], "function"),
                    "statement": _statement_scores(ranked["statements"], gold),
                }
                record["metrics"][config_name] = metrics
                record.setdefault("timings_ms", {})[config_name] = {
                    **result.timings_ms,
                    "end_to_end": round((time.perf_counter() - t0) * 1000, 2),
                }
            record["gold"] = {
                "files": sorted(gold_by_level["file"]),
                "functions": sorted(gold_by_level["function"]),
                "lines": {path: sorted(lines) for path, lines in gold.lines.items()},
                "new_files": sorted(gold.new_files),
            }
            record["instance_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            (instance_dir / f"{instance_id}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            summary["completed"] += 1
            summary["completed_instance_ids"].append(instance_id)
            for config_name, levels in record["metrics"].items():
                target = summary["config_results"].setdefault(config_name, {})
                for level, values in levels.items():
                    if not values:
                        continue
                    bucket = target.setdefault(level, {})
                    for metric, value in values.items():
                        bucket.setdefault(metric, []).append(value)
        except Exception as exc:
            print(f"\nFailed instance: {instance_id}")
            traceback.print_exc()
            summary["skipped"].append({
                "instance_id": instance_id,
                "reason": f"{type(exc).__name__}: {exc}",
            })

    # Keep the aggregate in memory for report.py and persist a machine-readable run summary.
    summary["config_results"] = {
        config: {
            level: {metric: mean(values) for metric, values in metrics.items()}
            for level, metrics in levels.items()
        }
        for config, levels in summary["config_results"].items()
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
