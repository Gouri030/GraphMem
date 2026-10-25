#!/usr/bin/env python3
"""Train a LambdaMART scope ranker from SWE-bench training instances.

Example (start small to validate the pipeline):
  python scripts/train_ranker.py --max-instances 30 --repo-dir .cache/swebench-repos

This command checks out each issue's base_commit and runs GraphMem retrieval.
It can take a long time because repositories are cloned and parsed locally.
Only the canonical SWE-bench *train* split is used; SWE-bench Lite test examples
are not used for fitting or hyperparameter selection.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.artifacts.models import Artifact
from graphmem.evaluation.datasets import prepare_repo
from graphmem.evaluation.gold import parse_patch, resolve_functions
from graphmem.graph.builder import GraphBuilder
from graphmem.ingestion.cache import load_or_parse
from graphmem.ingestion.scanner import RepositoryScanner
from graphmem.ranking.config import RankConfig, learned_rank_config
from graphmem.ranking.learned import FEATURE_NAMES, scope_features
from graphmem.retrieval.localizer import Localizer
from graphmem.retrieval.providers.history import HistoryProvider
from graphmem.retrieval.providers.artifact import ArtifactProvider


def make_rank_config() -> RankConfig:
    # Must match learned inference: provider activation affects feature values.
    return learned_rank_config(name="learned-training-candidates")



_ARTIFACT_FETCH_ATTEMPTED: set[str] = set()


def make_artifact_provider(
    repo_path: Path,
    repo: str,
    base_commit: str,
):
    """Fetch missing repository artifacts, then apply the historical cutoff."""
    artifact_path = (
        ROOT / "data" / "artifacts" / f"{repo.replace('/', '__')}.json"
    )

    if not artifact_path.exists() and repo not in _ARTIFACT_FETCH_ATTEMPTED:
        _ARTIFACT_FETCH_ATTEMPTED.add(repo)
        fetch_script = ROOT / "scripts" / "fetch_artifacts.py"

        print(f"Artifact cache missing for {repo}; fetching now...")

        try:
            proc = subprocess.run(
                [sys.executable, str(fetch_script), *repo.split("/", 1)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                # timeout=300,
                check=False,
            )

            if proc.stdout.strip():
                print(proc.stdout.strip())

            if proc.returncode != 0:
                print(
                    f"WARNING: artifact fetch failed for {repo}: "
                    f"{proc.stderr.strip()}"
                )

        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"WARNING: artifact fetch failed for {repo}: {exc}")

    if not artifact_path.exists():
        print(
            f"WARNING: no artifact cache for {repo}; "
            "artifact features will be zero"
        )
        return None

    try:
        records = json.loads(
            artifact_path.read_text(encoding="utf-8")
        )

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

        proc = subprocess.run(
            [
                "git", "-C", str(repo_path),
                "show", "-s", "--format=%cI", base_commit,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        cutoff = datetime.fromisoformat(
            proc.stdout.strip().replace("Z", "+00:00")
        )

        provider = ArtifactProvider(artifacts, as_of=cutoff)

        print(
            f"Artifact provider for {repo}: "
            f"{len(provider.artifacts)} artifacts available "
            f"by {base_commit[:12]}"
        )

        return provider

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(
            f"WARNING: could not load artifacts for {repo}: {exc}; "
            "artifact features will be zero"
        )
        return None


def extract_group(instance: dict, repo_dir: Path, top_k: int) -> tuple[list[list[float]], list[int]] | None:
    repo_path = prepare_repo(instance["repo"], instance["base_commit"], repo_dir)
    scanner = RepositoryScanner(str(repo_path))
    scanner.validate()
    parser = scanner.get_parser()
    parser.excluded_dirs.discard("tests")
    parsed = load_or_parse(scanner.repository_path, parser)
    gold = resolve_functions(parse_patch(instance["patch"]), parsed)
    if not gold.files or not gold.lines:
        return None

    graph = GraphBuilder().build(parsed)
    providers = [HistoryProvider(str(repo_path), as_of=instance["base_commit"])]
    artifact_provider = make_artifact_provider(repo_path, instance["repo"], instance["base_commit"])
    if artifact_provider is not None:
        providers.append(artifact_provider)
    localizer = Localizer(graph, config=make_rank_config(), providers=providers)
    result = localizer.localize(instance["problem_statement"], top_k=top_k, config=make_rank_config())
    if not result.scopes:
        return None

    rows: list[list[float]] = []
    labels: list[int] = []
    for hit in result.scopes:
        entity = hit.entity
        path = entity.path or ""
        start, end = entity.start_line or 0, entity.end_line or 0
        changed_lines = gold.lines.get(path, set())
        overlaps = bool(changed_lines) and any(start <= line <= end for line in changed_lines)
        exact_function = f"{path}::{entity.qualified_name or entity.name}" in gold.functions
        # Graded relevance: exact changed callable / changed-line overlap = 2;
        # a file-level candidate for a changed file = 1; other candidates = 0.
        if exact_function or overlaps:
            label = 2
        elif path in gold.files and getattr(entity.type, "value", "") == "file":
            label = 1
        else:
            label = 0
        rows.append(scope_features(hit, instance["problem_statement"]))
        labels.append(label)
    return rows, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-instances", type=int, default=200,
                        help="Maximum instances sampled from the SWE-bench train split (default: 200).")
    parser.add_argument("--top-k", type=int, default=100, help="Candidate scopes per issue (default: 100).")
    parser.add_argument("--repo-dir", type=Path, default=Path(".cache/swebench-repos"))
    parser.add_argument("--model-out", type=Path, default=Path("models/graphmem-lambdamart.txt"))
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--early-stopping-rounds", type=int, default=30,
                        help="Validation rounds without improvement before stopping (default: 30).")
    args = parser.parse_args()
    if args.max_instances < 10:
        parser.error("--max-instances must be at least 10 so train/validation can be separated.")

    try:
        from datasets import load_dataset
        import lightgbm as lgb
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Run `pip install -e .` first.", file=sys.stderr)
        return 2

    print("Loading princeton-nlp/SWE-bench train split (not Lite/Verified test data)...")
    dataset = load_dataset("princeton-nlp/SWE-bench", split="train")
    rows = list(dataset)
    random.Random(args.seed).shuffle(rows)
    rows = rows[:min(args.max_instances, len(rows))]
    # Split by issue before extracting features, so no issue appears in both sets.
    cut = max(1, int(len(rows) * 0.8))
    train_instances, valid_instances = rows[:cut], rows[cut:]

    def build(instances: list[dict], label: str):
        x_groups, y_groups, group_sizes = [], [], []
        for index, instance in enumerate(instances, 1):
            try:
                group = extract_group(instance, args.repo_dir, args.top_k)
                if group is None or not group[0] or not any(group[1]):
                    print(f"[{label} {index}/{len(instances)}] skipped {instance.get('instance_id')} (no usable candidates/positive label)")
                    continue
                x, y = group
                x_groups.extend(x)
                y_groups.extend(y)
                group_sizes.append(len(y))
                print(f"[{label} {index}/{len(instances)}] {instance['instance_id']}: {len(y)} candidates")
            except Exception as exc:
                print(f"[{label} {index}/{len(instances)}] failed {instance.get('instance_id')}: {exc}")
                traceback.print_exc()
        return x_groups, y_groups, group_sizes

    train_x, train_y, train_groups = build(train_instances, "train")
    valid_x, valid_y, valid_groups = build(valid_instances, "valid")
    if not train_groups or not valid_groups:
        print("Not enough successfully processed issue groups. Increase --max-instances or resolve repository/dependency errors.", file=sys.stderr)
        return 1

    # LightGBM expects array-like numeric matrices. Convert the nested Python
    # lists explicitly; some LightGBM versions reject a plain list-of-lists.
    import numpy as np

    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.int32)
    valid_x = np.asarray(valid_x, dtype=np.float32)
    valid_y = np.asarray(valid_y, dtype=np.int32)

    if train_x.ndim != 2 or train_x.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Invalid training feature matrix shape {train_x.shape}; "
            f"expected (n, {len(FEATURE_NAMES)})."
        )
    if valid_x.ndim != 2 or valid_x.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Invalid validation feature matrix shape {valid_x.shape}; "
            f"expected (n, {len(FEATURE_NAMES)})."
        )
    if int(sum(train_groups)) != len(train_y) or int(sum(valid_groups)) != len(valid_y):
        raise ValueError("Ranking group sizes must sum to the number of labels.")

    # A feature can only be learned if it varies across candidates.  In
    # particular, split importance of zero is expected when a small run has
    # too few issue groups or when baseline_score dominates the same signals.
    feature_stats = {}
    for index, name in enumerate(FEATURE_NAMES):
        values = train_x[:, index]
        feature_stats[name] = {
            "nonzero": int(np.count_nonzero(values)),
            "unique": int(np.unique(values).size),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    graph_stats = feature_stats["signal_graph"]
    print(
        "Training diagnostics: "
        f"{len(train_groups)} issue groups, {len(train_y)} candidates; "
        f"signal_graph nonzero={graph_stats['nonzero']} "
        f"unique={graph_stats['unique']} "
        f"range=[{graph_stats['min']:.4g}, {graph_stats['max']:.4g}]"
    )
    for signal in ("history", "artifact", "semantic"):
        stats = feature_stats[f"signal_{signal}"]
        print(f"signal_{signal}: nonzero={stats['nonzero']} unique={stats['unique']} range=[{stats['min']:.4g}, {stats['max']:.4g}]")
        if stats["unique"] < 2:
            print(f"WARNING: signal_{signal} is constant; the model cannot learn from it.")
    if len(train_groups) < 20 or len(valid_groups) < 5:
        print(
            "WARNING: this is a smoke-test-sized ranking dataset; feature "
            "importance is not reliable. Increase --max-instances."
        )
    if graph_stats["unique"] < 2:
        print(
            "WARNING: signal_graph is constant in the training candidates, "
            "so LightGBM cannot learn a graph split."
        )

    train_set = lgb.Dataset(
        train_x, label=train_y, group=train_groups,
        feature_name=list(FEATURE_NAMES),
    )
    valid_set = lgb.Dataset(
        valid_x, label=valid_y, group=valid_groups, reference=train_set,
        feature_name=list(FEATURE_NAMES),
    )
    model = lgb.train(
        {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 3, 5, 10],
         "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 10,
         "verbosity": -1, "label_gain": [0, 1, 3]},
        train_set,
        num_boost_round=args.rounds,
        valid_sets=[valid_set],
        valid_names=["validation"],
        callbacks=[lgb.early_stopping(args.early_stopping_rounds), lgb.log_evaluation(10)],
    )
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.model_out))
    metadata = {
        "model": "LightGBM LambdaMART", "dataset": "princeton-nlp/SWE-bench",
        "source_split": "train", "max_instances_requested": args.max_instances,
        "train_groups_processed": len(train_groups), "validation_groups_processed": len(valid_groups),
        "feature_names": list(FEATURE_NAMES), "label_definition": "2=changed-line/callable overlap; 1=file-level candidate for changed file; 0=other",
        "best_iteration": model.best_iteration,
        "feature_stats": feature_stats,
        "feature_importance_split": dict(zip(
            FEATURE_NAMES, model.feature_importance(importance_type="split").tolist()
        )),
        "feature_importance_gain": dict(zip(
            FEATURE_NAMES, model.feature_importance(importance_type="gain").tolist()
        )),
        "note": "Validation is a held-out subset sampled from SWE-bench train. Keep SWE-bench Lite/Verified test examples untouched for final evaluation.",
    }
    args.model_out.with_suffix(args.model_out.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model: {args.model_out}")
    print(f"Saved metadata: {args.model_out}.json")
    print(f"Processed train groups: {len(train_groups)}; validation groups: {len(valid_groups)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
