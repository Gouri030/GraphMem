#!/usr/bin/env python3
"""Run GraphMem's ADR-18 SWE-bench Lite localization evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphmem.evaluation.datasets import load_swebench_lite, save_subset, select_subset
from graphmem.evaluation.report import build_report
from graphmem.evaluation.runner import run_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-dataset", action="store_true", help="download/select SWE-bench Lite subset")
    parser.add_argument("--pilot", action="store_true", help="run five instances from dev split")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--limit", type=int, help="limit number of instances (overrides --pilot)")
    parser.add_argument("--total", type=int, default=40)
    parser.add_argument("--dev-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--subset", default="data/eval_subset.json")
    parser.add_argument("--repos", default="data/repos")
    parser.add_argument("--output", default="results")
    parser.add_argument("--ranker", help="Path to a trained LightGBM model; evaluates it as an additional 'learned' system.")
    args = parser.parse_args()

    subset_path = Path(args.subset)
    if args.prepare_dataset or not subset_path.exists():
        dataset = load_swebench_lite()
        subset = select_subset((dict(row) for row in dataset), args.total, args.dev_size, args.seed)
        save_subset(subset, subset_path)
        print(f"Saved subset: {subset_path} ({len(subset['dev'])} dev, {len(subset['test'])} test)")
        if args.prepare_dataset:
            return 0

    limit = args.limit if args.limit is not None else (5 if args.pilot else None)
    summary = run_evaluation(
        subset_path=subset_path, output_dir=args.output, repos_dir=args.repos,
        split=args.split, limit=limit, ranker_path=args.ranker,
    )
    csv_path, md_path, chart_path = build_report(
        args.output, set(summary["completed_instance_ids"])
    )
    print(json.dumps({
        "completed": summary["completed"], "requested": summary["requested_instances"],
        "skipped": summary["skipped"], "csv": str(csv_path), "markdown": str(md_path),
        "chart": str(chart_path),
    }, indent=2))
    return 0 if summary["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
