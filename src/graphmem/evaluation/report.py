"""Aggregate per-instance evaluation JSON into CSV, Markdown, and charts."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

METRICS = ("acc@1", "acc@3", "acc@5", "recall@5", "mrr", "coverage")


def build_report(
    output_dir: str | Path,
    instance_ids: set[str] | list[str] | None = None,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    included_ids = set(instance_ids) if instance_ids is not None else None
    rows: list[dict] = []
    for path in sorted((output_dir / "instances").glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if included_ids is not None and item.get("instance_id", path.stem) not in included_ids:
            continue
        for config, levels in item.get("metrics", {}).items():
            for level, values in levels.items():
                if not values:
                    continue
                rows.append({
                    "config": config, "level": level, "n": 1,
                    **{metric: values.get(metric, 0.0) for metric in METRICS},
                    "instance_id": item.get("instance_id", path.stem),
                    "total_ms": item.get("timings_ms", {}).get(config, {}).get("end_to_end", 0.0),
                })

    # CSV is aggregated by config and localization level, as requested.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["config"], row["level"])].append(row)
    aggregate = []
    for (config, level), items in sorted(groups.items()):
        aggregate.append({
            "config": config, "level": level, "n": len(items),
            **{metric: mean(row[metric] for row in items) for metric in METRICS},
            "mean_time_ms": mean(row["total_ms"] for row in items),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablation.csv"
    fields = ["config", "level", "n", *METRICS, "mean_time_ms"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate)

    md_path = output_dir / "ablation.md"
    lines = [
        "# GraphMem ablation results", "",
        "Metrics are macro-averaged over eligible instances. `n` is the number of instances contributing to that level/configuration.", "",
        "| Config | Level | n | Acc@1 | Acc@3 | Acc@5 | Recall@5 | MRR | Coverage | Mean time (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['config']} | {row['level']} | {row['n']} | "
            + " | ".join(f"{row[m]:.3f}" for m in METRICS)
            + f" | {row['mean_time_ms']:.1f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    chart_path = output_dir / "ablation_acc.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        configs = list(dict.fromkeys(row["config"] for row in aggregate))
        levels = ("file", "function", "statement")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for axis, level in zip(axes, levels):
            data = {(r["config"], r["level"]): r for r in aggregate}
            xs = list(range(len(configs)))
            width = 0.36
            acc1 = [data.get((c, level), {}).get("acc@1", 0.0) for c in configs]
            acc5 = [data.get((c, level), {}).get("acc@5", 0.0) for c in configs]
            axis.bar([x - width / 2 for x in xs], acc1, width, label="Acc@1")
            axis.bar([x + width / 2 for x in xs], acc5, width, label="Acc@5")
            axis.set_title(level.capitalize())
            axis.set_xticks(xs, configs, rotation=45, ha="right")
            axis.set_ylim(0, 1)
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("Accuracy")
        axes[-1].legend()
        fig.tight_layout()
        fig.savefig(chart_path, dpi=160)
        plt.close(fig)
    except ImportError:
        chart_path.write_text("matplotlib is required to render this chart.\n", encoding="utf-8")
    return csv_path, md_path, chart_path
