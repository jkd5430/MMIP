import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
def available_runs():
    return [
        path.name
        for path in sorted(RESULTS_ROOT.iterdir())
        if (path / "test_probabilities.npy").exists()
        and (path / "test_labels.npy").exists()
        and (path / "metrics.json").exists()
    ]



def analyze_run(run_name):
    run_dir = RESULTS_ROOT / run_name
    probabilities = np.load(run_dir / "test_probabilities.npy")
    labels = np.load(run_dir / "test_labels.npy").reshape(-1)
    class_names = json.loads((run_dir / "classes.json").read_text(encoding="utf-8"))
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    ranking = np.argsort(-probabilities, axis=1)
    ranks = np.argmax(ranking == labels[:, None], axis=1) + 1
    top1 = float(np.mean(ranks <= 1))
    top3 = float(np.mean(ranks <= 3))
    top5 = float(np.mean(ranks <= 5))

    per_class = []
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        per_class.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "samples": int(mask.sum()),
                "top1": float(np.mean(ranks[mask] <= 1)),
                "top3": float(np.mean(ranks[mask] <= 3)),
                "top5": float(np.mean(ranks[mask] <= 5)),
            }
        )

    with (run_dir / "per_class_top3.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_class[0].keys())
        writer.writeheader()
        writer.writerows(per_class)

    metrics["test_top3"] = top3
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    worst = sorted(per_class, key=lambda row: row["top3"])[:3]
    return {
        "run_name": run_name,
        "model": metrics["model"],
        "augmentation": metrics["augmentation"],
        "test_top1": top1,
        "test_top3": top3,
        "test_top5": top5,
        "rank2_only": float(np.mean(ranks == 2)),
        "rank3_only": float(np.mean(ranks == 3)),
        "top3_correct": int(np.sum(ranks <= 3)),
        "top3_missed": int(np.sum(ranks > 3)),
        "worst_top3_classes": "; ".join(
            f'{row["class_name"]}={row["top3"]:.4f}' for row in worst
        ),
    }


def update_comparison(rows):
    path = RESULTS_ROOT / "comparison.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    top3_by_run = {row["run_name"]: row["test_top3"] for row in rows}
    fields = list(old_rows[0].keys())
    if "test_top3" not in fields:
        fields.insert(fields.index("test_top5"), "test_top3")
    for row in old_rows:
        row["test_top3"] = top3_by_run[row["run_name"]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(old_rows)


def save_summary(rows):
    csv_path = RESULTS_ROOT / "top3_analysis.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: row["test_top3"])
    lines = [
        "# Test Top-3 analysis",
        "",
        "| Run | Top-1 | Top-3 | Top-5 | Top-3 correct | Top-3 missed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f'| {row["run_name"]} | {row["test_top1"]:.2%} | '
            f'{row["test_top3"]:.2%} | {row["test_top5"]:.2%} | '
            f'{row["top3_correct"]:,} | {row["top3_missed"]:,} |'
        )
    lines += [
        "",
        f'- Best Top-3: {best["run_name"]} ({best["test_top3"]:.2%}).',
        "",
        "## Lowest per-class Top-3",
        "",
    ]
    for row in rows:
        lines.append(f'- {row["run_name"]}: {row["worst_top3_classes"]}')
    (RESULTS_ROOT / "top3_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = [analyze_run(run_name) for run_name in available_runs()]
    update_comparison(rows)
    save_summary(rows)
    for row in rows:
        print(
            f'{row["run_name"]}: Top-1={row["test_top1"]:.4f}, '
            f'Top-3={row["test_top3"]:.4f}, Top-5={row["test_top5"]:.4f}'
        )
    print("Saved:", RESULTS_ROOT / "top3_analysis.csv")
    print("Saved:", RESULTS_ROOT / "top3_analysis.md")


if __name__ == "__main__":
    main()
