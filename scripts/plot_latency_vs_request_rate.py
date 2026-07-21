"""
Plot Latency vs Request Rate -- Dataset 1 and Dataset 2 as separate figures.

Reads result/results_dataset1.json and result/results_dataset2.json
(produced by 3_vllm_benchmark.py) and saves two standalone PNGs --
latency_dataset1.png and latency_dataset2.png -- one per dataset,
each showing all four schedulers (FCFS, Classification, Original LTR,
LTR + Prompt Length) plotted as latency vs request rate.

USAGE:
    python scripts/plot_latency_vs_request_rate.py

Run this from the repository root (so the relative result/... paths
resolve correctly), after result/results_dataset1.json and
result/results_dataset2.json exist.
"""

import json
import matplotlib.pyplot as plt
from collections import defaultdict

SCHEDULER_LABELS = {
    "fcfs": "FCFS",
    "classification": "Classification",
    "ltr": "Original LTR (OPT-125M)",
    "ltr_promptlen": "LTR + Prompt Length (ours)",
}
SCHEDULER_COLORS = {
    "fcfs": "#DC4C3E",
    "classification": "#D97706",
    "ltr": "#1E3A5F",
    "ltr_promptlen": "#0D9488",
}
SCHEDULER_STYLES = {
    "fcfs": "-o",
    "classification": "-s",
    "ltr": "-^",
    "ltr_promptlen": "-D",
}
SCHEDULER_ORDER = ["fcfs", "classification", "ltr", "ltr_promptlen"]

DATASETS = {
    "dataset1_eval": {
        "input": "result/results_dataset1.json",
        "output": "result/latency_dataset1.png",
        "title": "Dataset 1 (in-distribution)",
    },
    "dataset2_eval": {
        "input": "result/results_dataset2.json",
        "output": "result/latency_dataset2.png",
        "title": "Dataset 2 (out-of-distribution)",
    },
}


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def organize_by_scheduler(results):
    by_scheduler = defaultdict(list)
    for r in results:
        by_scheduler[r["scheduler"]].append((r["request_rate"], r["mean_latency_s_per_token"]))
    for k in by_scheduler:
        by_scheduler[k].sort(key=lambda x: x[0])
    return by_scheduler


def main():
    for dkey, cfg in DATASETS.items():
        results = load_results(cfg["input"])
        by_scheduler = organize_by_scheduler(results)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        for s in SCHEDULER_ORDER:
            if s not in by_scheduler:
                continue
            points = by_scheduler[s]
            rates = [p[0] for p in points]
            latencies = [p[1] for p in points]
            ax.plot(rates, latencies, SCHEDULER_STYLES[s], label=SCHEDULER_LABELS[s],
                    color=SCHEDULER_COLORS[s], linewidth=2, markersize=6)

        ax.set_xlabel("Request rate (req/s)", fontsize=12)
        ax.set_ylabel("Latency (s/token)", fontsize=12)
        ax.set_title(f"Llama-3.1-8B-Instruct: Latency vs Request Rate\n{cfg['title']}",
                     fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(cfg["output"], dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {cfg['output']}")


if __name__ == "__main__":
    main()
