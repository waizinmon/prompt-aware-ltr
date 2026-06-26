"""
EVALUATION PART 3: Plot Results (runs on Mac M4 -- no GPU required)
======================================================================

Takes the results.json produced by 3_vllm_benchmark.py (run on Colab)
and reproduces a chart in the same style as the original paper's
Figure 3 (latency vs. request rate), with your new scheduler added
as a fourth line.

USAGE:
    1. Download results.json from your Colab session.
    2. Run this script locally on your Mac:
         python 4_plot_results.py --input results.json --output comparison.png
"""

import argparse
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


def load_results(path: str):
    with open(path, "r") as f:
        return json.load(f)


def organize_by_scheduler(results):
    by_scheduler = defaultdict(list)
    for r in results:
        by_scheduler[r["scheduler"]].append((r["request_rate"], r["mean_latency_s_per_token"]))
    for k in by_scheduler:
        by_scheduler[k].sort(key=lambda x: x[0])
    return by_scheduler


def plot(by_scheduler, output_path: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for scheduler_name, points in by_scheduler.items():
        rates = [p[0] for p in points]
        latencies = [p[1] for p in points]
        label = SCHEDULER_LABELS.get(scheduler_name, scheduler_name)
        color = SCHEDULER_COLORS.get(scheduler_name, None)
        style = SCHEDULER_STYLES.get(scheduler_name, "-o")
        ax.plot(rates, latencies, style, label=label, color=color, linewidth=2, markersize=6)

    ax.set_xlabel("Request rate (req/s)", fontsize=12)
    ax.set_ylabel("Latency (s/token)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


def print_summary_table(by_scheduler):
    print("\nSummary table (mean latency, s/token):\n")
    all_rates = sorted({p[0] for points in by_scheduler.values() for p in points})
    header = "Request Rate".ljust(15) + "".join(
        SCHEDULER_LABELS.get(s, s).ljust(28) for s in by_scheduler
    )
    print(header)
    print("-" * len(header))
    for rate in all_rates:
        row = f"{rate:<15}"
        for s, points in by_scheduler.items():
            match = [v for r, v in points if r == rate]
            row += f"{match[0]:.4f}".ljust(28) if match else "—".ljust(28)
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results.json")
    parser.add_argument("--output", default="comparison.png")
    parser.add_argument("--title", default="Llama-3.1-8B Instruct: Latency vs Request Rate")
    parser.add_argument("--dataset_label", default="",
                        help="e.g. 'Dataset 1 (in-distribution)' or 'Dataset 2 (OOD)'")
    args = parser.parse_args()

    results = load_results(args.input)
    by_scheduler = organize_by_scheduler(results)

    title = args.title
    if args.dataset_label:
        title += f"\n{args.dataset_label}"

    print_summary_table(by_scheduler)
    plot(by_scheduler, args.output, title)


if __name__ == "__main__":
    main()
