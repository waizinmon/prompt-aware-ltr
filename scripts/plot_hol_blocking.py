"""
Plot HOL Blocking Indicators -- two separate figures, one script.

Reads hol_blocking_metrics.json (produced by scripts/6_hol_blocking_eval.py)
and saves two standalone PNGs:
  - hol_blocking_ratio.png : p90/mean latency ratio at high load, grouped
    bar chart by scheduler, one bar per dataset.
  - hol_blocking_slope.png : latency-vs-request-rate slope, same layout.

USAGE:
    python scripts/plot_hol_blocking.py

Run this from the repository root (so the relative hol_blocking_metrics.json
path resolves correctly).
"""

import json
import matplotlib.pyplot as plt
import numpy as np

SCHEDULER_LABELS = {
    "fcfs": "FCFS",
    "classification": "Classification",
    "ltr": "Original LTR\n(OPT-125M)",
    "ltr_promptlen": "LTR + Prompt\nLength (ours)",
}
SCHEDULER_ORDER = ["fcfs", "classification", "ltr", "ltr_promptlen"]

DATASET_LABELS = {
    "dataset1_eval": "Dataset 1 (in-distribution)",
    "dataset2_eval": "Dataset 2 (out-of-distribution)",
}
DATASET_COLORS = {
    "dataset1_eval": "#1E3A5F",
    "dataset2_eval": "#D97706",
}

INPUT_PATH = "hol_blocking_metrics.json"


def load_data(path):
    with open(path) as f:
        return json.load(f)


def plot_metric(hol_data, dataset_keys, schedulers, metric_key, value_fmt,
                ylabel, title, suptitle, output_path):
    x = np.arange(len(schedulers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for i, dkey in enumerate(dataset_keys):
        offset = (i - (len(dataset_keys) - 1) / 2) * width
        color = DATASET_COLORS.get(dkey, None)
        label = DATASET_LABELS.get(dkey, dkey)

        values = [hol_data[dkey][s][metric_key] for s in schedulers]
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, value_fmt.format(v),
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([SCHEDULER_LABELS.get(s, s) for s in schedulers], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    hol_data = load_data(INPUT_PATH)
    dataset_keys = [k for k in ["dataset1_eval", "dataset2_eval"] if k in hol_data]
    schedulers = [s for s in SCHEDULER_ORDER if all(s in hol_data[dk] for dk in dataset_keys)]

    plot_metric(
        hol_data, dataset_keys, schedulers,
        metric_key="mean_p90_to_mean_ratio_high_load",
        value_fmt="{:.3f}",
        ylabel="p90 / mean latency ratio",
        title="Tail Latency Ratio (high load)",
        suptitle="HOL Blocking Indicator: Tail Latency Ratio\n(lower = less Head-of-Line blocking)",
        output_path="hol_blocking_ratio.png",
    )

    plot_metric(
        hol_data, dataset_keys, schedulers,
        metric_key="latency_slope_per_req_rate",
        value_fmt="{:.5f}",
        ylabel="Latency slope (s/token per req/s)",
        title="Latency-vs-Request-Rate Slope",
        suptitle="HOL Blocking Indicator: Latency Slope\n(lower = less Head-of-Line blocking)",
        output_path="hol_blocking_slope.png",
    )


if __name__ == "__main__":
    main()
