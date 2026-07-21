import json
import matplotlib.pyplot as plt
import numpy as np

DATASET_LABELS = {
    "dataset1_eval": "Dataset 1 (in-distribution)",
    "dataset2_eval": "Dataset 2 (out-of-distribution)",
}

INPUT_PATH = "result/extended_metrics.json"

with open(INPUT_PATH) as f:
    ext_data = json.load(f)

gpu_data = ext_data.get("gpu_utilization", {})
dataset_keys = [k for k in ["dataset1_eval", "dataset2_eval"] if k in gpu_data and "peak_allocated_mb" in gpu_data.get(k, {})]

if not dataset_keys:
    print("[INFO] No GPU profiling data found in extended_metrics.json "
          "(re-run scripts/5_extended_metrics.py with --run_gpu_profiling).")
else:
    labels = [DATASET_LABELS.get(k, k) for k in dataset_keys]
    peak_allocated = [gpu_data[k]["peak_allocated_mb"] for k in dataset_keys]
    peak_reserved = [gpu_data[k]["peak_reserved_mb"] for k in dataset_keys]
    utilization_pct = [gpu_data[k]["memory_utilization_fraction"] * 100 for k in dataset_keys]

    x = np.arange(len(dataset_keys))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax2 = ax1.twinx()

    # All three as bars, grouped side by side -- MB bars on the left axis,
    # utilization % bar on the right axis, sharing the same x positions.
    b1 = ax1.bar(x - width, peak_allocated, width, label="Peak Allocated (MB)", color="#1E3A5F")
    b2 = ax1.bar(x, peak_reserved, width, label="Peak Reserved (MB)", color="#0D9488")
    b3 = ax2.bar(x + width, utilization_pct, width, label="Memory Utilization (%)", color="#D97706")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("GPU Memory (MB)", fontsize=11)
    ax2.set_ylabel("Memory Utilization (%)", fontsize=11)
    ax2.set_ylim(0, 100)

    for b, v in zip(b1, peak_allocated):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    for b, v in zip(b2, peak_reserved):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    for b, v in zip(b3, utilization_pct):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    # Combine legends from both axes into one
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title("GPU Resource Utilization by Dataset", fontsize=13, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("result/gpu_utilization.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved gpu_utilization.png")