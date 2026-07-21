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

INPUT_PATH = "result/extended_metrics.json"

with open(INPUT_PATH) as f:
    ext_data = json.load(f)

dataset_keys = [k for k in ["dataset1_eval", "dataset2_eval"] if k in ext_data]
schedulers = [s for s in SCHEDULER_ORDER if all(s in ext_data[dk] for dk in dataset_keys)]

x = np.arange(len(schedulers))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))

for i, dkey in enumerate(dataset_keys):
    offset = (i - (len(dataset_keys) - 1) / 2) * width
    color = DATASET_COLORS.get(dkey, None)
    label = DATASET_LABELS.get(dkey, dkey)

    avg_tps = []
    for s in schedulers:
        tlist = ext_data[dkey][s]["throughput"]
        avg_tps.append(sum(t["tokens_per_second"] for t in tlist) / len(tlist))

    bars = ax.bar(x + offset, avg_tps, width, label=label, color=color)
    for b, v in zip(bars, avg_tps):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([SCHEDULER_LABELS.get(s, s) for s in schedulers], fontsize=10)
ax.set_ylabel("Mean throughput (tokens/s)", fontsize=11)
ax.set_title("Mean Throughput by Scheduler (averaged across request rates)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig("result/throughput_by_scheduler.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved throughput_by_scheduler.png")