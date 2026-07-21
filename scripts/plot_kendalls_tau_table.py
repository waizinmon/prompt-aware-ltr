"""
Render Kendall's Tau results as a table figure.

Reads tau_results.json (produced by scripts/2_kendalls_tau_eval.py) and
saves kendalls_tau_table.png -- a matplotlib table comparing Original LTR
vs LTR + Prompt Length. Also prints a plain-text/markdown version of the
same table to the console for pasting into a LaTeX table if preferred.

USAGE:
    python scripts/plot_kendalls_tau_table.py

Run this from the repository root (so the relative tau_results.json
path resolves correctly).
"""

import json
import matplotlib.pyplot as plt

INPUT_PATH = "tau_results.json"
OUTPUT_PATH = "kendalls_tau_table.png"


def load_data(path):
    with open(path) as f:
        return json.load(f)


def main():
    data = load_data(INPUT_PATH)

    tau_original = data["tau_original"]
    tau_extended = data["tau_extended"]
    delta = data["delta"]
    n = data["n_test_records"]
    alpha = data["alpha"]
    beta = data["beta"]

    rows = [
        ["Original LTR", f"{tau_original:.4f}"],
        ["LTR + Prompt Length (ours)", f"{tau_extended:.4f}"],
        ["Delta", f"{delta:+.4f}"],
    ]
    col_labels = ["Scheduler / Score", "Kendall's τ"]

    # --- console table ---
    print(f"Kendall's Tau -- ranking quality (n={n} test records, "
          f"α={alpha}, β={beta})\n")
    header = f"{col_labels[0]:<30}{col_labels[1]}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r[0]:<30}{r[1]}")

    # --- table figure ---
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # bold header row
    for j in range(len(col_labels)):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#1E3A5F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # highlight the "ours" row
    ours_row_idx = 2  # row 1 in `rows` -> table row index 2 (1-indexed after header)
    for j in range(len(col_labels)):
        table[ours_row_idx, j].set_facecolor("#E6F4F1")

    ax.set_title(
        f"Kendall's Tau (ranking quality vs. true output length)\n"
        f"n={n} test records, α={alpha}, β={beta}",
        fontsize=12, fontweight="bold", pad=14,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
