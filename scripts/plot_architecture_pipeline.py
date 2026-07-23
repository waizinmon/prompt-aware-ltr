"""
Background Figure 1: LLM inference serving pipeline / scheduler architecture.

A matplotlib-drawn architecture diagram (boxes + arrows, not a screenshot)
showing where the scheduler sits in the vLLM offline serving pipeline:
incoming requests are ranked by a scheduling policy before being handed to
vLLM's own continuous-batching engine on the GPU.

USAGE:
    python scripts/plot_architecture_pipeline.py

Saves latex_source/figures/architecture_pipeline.png (run from repo root).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT_PATH = "latex_source/figures/architecture_pipeline.png"


def box(ax, xy, w, h, text, facecolor="#E8EEF5", edgecolor="#1E3A5F", fontsize=10.5):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6, edgecolor=edgecolor, facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, wrap=True)
    return (x, y, w, h)


def arrow(ax, start, end, color="#1E3A5F"):
    a = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=18,
        linewidth=1.6, color=color, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    # Row 1: incoming requests
    b_requests = box(ax, (0.3, 4.4), 2.2, 0.9, "Incoming\nrequests\n(prompts)")

    # Row 2: scheduler policy (the focus of this work)
    b_sched = box(ax, (0.3, 2.9), 4.5, 1.1,
                  "Scheduling policy\n(FCFS / Classification / LTR /\nLTR + Prompt Length)",
                  facecolor="#DCEFEA", edgecolor="#0D9488", fontsize=10)

    # Components feeding the scheduler
    b_ltr = box(ax, (5.2, 3.55), 2.1, 0.7, "OPT-125M\nLTR predictor", fontsize=9.5)
    b_len = box(ax, (5.2, 2.65), 2.1, 0.7, "Prompt length\n(token count)", fontsize=9.5)

    # Row 3: request queue re-ordered by composite score
    b_queue = box(ax, (0.3, 1.5), 4.5, 0.9, "Re-ordered request queue\n(by composite score)")

    # Row 4: vLLM engine
    b_vllm = box(ax, (0.3, 0.2), 4.5, 1.0,
                 "vLLM continuous-batching engine\n(PagedAttention, GPU)",
                 facecolor="#FBEBD9", edgecolor="#D97706")

    # Arrows
    arrow(ax, (1.4, 4.4), (1.4, 4.0))                     # requests -> scheduler
    arrow(ax, (5.2, 3.9), (4.8, 3.6))                     # LTR predictor -> scheduler
    arrow(ax, (5.2, 3.0), (4.8, 3.2))                     # prompt length -> scheduler
    arrow(ax, (2.55, 2.9), (2.55, 2.4))                   # scheduler -> queue
    arrow(ax, (2.55, 1.5), (2.55, 1.2))                   # queue -> vLLM

    ax.set_title("LLM Serving Pipeline with a Pluggable Scheduling Policy",
                  fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
