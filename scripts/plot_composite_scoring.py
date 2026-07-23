"""
Background Figure 2: Composite scoring architecture.

A matplotlib-drawn architecture diagram (boxes + arrows, not a screenshot)
showing how the LTR predictor score and prompt length are combined into
the composite ranking score used by the LTR + Prompt Length scheduler:

    score(x) = alpha * s_LTR(x) - beta * (SCALE / len(x))

USAGE:
    python scripts/plot_composite_scoring.py

Saves latex_source/figures/composite_scoring_architecture.png (run from repo root).
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT_PATH = "latex_source/figures/composite_scoring_architecture.png"


def box(ax, xy, w, h, text, facecolor="#E8EEF5", edgecolor="#1E3A5F", fontsize=10):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6, edgecolor=edgecolor, facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return (x, y, w, h)


def arrow(ax, start, end, color="#1E3A5F"):
    a = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=18,
        linewidth=1.6, color=color, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    b_prompt = box(ax, (0.3, 3.9), 2.0, 0.9, "Input\nprompt $x$")

    b_predictor = box(ax, (3.2, 4.2), 2.6, 0.8, "OPT-125M\npredictor",
                       facecolor="#DCEFEA", edgecolor="#0D9488")
    b_tokenizer = box(ax, (3.2, 2.9), 2.6, 0.8, "Tokenizer\n(prompt length)",
                       facecolor="#FBEBD9", edgecolor="#D97706")

    b_score_ltr = box(ax, (6.5, 4.2), 2.9, 0.8, r"$s_{\mathrm{LTR}}(x)$")
    b_len_term = box(ax, (6.5, 2.9), 2.9, 0.8, r"$\mathrm{SCALE}\,/\,\mathrm{len}(x)$")

    b_combine = box(ax, (3.2, 1.3), 4.3, 1.0,
                     r"$\mathrm{score}(x)=\alpha\, s_{\mathrm{LTR}}(x) - \beta\,\dfrac{\mathrm{SCALE}}{\mathrm{len}(x)}$",
                     facecolor="#F4E9F7", edgecolor="#6B3FA0", fontsize=11)

    b_rank = box(ax, (3.2, 0.15), 4.3, 0.8, "Request rank\n(admission order)")

    arrow(ax, (2.3, 4.5), (3.2, 4.6))
    arrow(ax, (2.3, 4.1), (3.2, 3.3))
    arrow(ax, (5.8, 4.6), (6.5, 4.6))
    arrow(ax, (5.8, 3.3), (6.5, 3.3))
    arrow(ax, (7.0, 4.2), (5.6, 2.3))
    arrow(ax, (7.9, 2.9), (5.6, 2.1))
    arrow(ax, (5.35, 1.3), (5.35, 0.95))

    ax.set_title("Composite Scoring: Combining Learned Rank and Prompt Length",
                  fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
