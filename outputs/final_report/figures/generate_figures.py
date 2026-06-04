"""Generate all publication-ready figures for Phase 10 thesis artifacts.

Produces PNG (300 dpi) and SVG versions of:
  1. BM25 vs Dense vs Hybrid (Recall@k line chart)
  2. Rule vs GPT-5 Planner comparison (bar chart)
  3. Reasoning-Type Performance heatmap
  4. Category-Level Performance grouped bars
  5. Cost vs Performance scatter (tradeoff)

Run from project root:
    python outputs/final_report/figures/generate_figures.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path(__file__).parent

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "bm25":         "#E74C3C",
    "dense":        "#3498DB",
    "hybrid":       "#2ECC71",
    "rule":         "#F39C12",
    "gpt5":         "#9B59B6",
}

# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: BM25 vs Dense vs Hybrid — Recall@k
# ──────────────────────────────────────────────────────────────────────────────

def fig1_retrieval_baselines():
    k = [1, 3, 5, 10]
    bm25_recall   = [0.2982, 0.5096, 0.6022, 0.7127]
    dense_recall  = [0.3456, 0.5399, 0.6294, 0.7404]
    hybrid_recall = [0.3680, 0.5689, 0.6228, 0.7447]

    bm25_mrr   = [0.4573] * 4
    dense_mrr  = [0.5014] * 4
    hybrid_mrr = [0.5295] * 4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Recall@k
    ax1.plot(k, bm25_recall,   "o-", color=COLORS["bm25"],   label="BM25",          lw=2, ms=7)
    ax1.plot(k, dense_recall,  "s-", color=COLORS["dense"],  label="Dense (BGE)",   lw=2, ms=7)
    ax1.plot(k, hybrid_recall, "^-", color=COLORS["hybrid"], label="Hybrid (RRF)",  lw=2, ms=7)
    ax1.set_xlabel("k")
    ax1.set_ylabel("Recall@k")
    ax1.set_title("(a) Recall@k — Baseline Retrieval Methods")
    ax1.set_xticks(k)
    ax1.set_ylim(0.25, 0.80)
    ax1.legend(loc="lower right")

    # Bar for MRR
    x = np.arange(3)
    mrr_vals = [0.4573, 0.5014, 0.5295]
    ndcg_vals = [0.5089, 0.5479, 0.5664]
    labels = ["BM25", "Dense\n(BGE)", "Hybrid\n(RRF)"]
    colors = [COLORS["bm25"], COLORS["dense"], COLORS["hybrid"]]

    bars_mrr = ax2.bar(x - 0.2, mrr_vals, 0.35, label="MRR", color=colors, alpha=0.85)
    bars_ndcg = ax2.bar(x + 0.2, ndcg_vals, 0.35, label="nDCG@10",
                        color=colors, alpha=0.50, hatch="//")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Score")
    ax2.set_title("(b) MRR and nDCG@10 — Baseline Methods")
    ax2.set_ylim(0.40, 0.62)
    ax2.legend(["MRR", "nDCG@10"])

    # Annotate bars
    for bar in bars_mrr:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_ndcg:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT / "fig1_retrieval_baselines.png")
    fig.savefig(OUT / "fig1_retrieval_baselines.svg")
    plt.close(fig)
    print("  fig1_retrieval_baselines done")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Rule Planner vs GPT-5 Planner
# ──────────────────────────────────────────────────────────────────────────────

def fig2_planner_comparison():
    metrics = ["Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR", "nDCG@10", "Hit@10"]
    rule  = [0.3575, 0.5491, 0.6294, 0.7553, 0.5220, 0.5632, 0.8079]
    gpt5  = [0.3680, 0.5662, 0.6491, 0.7500, 0.5325, 0.5702, 0.8026]
    delta = [g - r for g, r in zip(gpt5, rule)]

    x = np.arange(len(metrics))
    w = 0.32

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    bars_r = ax1.bar(x - w/2, rule, w, label="Rule Planner",  color=COLORS["rule"],  alpha=0.88)
    bars_g = ax1.bar(x + w/2, gpt5, w, label="GPT-5 Planner", color=COLORS["gpt5"], alpha=0.88)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=20, ha="right")
    ax1.set_ylabel("Score")
    ax1.set_title("(a) Retrieval Metrics: Rule vs. GPT-5 Planner")
    ax1.set_ylim(0.30, 0.88)
    ax1.legend()

    # Delta bars
    bar_colors = [COLORS["gpt5"] if d >= 0 else COLORS["rule"] for d in delta]
    ax2.barh(metrics[::-1], delta[::-1], color=[COLORS["gpt5"] if d >= 0 else COLORS["rule"] for d in delta[::-1]], alpha=0.85)
    ax2.axvline(0, color="black", lw=0.8, linestyle="--")
    ax2.set_xlabel(r"$\Delta$ (GPT-5 $-$ Rule Planner)")
    ax2.set_title("(b) Delta: GPT-5 vs. Rule Planner")
    for i, (d, m) in enumerate(zip(delta[::-1], metrics[::-1])):
        ax2.text(d + (0.0005 if d >= 0 else -0.0005),
                 i, f"{d:+.4f}", va="center",
                 ha="left" if d >= 0 else "right", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT / "fig2_planner_comparison.png")
    fig.savefig(OUT / "fig2_planner_comparison.svg")
    plt.close(fig)
    print("  fig2_planner_comparison done")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Reasoning-Type Performance Heatmap
# ──────────────────────────────────────────────────────────────────────────────

def fig3_reasoning_type_heatmap():
    rt_labels = ["aggregation\n(n=35)", "comparison\n(n=33)", "exception\n(n=34)",
                 "multi_hop\n(n=53)", "single_hop\n(n=220)", "temporal\n(n=5)"]
    systems   = ["BM25", "Dense", "Hybrid", "Rule\nPlanner", "GPT-5\nPlanner"]

    # Recall@10
    data = np.array([
        [0.9238, 0.8952, 0.9143, 0.9143, 0.9143],
        [0.8182, 0.8485, 0.8485, 0.8788, 0.8485],
        [0.4118, 0.3824, 0.4118, 0.3684, 0.3824],
        [0.5377, 0.4906, 0.5189, 0.5309, 0.5189],
        [0.7636, 0.8318, 0.8227, 0.8364, 0.8364],
        [0.2000, 0.0000, 0.1000, 0.2000, 0.1000],
    ])

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Recall@10", shrink=0.85)

    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(systems)
    ax.set_yticks(range(len(rt_labels)))
    ax.set_yticklabels(rt_labels)
    ax.set_title("Recall@10 by Reasoning Type and Retrieval System")

    for i in range(len(rt_labels)):
        for j in range(len(systems)):
            val = data[i, j]
            color = "black" if 0.3 < val < 0.8 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT / "fig3_reasoning_type_heatmap.png")
    fig.savefig(OUT / "fig3_reasoning_type_heatmap.svg")
    plt.close(fig)
    print("  fig3_reasoning_type_heatmap done")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: Category-Level Performance (Recall@10)
# ──────────────────────────────────────────────────────────────────────────────

def fig4_category_performance():
    categories = ["API Docs\n(n=68)", "HR Policies\n(n=54)", "Meeting Notes\n(n=100)",
                  "Security\n(n=64)", "System Design\n(n=45)", "Travel\n(n=49)"]
    bm25_r10  = [0.9706, 0.6019, 0.6083, 0.6172, 0.5111, 1.0000]
    dense_r10 = [1.0000, 0.6111, 0.5283, 0.7344, 0.7000, 1.0000]
    hyb_r10   = [1.0000, 0.5648, 0.6150, 0.7266, 0.6111, 1.0000]
    rule_r10  = [1.0000, 0.5463, 0.6050, 0.7109, 0.7222, 1.0000]

    x = np.arange(len(categories))
    w = 0.20

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - 1.5*w, bm25_r10,  w, label="BM25",         color=COLORS["bm25"],   alpha=0.85)
    ax.bar(x - 0.5*w, dense_r10, w, label="Dense (BGE)",  color=COLORS["dense"],  alpha=0.85)
    ax.bar(x + 0.5*w, hyb_r10,   w, label="Hybrid (RRF)", color=COLORS["hybrid"], alpha=0.85)
    ax.bar(x + 1.5*w, rule_r10,  w, label="Rule Planner", color=COLORS["rule"],   alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall@10 by Document Category")
    ax.set_ylim(0.40, 1.08)
    ax.legend(loc="lower left", ncol=4)
    ax.axhline(1.0, color="gray", lw=0.6, linestyle=":")

    plt.tight_layout()
    fig.savefig(OUT / "fig4_category_performance.png")
    fig.savefig(OUT / "fig4_category_performance.svg")
    plt.close(fig)
    print("  fig4_category_performance done")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5: Cost vs Performance Tradeoff
# ──────────────────────────────────────────────────────────────────────────────

def fig5_cost_performance_tradeoff():
    systems = ["BM25", "Dense (BGE)", "Hybrid (RRF)", "Rule Planner", "GPT-5 Planner"]
    costs   = [0.000, 0.000, 0.000, 0.000, 6.184]   # total USD for 380 queries
    recall10 = [0.7127, 0.7404, 0.7447, 0.7553, 0.7500]
    mrr      = [0.4573, 0.5014, 0.5295, 0.5220, 0.5325]
    ndcg10   = [0.5089, 0.5479, 0.5664, 0.5632, 0.5702]
    colors_list = [COLORS["bm25"], COLORS["dense"], COLORS["hybrid"], COLORS["rule"], COLORS["gpt5"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: cost vs Recall@10 (log-ish x axis with jitter for zero-cost)
    cost_plot = [0.05, 0.10, 0.15, 0.20, 6.184]  # visual jitter for zero-cost systems
    for i, (c, r, name, col) in enumerate(zip(cost_plot, recall10, systems, colors_list)):
        ax1.scatter(c, r, color=col, s=140, zorder=5, edgecolors="white", lw=1.5)
        offset_y = 0.003 if i < 4 else -0.006
        ax1.annotate(name, (c, r), textcoords="offset points",
                     xytext=(6, 5 if i % 2 == 0 else -15), fontsize=9,
                     color=col, fontweight="bold")
    ax1.axvline(0.5, color="gray", lw=0.8, linestyle="--", alpha=0.6)
    ax1.text(0.55, 0.715, "← zero cost\n  region", fontsize=8, color="gray")
    ax1.set_xlabel("Total Cost for 380 Queries (USD)\n[zero-cost systems shown with jitter]")
    ax1.set_ylabel("Recall@10")
    ax1.set_title("(a) Cost vs. Recall@10 Tradeoff")
    ax1.set_xlim(-0.5, 7.0)
    ax1.set_ylim(0.70, 0.77)

    # Right: radar-style bar for multi-metric comparison
    metric_names = ["Recall@10", "MRR", "nDCG@10"]
    x = np.arange(3)
    w = 0.15
    offsets = [-2, -1, 0, 1, 2]
    for i, (name, r10, mr, nd, col) in enumerate(zip(systems, recall10, mrr, ndcg10, colors_list)):
        vals = [r10, mr, nd]
        ax2.bar(x + offsets[i]*w, vals, w, label=name, color=col, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_names)
    ax2.set_ylabel("Score")
    ax2.set_title("(b) Multi-Metric Comparison")
    ax2.set_ylim(0.42, 0.79)
    ax2.legend(fontsize=8.5, ncol=2, loc="lower right")

    plt.tight_layout()
    fig.savefig(OUT / "fig5_cost_performance_tradeoff.png")
    fig.savefig(OUT / "fig5_cost_performance_tradeoff.svg")
    plt.close(fig)
    print("  fig5_cost_performance_tradeoff done")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_retrieval_baselines()
    fig2_planner_comparison()
    fig3_reasoning_type_heatmap()
    fig4_category_performance()
    fig5_cost_performance_tradeoff()
    print(f"All figures written to: {OUT}")
