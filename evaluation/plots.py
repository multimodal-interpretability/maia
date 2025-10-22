# ==== summary_panel.py (balanced widths, figure-level legend, a)/b)/c/)
import glob
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import (
    AutoMinorLocator,
    MultipleLocator,
    PercentFormatter,
)
from tqdm.auto import tqdm

# ---------------- Config ----------------
FAMILIES = {
    "Gemma 3 27B": "../results/local-google/gemma-3-27b-it",
    "Mistral Small 3.2 24B": "../results/local-mistralai/Mistral-Small-3.2-24B",
    "Claude Sonnet 4": "../results/claude",
}
FRIENDLY_MAP = {
    "resnet152": "ResNet-152",
    "dino_vits8": "DINO",
    "clip-RN50": "CLIP",
    "synthetic_neurons": "Synthetic",
}
MODEL_ORDER = ["ResNet-152", "DINO", "CLIP", "Synthetic"]
YLIMS_BOTTOM = {"ResNet-152": 0.5, "DINO": -1.2, "CLIP": 1, "Synthetic": 0.05}

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

sns.set_theme(context="paper", style="whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "font.family": "DejaVu Sans",
        "axes.linewidth": 1.0,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    }
)

FAMILY_LIST = list(FAMILIES.keys())
PALETTE = sns.color_palette("colorblind", n_colors=len(FAMILY_LIST) + 1)  # +1 MAIA
FAMILY_COLORS = {fam: PALETTE[i] for i, fam in enumerate(FAMILY_LIST)}
MAIA_COLOR = PALETTE[len(FAMILY_LIST)]

MAIA_REFERENCE = {
    "positive": [3.357, 1.820, 3.329, 0.563],
    "negative": [1.239, -0.155, 1.804, 0.228],
}


# ------------- Activations helpers -------------
def collect_file_map(
    base_dir: str, friendly_map: Optional[Dict[str, str]] = None
) -> Dict[str, List[str]]:
    pattern = os.path.join(base_dir, "*", "*", "*", "eval_results.json")
    paths = glob.glob(pattern)
    file_map: Dict[str, List[str]] = {}
    base_parts = os.path.normpath(base_dir).split(os.sep)
    for p in paths:
        parts = os.path.normpath(p).split(os.sep)
        if len(parts) < len(base_parts) + 4:
            continue
        submodel_dir = parts[len(base_parts)]
        key = (
            friendly_map.get(submodel_dir, submodel_dir)
            if friendly_map
            else submodel_dir
        )
        file_map.setdefault(key, []).append(p)
    for model in MODEL_ORDER:
        file_map.setdefault(model, [])
    return file_map


def compute_means(file_map: Dict[str, List[str]]):
    results_dict = {"positive": [], "negative": []}
    for model_name in MODEL_ORDER:
        files = file_map.get(model_name, [])
        pos, neg = [], []
        for f in tqdm(files, desc=f"{model_name}", leave=False):
            with open(f, "r") as fp:
                results = json.load(fp)
            pos_scores = np.array(results["positive_scores"])
            neg_scores = np.array(results["negative_scores"])
            pos.extend(pos_scores)
            neg.extend(neg_scores)
        m = lambda arr: float(np.mean(arr)) if len(arr) else float("nan")
        results_dict["positive"].append(m(pos))
        results_dict["negative"].append(m(neg))

    return results_dict


def plot_activations_panel(fig, outer_cell, all_results):
    """Builds col-1 with a *sub*-GridSpec; returns legend handles for fig-level legend."""
    families = FAMILY_LIST
    offsets = np.linspace(-0.18, 0.18, num=len(families) + 1)

    gs1 = outer_cell.subgridspec(1, len(MODEL_ORDER), wspace=0.35)
    axes = [fig.add_subplot(gs1[0, i]) for i in range(len(MODEL_ORDER))]

    for i, model in enumerate(MODEL_ORDER):
        ax = axes[i]
        x0 = 0.0
        bottom = YLIMS_BOTTOM.get(model, None)
        all_vals = []
        for fam in families:
            pos = all_results[fam]["positive"][i]
            neg = all_results[fam]["negative"][i]
            if not np.isnan(pos):
                all_vals.append(pos)
            if not np.isnan(neg):
                all_vals.append(neg)

        all_vals += [
            MAIA_REFERENCE["positive"][i],
            MAIA_REFERENCE["negative"][i],
        ]

        if not all_vals:
            ymin, ymax = (-1, 1)
        else:
            vmin, vmax = np.min(all_vals), np.max(all_vals)
            pad = 0.12 * (abs(vmax) + abs(vmin) + 1e-9)
            ymin = bottom if bottom is not None else (vmin - pad)
            ymax = vmax + pad
        ax.set_ylim(ymin, ymax)

        # MAIA
        x_ref = x0 + offsets[0]
        pos_ref = MAIA_REFERENCE["positive"][i]
        neg_ref = MAIA_REFERENCE["negative"][i]
        ax.plot(
            [x_ref, x_ref],
            [neg_ref, pos_ref],
            color="black",
            lw=1.3,
            alpha=0.6,
            zorder=2,
        )
        ax.plot(
            x_ref,
            pos_ref,
            marker="+",
            markersize=12,
            color=MAIA_COLOR,
            markeredgewidth=2,
            zorder=3,
        )
        ax.plot(
            x_ref,
            neg_ref,
            marker="_",
            markersize=12,
            color=MAIA_COLOR,
            markeredgewidth=2,
            zorder=3,
        )

        # Families
        for off, fam in zip(offsets[1:], families):
            pos = all_results[fam]["positive"][i]
            neg = all_results[fam]["negative"][i]
            x = x0 + off
            ax.plot([x, x], [neg, pos], color="black", lw=1.3, alpha=0.6, zorder=2)
            ax.plot(
                x,
                pos,
                marker="+",
                markersize=12,
                color=FAMILY_COLORS[fam],
                markeredgewidth=2,
                zorder=3,
            )
            ax.plot(
                x,
                neg,
                marker="_",
                markersize=12,
                color=FAMILY_COLORS[fam],
                markeredgewidth=2,
                zorder=3,
            )

        ax.grid(axis="y", which="major", linewidth=0.6, alpha=0.35, zorder=0)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(axis="y", which="minor", linewidth=0.4, alpha=0.2, zorder=0)
        ax.set_xticks([x0])
        ax.set_xticklabels([model])
        if i == 0:
            ax.set_ylabel("Activations")
        ax.tick_params(axis="y")
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(1.0)
        ax.margins(x=0.30)

    # Panel label only
    axes[0].set_title("a)", loc="left")

    handles = [
        plt.Line2D(
            [],
            [],
            color=MAIA_COLOR,
            marker="s",
            linestyle="None",
            markersize=8,
            label="MAIA (GPT-4V)",
        )
    ] + [
        plt.Line2D(
            [],
            [],
            color=FAMILY_COLORS[fam],
            marker="s",
            linestyle="None",
            markersize=8,
            label=fam,
        )
        for fam in families
    ]
    return handles


# ------------- Tool usage -------------
TOOLS = [
    "text2image",
    "edit_images",
    "dataset_exemplars",
    "summarize_images",
    "describe_images",
    "sampler",
]


def gather_histories():
    return {
        "Gemma 3 27B": glob.glob(
            "../results/local-google/gemma-3-27b-it/*/*/*/history.json"
        ),
        "Mistral Small 3.2 24B": glob.glob(
            "../results/local-mistralai/Mistral-Small-3.2-24B/*/*/*/history.json"
        ),
        "Claude Sonnet 4": glob.glob("../results/claude/*/*/*/history.json"),
    }


def compute_tool_usage(history_files: Dict[str, List[str]]):
    model_to_percent = {}
    for title, files in history_files.items():
        per_expt_counts = {tool: 0 for tool in TOOLS}
        total_experiments = len(files)
        for file in tqdm(files, desc=f"Tools {title}", leave=False):
            with open(file, "r") as f:
                history = json.load(f)[2:]
                history = [
                    h["content"][0]["text"] for h in history if h["role"] == "assistant"
                ]
            tools_used_in_experiment = set()
            for entry in history:
                for tool in TOOLS:
                    if tool in entry:
                        tools_used_in_experiment.add(tool)
            for tool in tools_used_in_experiment:
                per_expt_counts[tool] += 1
        denom = total_experiments if total_experiments > 0 else 1
        model_to_percent[title] = [(per_expt_counts[t] / denom) * 100 for t in TOOLS]
    return model_to_percent


def plot_tool_histogram(ax, model_to_percent: Dict[str, List[float]]):
    x = np.arange(len(TOOLS))
    model_names = list(model_to_percent.keys())
    num_models = len(model_names)
    width = min(0.8 / max(num_models, 1), 0.22)
    palette = sns.color_palette("colorblind", n_colors=num_models)

    for i, (model, color) in enumerate(zip(model_names, palette)):
        offsets = x - (width * (num_models - 1) / 2.0) + i * width
        ax.bar(
            offsets,
            model_to_percent[model],
            width=width,
            label=model,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

    ax.grid(axis="y", which="major", linewidth=0.6, alpha=0.35, zorder=0)
    ax.grid(axis="y", which="minor", linewidth=0.4, alpha=0.2, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(TOOLS, rotation=25, ha="right")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_ylabel("Experiments with ≥1 Call (%)")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.0)
    ax.set_title("b)", loc="left")


# ------------- Average turns -------------
def compute_avg_turns(history_files: Dict[str, List[str]]):
    model_to_avgturns = {}
    for title, files in history_files.items():
        total_turns = 0
        for file in tqdm(files, desc=f"AvgTurns {title}", leave=False):
            with open(file, "r") as f:
                history = json.load(f)[2:]
                history = [
                    h["content"][0]["text"] for h in history if h["role"] == "assistant"
                ]
            total_turns += len(history)
        denom = len(files) if files else 1
        model_to_avgturns[title] = total_turns / denom
    return model_to_avgturns


def plot_avg_turns(ax, model_to_avgturns: Dict[str, float]):
    models = list(model_to_avgturns.keys())
    values = list(model_to_avgturns.values())
    n = len(models)

    # same color scheme as other panels
    palette = sns.color_palette("colorblind", n_colors=n)

    # --- cluster the three bars tightly like a grouped bar plot ---
    width = 0.28  # bar width
    gap = 0.06  # small gap between bars
    total = n * width + (n - 1) * gap
    start = -total / 2.0  # center the cluster at x=0
    xs = [start + i * (width + gap) + width / 2.0 for i in range(n)]

    for i, (x, val) in enumerate(zip(xs, values)):
        ax.bar(
            x,
            val,
            width=width,
            color=palette[i],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

    # y-axis styling
    ax.grid(axis="y", which="major", linewidth=0.6, alpha=0.35, zorder=0)
    ax.grid(axis="y", which="minor", linewidth=0.4, alpha=0.2, zorder=0)
    ax.set_ylabel("Average Turns per Experiment")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.0)

    ymax = max(values) * 1.15 if values else 1
    ax.set_ylim(0, ymax)

    # --- NO x ticks; just a single x-axis label ---
    ax.set_xticks([])
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_xlabel("LLM backbone")
    ax.set_xlim(start - 0.15, start + total + 0.15)

    ax.set_title("c)", loc="left")


# ------------- Run: 1×3 with width ratios -------------
if __name__ == "__main__":
    # Activations data
    all_results = {}
    for fam, base in FAMILIES.items():
        fmap = collect_file_map(base, FRIENDLY_MAP)
        all_results[fam] = compute_means(fmap)

    # Histories
    histories = gather_histories()
    model_to_percent = compute_tool_usage(histories)
    model_to_avgturns = compute_avg_turns(histories)

    # Figure + outer GridSpec (col-1 dominant and overall wider)
    fig = plt.figure(figsize=(19, 4.3), constrained_layout=False)
    outer = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[
            3.6,
            1.8,
            0.8,
        ],  # ← more weight to col 1; cols 2/3 slightly slimmer
        wspace=0.38,  # ← a touch more horizontal breathing room
    )

    # Col 1 (returns legend handles)
    legend_handles = plot_activations_panel(fig, outer[0, 0], all_results)

    # Col 2 & 3
    ax2 = fig.add_subplot(outer[0, 1])
    ax3 = fig.add_subplot(outer[0, 2])
    plot_tool_histogram(ax2, model_to_percent)
    plot_avg_turns(ax3, model_to_avgturns)

    # Global legend at top center
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),  # move legend further up
        ncol=len(legend_handles),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        prop={"size": 14},
        handlelength=1.6,
        handleheight=1.2,
        borderpad=0.5,
    )

    # Room for legend (shrink rect a bit more to increase gap)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out_png = os.path.join(OUTDIR, "summary_panel.png")
    out_pdf = os.path.join(OUTDIR, "summary_panel.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n - {out_pdf}\n - {out_png}")
