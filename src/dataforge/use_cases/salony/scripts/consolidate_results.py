"""
Consolidated results visualisation for Salony dual-generator experiments.

Reads judge result CSVs from a test directory, computes per-criterion and
aggregate metrics, and generates comparative visualisations.

Configuration (model mappings, threshold) is loaded from
``use_cases/salony/config/experiments.yaml`` so that adding a new experiment
does not require touching Python code.
"""

import os
import yaml
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path


# ===========================================================================
# Config
# ===========================================================================

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

with open(_CONFIG_DIR / "experiments.yaml", encoding="utf-8") as f:
    _exp_raw = yaml.safe_load(f)
    model_map = _exp_raw.get("configs", {})

CRITERIA = ["coherence", "completeness", "feasibility", "format", "granularity"]
CRITERIA_LABELS_EN = [
    "Coherence", "Completeness", "Feasibility", "Format", "Granularity",
]
THRESHOLD = 35  # Pass-rate threshold

SCRIPT_DIR = Path(__file__).resolve().parent
TESTS_DIR = SCRIPT_DIR / "../tests"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ===========================================================================
# Load & process test files
# ===========================================================================

test_files = sorted(
    f
    for f in os.listdir(TESTS_DIR)
    if f.endswith("_judge_results.csv")
)

criterion_results = []
aggregate_results = []

for test_file in test_files:
    file_path = os.path.join(TESTS_DIR, test_file)
    df = pd.read_csv(file_path)
    test_name = os.path.splitext(test_file)[0]

    # --- Per-criterion means --------------------------------------------
    criterion_means = {"Test": test_name}
    g1_total, g2_total = 0, 0

    for crit in CRITERIA:
        col_g1 = f"judge_score_a_{crit}"
        col_g2 = f"judge_score_b_{crit}"

        if col_g1 in df.columns:
            g1_mean = pd.to_numeric(df[col_g1], errors="coerce").mean()
            criterion_means[f"G1_{crit}"] = g1_mean
            g1_total += g1_mean
        else:
            print(f"⚠ Columna {col_g1} faltante en {test_file}")
            criterion_means[f"G1_{crit}"] = None

        if col_g2 in df.columns:
            g2_mean = pd.to_numeric(df[col_g2], errors="coerce").mean()
            criterion_means[f"G2_{crit}"] = g2_mean
            g2_total += g2_mean
        else:
            print(f"⚠ Columna {col_g2} faltante en {test_file}")
            criterion_means[f"G2_{crit}"] = None

    criterion_means["G1_Total"] = g1_total
    criterion_means["G2_Total"] = g2_total

    config_id = "C" + test_name.split("_")[2][-1] if "_" in test_name else test_name

    for gen_label, prefix in [("G1", "G1"), ("G2", "G2")]:
        row = {"Config": config_id, "Gen": gen_label}
        for crit, label in zip(CRITERIA, CRITERIA_LABELS_EN):
            row[label] = criterion_means[f"{prefix}_{crit}"]
        criterion_results.append(row)

    # --- Aggregate metrics -----------------------------------------------
    aggregate_means = {"Test": test_name}

    for gen_prefix, score_col, winner_val in [
        ("G1", "judge_score_a_total", "A"),
        ("G2", "judge_score_b_total", "B"),
    ]:
        if score_col in df.columns:
            aggregate_means[f"{gen_prefix}_Mean"] = df[score_col].mean()
            aggregate_means[f"{gen_prefix}_Std"] = df[score_col].std()
            aggregate_means[f"{gen_prefix}_Pass_Rate"] = (
                df[score_col] >= THRESHOLD
            ).mean() * 100
        else:
            print(f"⚠ Columna {score_col} faltante en {test_file}")
            aggregate_means[f"{gen_prefix}_Mean"] = None
            aggregate_means[f"{gen_prefix}_Std"] = None
            aggregate_means[f"{gen_prefix}_Pass_Rate"] = None

        if "judge_winner" in df.columns:
            aggregate_means[f"{gen_prefix}_Win_Rate"] = (
                df["judge_winner"] == winner_val
            ).mean() * 100
        else:
            aggregate_means[f"{gen_prefix}_Win_Rate"] = None

    for gen_label, prefix in [("G1", "G1"), ("G2", "G2")]:
        aggregate_results.append({
            "Config": config_id,
            "Gen": gen_label,
            "Mean": aggregate_means[f"{prefix}_Mean"],
            "Std": aggregate_means[f"{prefix}_Std"],
            "Pass": aggregate_means[f"{prefix}_Pass_Rate"],
            "Win": aggregate_means[f"{prefix}_Win_Rate"],
        })

# ===========================================================================
# Save consolidated tables
# ===========================================================================

criterion_df = (
    pd.DataFrame(criterion_results)
    .sort_values(by=["Config", "Gen"])
    .round(2)
)
criterion_df.to_csv(os.path.join(OUTPUTS_DIR, "criterion_summary.csv"), index=False)

aggregate_df = (
    pd.DataFrame(aggregate_results)
    .sort_values(by=["Config", "Gen"])
    .round(2)
)
aggregate_df.to_csv(os.path.join(OUTPUTS_DIR, "aggregate_summary.csv"), index=False)

# ===========================================================================
# Chart 1 — Criteria comparison overlay
# ===========================================================================

plt.figure(figsize=(8, 6))
configs = sorted(criterion_df["Config"].unique())
x_index = range(len(configs))

for crit, label in zip(CRITERIA, CRITERIA_LABELS_EN):
    g1_scores = [
        criterion_df[(criterion_df["Config"] == c) & (criterion_df["Gen"] == "G1")][label].values[0]
        for c in configs
    ]
    g2_scores = [
        criterion_df[(criterion_df["Config"] == c) & (criterion_df["Gen"] == "G2")][label].values[0]
        for c in configs
    ]
    plt.plot(x_index, g1_scores, marker="x", label=f"G1_{label}")
    plt.plot(x_index, g2_scores, marker="o", label=f"G2_{label}")

plt.xticks(x_index, configs)
plt.ylabel("Average Score")
plt.title("Comparison of Evaluation Criteria Across Configurations")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUTS_DIR, "criterion_comparison_all.png"),
    dpi=300, bbox_inches="tight",
)
plt.close()

# ===========================================================================
# Chart 2 — Total score distribution (boxplot)
# ===========================================================================

test_files = sorted(
    f for f in os.listdir(TESTS_DIR) if f.endswith("_judge_results.csv")
)

data_by_group = {"G1": [], "G2": []}
labels = []

for test_file in test_files:
    file_path = os.path.join(TESTS_DIR, test_file)
    df = pd.read_csv(file_path)

    if "judge_score_a_total" in df.columns and "judge_score_b_total" in df.columns:
        data_by_group["G1"].append(df["judge_score_a_total"].values)
        data_by_group["G2"].append(df["judge_score_b_total"].values)
        labels.append("C" + os.path.splitext(test_file)[0].split("_")[2][-1])

boxplot_data = []
boxplot_labels = []

for i, label in enumerate(labels):
    if label not in model_map:
        continue
    boxplot_data.append(data_by_group["G1"][i])
    boxplot_labels.append(f"{label} (G1)\n{model_map[label]['G1']}")
    boxplot_data.append(data_by_group["G2"][i])
    boxplot_labels.append(f"{label} (G2)\n{model_map[label]['G2']}")

plt.figure()
plt.boxplot(boxplot_data, labels=boxplot_labels, vert=True)
plt.ylabel("Total Scores")
plt.title("Score Distribution Across Tests")
plt.xticks(rotation=0, ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "total_score_distribution.png"), dpi=300)
plt.close()

# ===========================================================================
# Chart 3 — Criteria subplots (bar charts)
# ===========================================================================

configs = sorted(criterion_df["Config"].unique())
x = np.arange(len(configs))
width = 0.35

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(3, 2, figure=fig)

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[2, :]),
]

for idx, (crit, label) in enumerate(zip(CRITERIA, CRITERIA_LABELS_EN)):
    ax = axes[idx]

    g1_vals = [
        criterion_df[(criterion_df["Config"] == c) & (criterion_df["Gen"] == "G1")][label].values[0]
        for c in configs
    ]
    g2_vals = [
        criterion_df[(criterion_df["Config"] == c) & (criterion_df["Gen"] == "G2")][label].values[0]
        for c in configs
    ]

    bars_g1 = ax.bar(x - width / 2, g1_vals, width, label="G1")
    bars_g2 = ax.bar(x + width / 2, g2_vals, width, label="G2")

    if idx == 4 and label in model_map.get(configs[0], {}):
        for i, bar in enumerate(bars_g1):
            config = configs[i]
            if config in model_map:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    model_map[config]["G1"],
                    ha="center", va="bottom", fontsize=7,
                )
        for i, bar in enumerate(bars_g2):
            config = configs[i]
            if config in model_map:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    model_map[config]["G2"],
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_title(label)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Average Score")
    ax.set_ylim(0, 10)
    ax.legend()

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUTS_DIR, "criteria_subplots.png"),
    dpi=300, bbox_inches="tight",
)
plt.close()
