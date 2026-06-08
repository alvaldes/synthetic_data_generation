import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


# ============================
# DIRECTORIOS Y CONFIGURACION
# ============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.join(SCRIPT_DIR, "../tests")
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
CRITERIA = ["coherence", "completeness", "feasibility", "format", "granularity"]
CRITERIA_LABELS_ES = [
    "Coherencia",
    "Completitud",
    "Viabilidad",
    "Formato",
    "Granularidad",
]
CRITERIA_LABELS_EN = [
    "Coherence",
    "Completeness",
    "Feasibility",
    "Format",
    "Granularity",
]
THRESHOLD = 35  # Umbral para Pass Rate
model_map = {
    "C1": {"G1": "Llama 3.1", "G2": "Mistral"},
    "C2": {"G1": "Qwen 3", "G2": "Gemma 4"},
    "C3": {"G1": "Gemma 3", "G2": "Mistral Nemo"},
    "C4": {"G1": "Qwen 3", "G2": "Phi 4"},
}

os.makedirs(OUTPUTS_DIR, exist_ok=True)


def get_config_id(test_name):
    return "C" + test_name.split("_")[2][-1] if "_" in test_name else test_name


# ============================
# CARGAR Y PROCESAR TESTS
# ============================
tests_files = [f for f in os.listdir(TESTS_DIR) if f.endswith("_judge_results.csv")]
tests_files = sorted(tests_files)

criterion_results = []
aggregate_results = []

# cache de dataframes
dfs = {}

for test_file in tests_files:
    file_path = os.path.join(TESTS_DIR, test_file)
    df = pd.read_csv(file_path)

    # convertir una sola vez
    df = df.apply(
        lambda col: pd.to_numeric(col, errors="coerce")
        if col.dtype == "object"
        else col
    )

    dfs[test_file] = df

    test_name = os.path.splitext(test_file)[0]
    config_id = get_config_id(test_name)

    # ============================
    # CRITERIOS
    # ============================
    criterion_means = {"Test": test_name}

    g1_total, g2_total = 0, 0

    for crit in CRITERIA:
        col_g1 = f"judge_score_a_{crit}"
        col_g2 = f"judge_score_b_{crit}"

        if col_g1 in df.columns:
            g1_mean = df[col_g1].mean()
            criterion_means[f"G1_{crit}"] = g1_mean
            g1_total += g1_mean
        else:
            print(f"Advertencia: Columna {col_g1} faltante en {test_file}")
            criterion_means[f"G1_{crit}"] = None

        if col_g2 in df.columns:
            g2_mean = df[col_g2].mean()
            criterion_means[f"G2_{crit}"] = g2_mean
            g2_total += g2_mean
        else:
            print(f"Advertencia: Columna {col_g2} faltante en {test_file}")
            criterion_means[f"G2_{crit}"] = None

    criterion_means["G1_Total"] = g1_total
    criterion_means["G2_Total"] = g2_total

    row_g1 = {"Config": config_id, "Gen": "G1"}
    row_g2 = {"Config": config_id, "Gen": "G2"}

    for crit, label in zip(CRITERIA, CRITERIA_LABELS_EN):
        row_g1[label] = criterion_means[f"G1_{crit}"]
        row_g2[label] = criterion_means[f"G2_{crit}"]

    criterion_results.append(row_g1)
    criterion_results.append(row_g2)

    # ============================
    # METRICAS AGREGADAS
    # ============================
    aggregate_means = {"Test": test_name}

    if "judge_score_a_total" in df.columns:
        aggregate_means["G1_Mean"] = df["judge_score_a_total"].mean()
        aggregate_means["G1_Std"] = df["judge_score_a_total"].std()
        aggregate_means["G1_Pass_Rate"] = (
            df["judge_score_a_total"] >= THRESHOLD
        ).mean() * 100
    else:
        print(f"Advertencia: Columna judge_score_a_total faltante en {test_file}")
        aggregate_means["G1_Mean"] = None
        aggregate_means["G1_Std"] = None
        aggregate_means["G1_Pass_Rate"] = None

    if "judge_score_b_total" in df.columns:
        aggregate_means["G2_Mean"] = df["judge_score_b_total"].mean()
        aggregate_means["G2_Std"] = df["judge_score_b_total"].std()
        aggregate_means["G2_Pass_Rate"] = (
            df["judge_score_b_total"] >= THRESHOLD
        ).mean() * 100
    else:
        print(f"Advertencia: Columna judge_score_b_total faltante en {test_file}")
        aggregate_means["G2_Mean"] = None
        aggregate_means["G2_Std"] = None
        aggregate_means["G2_Pass_Rate"] = None

    if "judge_winner" in df.columns:
        aggregate_means["G1_Win_Rate"] = (df["judge_winner"] == "A").mean() * 100
        aggregate_means["G2_Win_Rate"] = (df["judge_winner"] == "B").mean() * 100
    else:
        aggregate_means["G1_Win_Rate"] = None
        aggregate_means["G2_Win_Rate"] = None

    aggregate_results.append(
        {
            "Config": config_id,
            "Gen": "G1",
            "Mean": aggregate_means["G1_Mean"],
            "Std": aggregate_means["G1_Std"],
            "Pass": aggregate_means["G1_Pass_Rate"],
            "Win": aggregate_means["G1_Win_Rate"],
        }
    )

    aggregate_results.append(
        {
            "Config": config_id,
            "Gen": "G2",
            "Mean": aggregate_means["G2_Mean"],
            "Std": aggregate_means["G2_Std"],
            "Pass": aggregate_means["G2_Pass_Rate"],
            "Win": aggregate_means["G2_Win_Rate"],
        }
    )

# ============================
# TABLAS
# ============================
criterion_df = pd.DataFrame(criterion_results).sort_values(by=["Config", "Gen"])
criterion_df = criterion_df.round(2)

aggregate_df = pd.DataFrame(aggregate_results).sort_values(by=["Config", "Gen"])
aggregate_df = aggregate_df.round(2)

configs = sorted(criterion_df["Config"].unique())

criterion_df.to_csv(os.path.join(OUTPUTS_DIR, "criterion_summary.csv"), index=False)
aggregate_df.to_csv(os.path.join(OUTPUTS_DIR, "aggregate_summary.csv"), index=False)

# pivot para acceso eficiente
pivot_df = criterion_df.pivot(index="Config", columns="Gen")

# ============================
# GRAFICO LINEAL
# ============================
plt.figure(figsize=(8, 6))
x_index = range(len(configs))

for label in CRITERIA_LABELS_EN:
    g1_scores = pivot_df[(label, "G1")].values
    g2_scores = pivot_df[(label, "G2")].values

    plt.plot(x_index, g1_scores, marker="x", label=f"G1_{label}")
    plt.plot(x_index, g2_scores, marker="o", label=f"G2_{label}")

plt.xticks(x_index, configs)
plt.ylabel("Average Score")
plt.title("Comparison of Evaluation Criteria Across Configurations")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUTS_DIR, "criterion_comparison_all.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ============================
# BOXPLOT
# ============================
data_by_group = {"G1": [], "G2": []}
labels = []

for test_file in tests_files:
    df = dfs[test_file]
    test_name = os.path.splitext(test_file)[0]
    config_id = get_config_id(test_name)

    if "judge_score_a_total" in df.columns and "judge_score_b_total" in df.columns:
        data_by_group["G1"].append(df["judge_score_a_total"].values)
        data_by_group["G2"].append(df["judge_score_b_total"].values)
        labels.append(config_id)

boxplot_data = []
boxplot_labels = []

for config in configs:
    i = labels.index(config)

    boxplot_data.append(data_by_group["G1"][i])
    boxplot_labels.append(f"{config} (G1)\n{model_map[config]['G1']}")

    boxplot_data.append(data_by_group["G2"][i])
    boxplot_labels.append(f"{config} (G2)\n{model_map[config]['G2']}")

plt.figure()
plt.boxplot(boxplot_data, tick_labels=boxplot_labels, vert=True)
plt.ylabel("Total Scores")
plt.title("Score Distribution Across Tests")
plt.xticks(rotation=0, ha="center", fontsize=8)
plt.tight_layout()

plt.savefig(os.path.join(OUTPUTS_DIR, "total_score_distribution.png"), dpi=300)

plt.figure()
plt.boxplot(boxplot_data, tick_labels=boxplot_labels, vert=True)
plt.ylabel("Puntajes Totales")
plt.title("Distribución de Puntajes entre Tests")
plt.xticks(rotation=0, ha="center", fontsize=8)
plt.tight_layout()

plt.savefig(os.path.join(OUTPUTS_DIR, "total_score_distribution_es.png"), dpi=300)

# ============================
# SUBPLOTS
# ============================
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

for idx, label in enumerate(CRITERIA_LABELS_EN):
    ax = axes[idx]

    g1_vals = pivot_df[(label, "G1")].values
    g2_vals = pivot_df[(label, "G2")].values

    bars_g1 = ax.bar(x - width / 2, g1_vals, width, label="G1")
    bars_g2 = ax.bar(x + width / 2, g2_vals, width, label="G2")

    if idx == 4:
        for i, bar in enumerate(bars_g1):
            config = configs[i]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                model_map[config]["G1"],
                ha="center",
                va="bottom",
                fontsize=7,
            )

        for i, bar in enumerate(bars_g2):
            config = configs[i]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                model_map[config]["G2"],
                ha="center",
                va="bottom",
                fontsize=7,
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
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ============================
# SUBPLOTS (ES)
# ============================
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(3, 2, figure=fig)

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[2, :]),
]

for idx, label in enumerate(CRITERIA_LABELS_ES):
    ax = axes[idx]

    g1_vals = pivot_df[(CRITERIA_LABELS_EN[idx], "G1")].values
    g2_vals = pivot_df[(CRITERIA_LABELS_EN[idx], "G2")].values

    bars_g1 = ax.bar(x - width / 2, g1_vals, width, label="G1")
    bars_g2 = ax.bar(x + width / 2, g2_vals, width, label="G2")

    if idx == 4:
        for i, bar in enumerate(bars_g1):
            config = configs[i]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                model_map[config]["G1"],
                ha="center",
                va="bottom",
                fontsize=7,
            )

        for i, bar in enumerate(bars_g2):
            config = configs[i]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                model_map[config]["G2"],
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_title(label)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Puntaje Promedio")
    ax.set_ylim(0, 10)
    ax.legend()

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUTS_DIR, "criteria_subplots_es.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()
