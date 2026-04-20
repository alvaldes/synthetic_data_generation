import os
import pandas as pd
from matplotlib import pyplot as plt

# ============================
# DIRECTORIOS Y CONFIGURACION
# ============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.join(SCRIPT_DIR, "../tests")
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
CRITERIA = ["coherence", "completeness", "feasibility", "format", "granularity"]
CRITERIA_LABELS = ["Coherencia", "Completitud", "Viabilidad", "Formato", "Granularidad"]
THRESHOLD = 35  # Umbral para Pass Rate

# Asegurarse de que el directorio de salidas exista
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ============================
# CARGAR Y PROCESAR TESTS
# ============================
# Buscar archivos CSV en la carpeta de tests
tests_files = [
    f
    for f in os.listdir(TESTS_DIR)
    # OMITIR ESTE ARCHIVO
    if f.endswith("_judge_results.csv")
    and f != "salony_dual_test1_output_judge_results.csv"
]

criterion_results = []
aggregate_results = []

for test_file in tests_files:
    # Cargar el archivo CSV
    file_path = os.path.join(TESTS_DIR, test_file)
    df = pd.read_csv(file_path)

    # Identificar nombre del test
    test_name = os.path.splitext(test_file)[0]

    # Calcular promedios por criterio para G1 y G2 y totales
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

    # Agregar totales y ganador a los criterios
    criterion_means["G1_Total"] = g1_total
    criterion_means["G2_Total"] = g2_total
    criterion_means["Winner"] = 1 if g1_total > g2_total else 2

    criterion_results.append(criterion_means)

    # Calcular métricas agregadas (totales, medias, tasas, etc.)
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

    aggregate_results.append(aggregate_means)

# ============================
# CREAR TABLAS CONSOLIDADAS
# ============================
# Tabla por criterios
criterion_df = pd.DataFrame(criterion_results)
criterion_output_path = os.path.join(OUTPUTS_DIR, "criterion_summary.csv")
criterion_df.to_csv(criterion_output_path, index=False)

# Tabla de métricas agregadas
aggregate_df = pd.DataFrame(aggregate_results)
aggregate_output_path = os.path.join(OUTPUTS_DIR, "aggregate_summary.csv")
aggregate_df.to_csv(aggregate_output_path, index=False)

# ============================
# GENERAR GRAFICOS CONSOLIDADOS
# ============================
# Gráfico único con todos los criterios
plt.figure(figsize=(8, 6))
x = criterion_df["Test"]
width = 0.1
x_index = range(len(x))

for i, crit in enumerate(CRITERIA):
    g1_scores = criterion_df[f"G1_{crit}"].fillna(0)
    g2_scores = criterion_df[f"G2_{crit}"].fillna(0)

    plt.plot(
        x_index,
        g1_scores,
        marker="x",
        label=f"G1_{crit}",
    )
    plt.plot(x_index, g2_scores, marker="o", label=f"G2_{crit}")

plt.xticks(x_index, [label.split("_")[2] for label in x], rotation=0)
plt.yticks(fontsize=10)
plt.ylabel("Puntuación Promedio")
plt.title("Comparación de Criterios por Pruebas")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUTS_DIR, "criterion_comparison_all.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

boxplot_path = os.path.join(OUTPUTS_DIR, "total_score_distribution.png")

# ============================
# PROCESAR TESTS
# ============================
tests_files = [f for f in os.listdir(TESTS_DIR) if f.endswith("_judge_results.csv")]

data_by_group = {"G1": [], "G2": []}
labels = []

for test_file in tests_files:
    file_path = os.path.join(TESTS_DIR, test_file)
    df = pd.read_csv(file_path)

    if "judge_score_a_total" in df.columns and "judge_score_b_total" in df.columns:
        data_by_group["G1"].append(df["judge_score_a_total"].values)
        data_by_group["G2"].append(df["judge_score_b_total"].values)

        # Guardar el nombre del test
        labels.append(os.path.splitext(test_file)[0].split("_")[2])

# Crear lista de datos para el boxplot
boxplot_data = data_by_group["G1"] + data_by_group["G2"]
boxplot_labels = [f"{label} (G1)" for label in labels] + [
    f"{label} (G2)" for label in labels
]

# ============================
# GENERAR EL GRAFICO
# ============================
plt.figure()
plt.boxplot(boxplot_data, labels=boxplot_labels, vert=True)
plt.ylabel("Puntuaciones totales")
plt.title("Distribución de puntuaciones por Pruebas")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()

# Guardar el gráfico
plt.savefig(boxplot_path, dpi=300)
