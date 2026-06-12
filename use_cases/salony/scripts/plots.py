import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# =========================
# VALIDACIÓN DE ARGUMENTOS
# =========================
if len(sys.argv) < 2:
    print("Uso: python plots.py <archivo.csv> [--show]")
    sys.exit(1)

FILE = sys.argv[1]
SHOW = "--show" in sys.argv

# =========================
# CONFIG
# =========================
CRITERIA = [
    "coherence",
    "completeness",
    "feasibility",
    "format",
    "granularity",
]

LABELS = [
    "Coherencia",
    "Completitud",
    "Viabilidad",
    "Formato",
    "Granularidad",
]

PREFIX_A = "judge_score_a_"
PREFIX_B = "judge_score_b_"

COL_G1 = "judge_score_a_total"
COL_G2 = "judge_score_b_total"

# =========================
# LOAD
# =========================
df = pd.read_csv(FILE)

# =========================
# OUTPUT PATH (mismo dir del script)
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

barplot_path = os.path.join(SCRIPT_DIR, "barplot_criteria.png")
boxplot_path = os.path.join(SCRIPT_DIR, "boxplot_scores.png")

# =========================
# BARPLOT (criterios)
# =========================
g1_means = [df[PREFIX_A + c].mean() for c in CRITERIA]
g2_means = [df[PREFIX_B + c].mean() for c in CRITERIA]

x = range(len(CRITERIA))
width = 0.35

plt.figure()
plt.bar([i - width / 2 for i in x], g1_means, width, label="G1")
plt.bar([i + width / 2 for i in x], g2_means, width, label="G2")

plt.xticks(x, LABELS, rotation=20)
plt.ylabel("Puntuación promedio")
plt.title("Comparación por criterio")
plt.legend()

plt.tight_layout()
plt.savefig(barplot_path, dpi=300)

if SHOW:
    plt.show()

plt.close()

# =========================
# BOXPLOT (scores)
# =========================
plt.figure()

plt.boxplot([df[COL_G1], df[COL_G2]], labels=["G1", "G2"])

plt.ylabel("Puntuación total")
plt.title("Distribución de puntuaciones")

plt.tight_layout()
plt.savefig(boxplot_path, dpi=300)

if SHOW:
    plt.show()

plt.close()

# =========================
# DONE
# =========================
print("Gráficas generadas:")
print(f"- {barplot_path}")
print(f"- {boxplot_path}")
