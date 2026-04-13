import pandas as pd
import sys

# =========================
# VALIDACIÓN DE ARGUMENTOS
# =========================
if len(sys.argv) != 2:
    print("Uso: python script_criterios.py <archivo.csv>")
    sys.exit(1)

FILE = sys.argv[1]

CRITERIA = [
    "completeness",
    "clarity",
    "actionability",
    "logical_structure",
    "granularity",
]

PREFIX_A = "judge_score_a_"
PREFIX_B = "judge_score_b_"

nice_names = {
    "completeness": "Completitud",
    "clarity": "Claridad",
    "actionability": "Accionabilidad",
    "logical_structure": "Estructura lógica",
    "granularity": "Granularidad",
}

# =========================
# LOAD
# =========================
df = pd.read_csv(FILE)

# =========================
# COMPUTE
# =========================
results = {}

for c in CRITERIA:
    col_a = PREFIX_A + c
    col_b = PREFIX_B + c

    results[c] = {"g1": df[col_a].mean(), "g2": df[col_b].mean()}

# =========================
# OUTPUT
# =========================
print("=== PROMEDIO POR CRITERIO ===")

for c, vals in results.items():
    print(f"{c}: G1={vals['g1']:.2f}, G2={vals['g2']:.2f}")

rows = ""
for c in CRITERIA:
    rows += f"{nice_names[c]} & {results[c]['g1']:.2f} & {results[c]['g2']:.2f} \\\\\n"

latex = f"""
\\begin{{tabular}}{{lcc}}
\\toprule
Criterio & G1 & G2 \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
"""

print("\n=== LATEX TABLE ===")
print(latex)
