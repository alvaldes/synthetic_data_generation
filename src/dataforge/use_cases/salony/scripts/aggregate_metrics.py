"""
Compute aggregate metrics from a dual-generator judge results CSV.

Usage:
    python aggregate_metrics.py path/to/judge_results.csv

Configuration (threshold, column names) is loaded from the centralised
DataForge settings.
"""

import pandas as pd
import sys
from dataforge.config import get_settings

# =========================================================================
# Config
# =========================================================================

cfg = get_settings(use_case="salony")
THRESHOLD = cfg.judge.approval_threshold
COL_G1 = "judge_score_a_total"
COL_G2 = "judge_score_b_total"
COL_WINNER = "judge_winner"

# =========================================================================
# Args
# =========================================================================

if len(sys.argv) != 2:
    print("Uso: python aggregate_metrics.py <archivo.csv>")
    sys.exit(1)

FILE = sys.argv[1]

# =========================================================================
# Load & compute
# =========================================================================

df = pd.read_csv(FILE)

results = {}

results["G1"] = {
    "mean": df[COL_G1].mean(),
    "std": df[COL_G1].std(),
    "pass_rate": (df[COL_G1] >= THRESHOLD).mean() * 100,
    "win_rate": (df[COL_WINNER] == "A").mean() * 100,
}

results["G2"] = {
    "mean": df[COL_G2].mean(),
    "std": df[COL_G2].std(),
    "pass_rate": (df[COL_G2] >= THRESHOLD).mean() * 100,
    "win_rate": (df[COL_WINNER] == "B").mean() * 100,
}

# =========================================================================
# Output
# =========================================================================

print("=== RESULTADOS AGREGADOS ===")

for g in ["G1", "G2"]:
    print(f"\n{g}:")
    print(f"  Mean: {results[g]['mean']:.2f}")
    print(f"  Std: {results[g]['std']:.2f}")
    print(f"  Pass Rate (%): {results[g]['pass_rate']:.2f}")
    print(f"  Win Rate (%): {results[g]['win_rate']:.2f}")

latex = f"""
\\begin{{tabular}}{{lcc}}
\\toprule
Métrica & G1 & G2 \\\\
\\midrule
Puntuación promedio & {results["G1"]["mean"]:.2f} & {results["G2"]["mean"]:.2f} \\\\
Desviación estándar & {results["G1"]["std"]:.2f} & {results["G2"]["std"]:.2f} \\\\
Tasa de aprobación (\\%) & {results["G1"]["pass_rate"]:.2f} & {results["G2"]["pass_rate"]:.2f} \\\\
Tasa de victoria (\\%) & {results["G1"]["win_rate"]:.2f} & {results["G2"]["win_rate"]:.2f} \\\\
\\bottomrule
\\end{{tabular}}
"""

print("\n=== LATEX TABLE ===")
print(latex)
