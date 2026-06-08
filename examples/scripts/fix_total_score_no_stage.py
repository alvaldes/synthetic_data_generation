import pandas as pd

# Cargar el CSV
input_path = "examples/tests/salony_dual_test4_output_judge_results.csv"
output_path = "examples/tests/salony_dual_test4_output_judge_results.csv"

df = pd.read_csv(input_path)

# Columnas para A y B
a_cols = [
    "judge_score_a_coherence",
    "judge_score_a_completeness",
    "judge_score_a_feasibility",
    "judge_score_a_format",
    "judge_score_a_granularity",
]

b_cols = [
    "judge_score_b_coherence",
    "judge_score_b_completeness",
    "judge_score_b_feasibility",
    "judge_score_b_format",
    "judge_score_b_granularity",
]

# Recalcular totales (sobrescribiendo)
df["judge_score_a_total"] = df[a_cols].sum(axis=1)
df["judge_score_b_total"] = df[b_cols].sum(axis=1)

# Guardar resultado
df.to_csv(output_path, index=False)

print("Listo: totales recalculados y archivo guardado en:", output_path)
