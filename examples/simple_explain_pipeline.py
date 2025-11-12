# examples/simple_explain_pipeline.py

import pandas as pd
from typing import Dict
from pydantic import BaseModel
from typing import Optional

from simple_pipeline.pipeline import SimplePipeline
from simple_pipeline.steps.load_dataframe import LoadDataFrame
from simple_pipeline.steps.ollama_step import OllamaLLMStep
from simple_pipeline.steps.keep_columns import KeepColumns

# --------------------------
# 1. Datos iniciales
# --------------------------
data = pd.DataFrame({
    "concept": ["Machine Learning", "Blockchain", "Quantum Computing"],
    "audience": ["beginner", "intermediate", "advanced"]
})

# --------------------------
# 2. Crear pipeline
# --------------------------
pipeline = SimplePipeline(
    name="concept-explanation",
    description="Genera explicaciones de conceptos técnicos para diferentes audiencias",
    log_level="DEBUG"
)

# Step 1: cargar datos
load_step = LoadDataFrame(name="load_data", df=data)

# Step 2: generar explicaciones
def explanation_prompt_template(row: Dict) -> str:
    return f"Explain {row['concept']} to a {row['audience']} audience in 2-3 sentences."

generate_explanation = OllamaLLMStep(
    name="generate_explanation",
    # model_name="deepseek-r1:8b",
    model_name="llama3.1:8b",
    prompt_column="concept",
    output_column="explanation",
    prompt_template=explanation_prompt_template,
    system_prompt="You are a clear, concise technical educator.",
    batch_size=5,
    generation_kwargs = {
        "temperature": 0.3,
        "num_predict": 200,
        "top_p": 0.9,
        "seed": 42,
        "repeat_penalty": 1.2,
        "stop": ["</think>", "\nuser:"]
    }
)

# Step 3: quedarnos solo con las columnas relevantes
keep_cols = KeepColumns(
    name="keep_columns",
    columns=["concept", "audience", "explanation"]
)

# --------------------------
# 3. Pipeline construction
# --------------------------
pipeline.add_step(load_step)
pipeline.add_step(generate_explanation)
pipeline.add_step(keep_cols)

# --------------------------
# 4. Execution
# --------------------------
if __name__ == "__main__":
    result_df = pipeline.run(use_cache=False)

    print("\n" + "="*50)
    print("FINAL DATASET")
    print("="*50)
    print(result_df)

    # Guardar resultado
    result_df.to_csv("examples/explanation_dataset.csv", index=False)
    print("\n✓ Saved to examples/explanation_dataset.csv")