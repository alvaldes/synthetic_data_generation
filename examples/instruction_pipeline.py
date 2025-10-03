# examples/instruction_pipeline.py

import pandas as pd
from typing import Dict

from simple_pipeline.pipeline import SimplePipeline
from simple_pipeline.steps.load_dataframe import LoadDataFrame
from simple_pipeline.steps.ollama_step import OllamaLLMStep
from simple_pipeline.steps.keep_columns import KeepColumns

# --------------------------
# 1. Datos iniciales
# --------------------------
data = pd.DataFrame({
    "topic": ["Python", "Machine Learning", "Web Development", "Data Science"],
    "difficulty": ["beginner", "intermediate", "beginner", "advanced"]
})

# --------------------------
# 2. Crear pipeline
# --------------------------
pipeline = SimplePipeline(
    name="instruction-generation",
    description="Genera instrucciones y respuestas para temas de programación"
)

# Step 1: cargar datos
load_step = LoadDataFrame(name="load_data", df=data)

# Step 2: generar instrucciones
def instruction_prompt_template(row: Dict) -> str:
    return f"Generate a {row['difficulty']} level instruction or question about {row['topic']}."

generate_instruction = OllamaLLMStep(
    name="generate_instruction",
    model_name="deepseek-r1:8b",
    prompt_column="topic",
    output_column="instruction",
    prompt_template=instruction_prompt_template,
    system_prompt="You are an expert educator creating programming exercises.",
    batch_size=2,
    generation_kwargs={"temperature": 0.7, "num_predict": 100}
)

# Step 3: generar respuestas
def response_prompt_template(row: Dict) -> str:
    return f"Provide a detailed answer to this instruction: {row['instruction']}"

generate_response = OllamaLLMStep(
    name="generate_response",
    model_name="deepseek-r1:8b",
    prompt_column="instruction",
    output_column="response",
    prompt_template=response_prompt_template,
    system_prompt="You are a helpful programming instructor.",
    batch_size=2,
    generation_kwargs={"temperature": 0.6, "num_predict": 300}
)

# Step 4: quedarnos solo con las columnas relevantes
keep_cols = KeepColumns(
    name="keep_columns",
    columns=["topic", "difficulty", "instruction", "response"]
)

# --------------------------
# 3. Construcción del pipeline
# --------------------------
pipeline.add_step(load_step)
pipeline.add_step(generate_instruction)
pipeline.add_step(generate_response)
pipeline.add_step(keep_cols)

# --------------------------
# 4. Ejecución
# --------------------------
if __name__ == "__main__":
    result_df = pipeline.run(use_cache=True)

    print("\n" + "="*50)
    print("FINAL DATASET")
    print("="*50)
    print(result_df)

    # Guardar resultado
    result_df.to_csv("examples/instruction_dataset.csv", index=False)
    print("\n✓ Saved to examples/instruction_dataset.csv")