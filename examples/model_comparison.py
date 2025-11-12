# examples/model_comparison.py

import pandas as pd

from simple_pipeline.pipeline import SimplePipeline
from simple_pipeline.steps.load_dataframe import LoadDataFrame
from simple_pipeline.steps.ollama_step import OllamaLLMStep

# --------------------------
# 1. Datos iniciales
# --------------------------
data = pd.DataFrame({
    "prompt": [
        "Explain quantum computing in simple terms",
        "What are the benefits of functional programming?",
        "How does a neural network work?"
    ]
})

# --------------------------
# 2. Crear pipeline
# --------------------------
pipeline = SimplePipeline(name="model-comparison")

# Step 1: cargar datos
load_step = LoadDataFrame(name="load", df=data)

# Step 2: generar con Llama
gen_llama = OllamaLLMStep(
    name="llama3",
    model_name="llama3.2",
    prompt_column="prompt",
    output_column="llama_response",
    batch_size=3,
    output_mappings={"model_name": "llama_model"}
)

# Step 3: generar con Mistral
gen_mistral = OllamaLLMStep(
    name="mistral",
    model_name="mistral",
    prompt_column="prompt",
    output_column="mistral_response",
    batch_size=3,
    output_mappings={"model_name": "mistral_model"}
)

# Note: here the steps run sequentially,
# but they could be parallelized in a more advanced version.

# --------------------------
# 3. Pipeline construction
# --------------------------
pipeline.add_step(load_step)
pipeline.add_step(gen_llama)
pipeline.add_step(gen_mistral)

# --------------------------
# 4. Execution
# --------------------------
if __name__ == "__main__":
    result = pipeline.run(use_cache=True)

    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    print(result)

    # Guardar resultados
    result.to_csv("model_comparison.csv", index=False)
    print("\n✓ Saved to model_comparison.csv")