# LocalLLM-DataForge

A framework for synthetic dataset creation using pandas DataFrames and local Ollama models.

## 🚀 How to run it

```bash
# 1. Install dependencies
pip install -e .

# 2. Make sure Ollama is running
ollama serve

# 3. Pull a model (if not already done)
ollama pull llama3.1:8b

# 4. Download or use the Salony dataset in data/ folder
# If you want to download it go to https://huggingface.co/datasets/salony/User_story

# 5. Run Salony script with model specification
# python script.py output.csv --model-a [model-name] --model-b [model-name] --judge-model [model-name] --no-cache

python scripts/salony_dual_generator_pipeline.py [output-file-name].csv --model-a [model-name] --model-b [model-name] --judge-model [model-name] --no-cache
```
