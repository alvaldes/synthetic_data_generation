# Simple Pipeline

A simplified Distilabel-inspired pipeline for synthetic dataset creation using pandas DataFrames and local Ollama models.

## 🚀 Quick Start

```bash
# 1. Clonar/crear el proyecto
cd simple_pipeline_project

# 2. Instalar en modo desarrollo
pip install -e .

# 3. Ejecutar ejemplo
python examples/instruction_pipeline.py

# 4. Ejecutar tests
pytest tests/ -v

# 5. Limpiar caché
python scripts/clear_cache.py
```

## 📖 Features

- 🐼 Native pandas DataFrame support
- 🦙 Ollama local model integration
- 🔄 Step-based processing architecture
- 💾 Automatic caching
- 🔁 Batch processing
- 🛠️ Easy to extend with custom steps

## 🏗️ Architecture

```
SimplePipeline
├── LoadDataFrame (load data)
├── OllamaLLMStep (generate with LLM)
├── AddColumn (transform)
└── KeepColumns (filter)
```

## 📚 Documentation

See `examples/` for complete usage examples.
