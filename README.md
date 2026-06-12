# LocalLLM-DataForge

**Data Pipeline Architecture** para descomposición de historias de usuario en tareas de desarrollo usando LLMs locales con Ollama.

Inspirado en [Distilabel](https://github.com/argilla-io/distilabel), organiza el procesamiento siguiendo el flujo natural de los datos: **Entrada ➔ Procesamiento LLM ➔ Validación/Reparación ➔ Salida**.

---

## 🚀 Quick Start

```bash
# 1. Install the package
pip install -e .

# 2. Make sure Ollama is running
ollama serve

# 3. Pull a model
ollama pull llama3.2

# 4. Run example
python examples/demo_pipeline.py

# 5. Run tests
pytest tests/ -v
```

---

## 📖 Features

- 🐼 **Native pandas DataFrame support** — Work with familiar data structures
- 🦙 **Ollama local model integration** — Run LLMs locally without API costs
- 🔄 **Step-based processing architecture** — Modular and extensible design
- 💾 **Automatic caching** — Speed up iterations with intelligent caching
- 🔁 **Batch processing** — Efficient processing of large datasets
- 🛠️ **Easy to extend** — Create custom steps with simple API
- ⚖️ **LLM-as-a-judge validation** — Quality control with judge models
- 🔧 **JSON repair** — Automatic recovery from malformed LLM outputs

---

## 🏗️ Architecture

```
                  ┌─────────────────────────────────────┐
                  │         DataForgePipeline            │
                  │  (Orquestador de pasos secuenciales) │
                  └─────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌──────────────────┐     ┌──────────────┐
│  transformers/ │     │     llm/         │     │  validators/ │
│  (carga y      │     │  (generación y   │     │  (reglas de  │
│   transforma)  │     │   evaluación)    │     │   negocio)   │
├───────────────┤     ├──────────────────┤     ├──────────────┤
│ LoadDataFrame │     │ OllamaLLMStep    │     │ValidateUser  │
│ AddColumn     │     │ OllamaJudgeStep  │     │Stories       │
│ KeepColumns   │     │ ComparisonJudge  │     └──────────────┘
│ ExplodeTasks  │     └──────────────────┘
│ json_repair   │
└───────────────┘
```

---

## 📚 Examples

### Basic Pipeline

```python
import pandas as pd
from dataforge import DataForgePipeline
from dataforge.transformers import LoadDataFrame
from dataforge.llm import OllamaLLMStep

# Create data
df = pd.DataFrame({
    'input': ['As a user, I want to login so that I can access my account']
})

# Create pipeline
pipeline = DataForgePipeline(name="basic-example")

# Add steps
pipeline.add_step(LoadDataFrame(name="load", df=df))
pipeline.add_step(OllamaLLMStep(
    name="generate",
    model_name="llama3.2",
    prompt_column="input",
    output_column="tasks",
    batch_size=3
))

# Run
result = pipeline.run()
print(result)
```

### Pipeline with Validation

```python
from dataforge import DataForgePipeline
from dataforge.transformers import LoadDataFrame, AddColumn, ExplodeTasks
from dataforge.llm import OllamaLLMStep, OllamaJudgeStep
from dataforge.validators import ValidateUserStories

pipeline = DataForgePipeline(name="salony-pipeline")

(pipeline
    >> LoadDataFrame(name="load", df=stories_df)
    >> ValidateUserStories(name="validate", story_column="input")
    >> AddColumn(name="add_id", input_columns=[], output_column="us_id", func=lambda: counter())
    >> OllamaLLMStep(name="generate", model_name="llama3.2", ...)
    >> OllamaJudgeStep(name="judge", model_name="llama3.1:8b", ...)
    >> ExplodeTasks(name="explode", tasks_column="tasks", output_column="task")
)

result = pipeline.run(use_cache=True)
```

---

## 🔧 Available Steps

### Data Loading & Transformation (`dataforge.transformers`)

| Step | Description |
|------|-------------|
| **LoadDataFrame** | Load data from DataFrame or CSV |
| **AddColumn** | Add computed column with custom function |
| **KeepColumns** | Select specific columns |
| **ExplodeTasks** | Split concatenated numbered tasks into rows |

### LLM Generation & Evaluation (`dataforge.llm`)

| Step | Description |
|------|-------------|
| **OllamaLLMStep** | Generate text with Ollama models (with retry logic) |
| **OllamaJudgeStep** | Validate generated content using LLM-as-a-judge |
| **ComparisonJudgeStep** | Compare outputs from dual generators, select best |

### Business Validation (`dataforge.validators`)

| Step | Description |
|------|-------------|
| **ValidateUserStories** | Filter user stories matching Agile format ("As a... I want... so that...") |

### Utilities

| Module | Description |
|--------|-------------|
| **json_repair** | Fix common JSON issues from LLM outputs (newlines, missing commas, etc.) |
| **CacheManager** | Automatic pipeline step caching with content-hash keys |
| **batching** | DataFrame batch iteration utilities |

---

## 🎨 Creating Custom Steps

```python
from dataforge.base_step import BaseStep
import pandas as pd


class MyCustomStep(BaseStep):
    def __init__(self, name: str, param: str, **kwargs):
        super().__init__(name, **kwargs)
        self.param = param

    @property
    def inputs(self):
        return ["input_column"]

    @property
    def outputs(self):
        return ["output_column"]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["output_column"] = df["input_column"].apply(
            lambda x: self.transform(x)
        )
        return df

    def transform(self, value):
        return value.upper()
```

---

## 📊 Caching

Caching is automatic and based on step configuration + input data hash:

```python
# Use cache (default)
result = pipeline.run(use_cache=True)

# Skip cache
result = pipeline.run(use_cache=False)

# Clear cache
pipeline.clear_cache()
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=dataforge
```

---

## 📁 Project Structure

```
LocalLLM-DataForge/
├── data/                    # Input datasets and pipeline outputs
│   ├── raw/                 #   CSV inputs (user stories, etc.)
│   └── outputs/             #   Pipeline results and benchmarks
├── examples/                # Usage demonstrations
│   └── demo_pipeline.py
├── scripts/                 # CLI utilities
│   ├── clear_cache.py
│   └── run_pipeline.py
├── src/                     # 🌟 Source code
│   └── dataforge/           #   Main package
│       ├── pipeline.py      #   Pipeline orchestrator
│       ├── base_step.py     #   Abstract step base class
│       ├── llm/             #   🦙 LLM generation and validation steps
│       ├── transformers/    #   🔄 Data transformation steps
│       ├── validators/      #   ✅ Business rule validators
│       ├── use_cases/       #   🏗️ Client-specific configurations
│       └── utils/           #   🛠️ Shared utilities
├── tests/                   # Automated tests
└── docs/                    # Documentation
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your code and tests
4. Update documentation
5. Submit a pull request

---

## 📄 License

MIT License

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/alvaldes/LocalLLM-DataForge/issues)
- **Documentation**: See `examples/`, `docs/`, and `AGENTS.md`
