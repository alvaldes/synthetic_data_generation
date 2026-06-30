# LocalLLM-DataForge

**Data Pipeline Architecture** for decomposing user stories into development tasks using local LLMs with Ollama.

Inspired by [Distilabel](https://github.com/argilla-io/distilabel), it organizes processing around the natural data flow: **Input ➔ LLM Processing ➔ Validation/Repair ➔ Output**.

---

## 🚀 Quick Start

```bash
# 1. Install the package
pip install -e .

# 2. Install dev/test dependencies
pip install -r requirements.txt

# 3. Make sure Ollama is running
ollama serve

# 4. Pull a model
ollama pull llama3.2

# 5. Run example
python examples/demo_pipeline.py

# 6. Run tests
pytest tests/ -v
```

---

## 🐳 Docker

The project includes **Docker Compose** to run the pipeline alongside Ollama in containers, without needing to install Python or Ollama on the host.

### Services

| Service | Image | Role |
|---------|-------|------|
| **ollama** | `ollama/ollama:latest` | Local LLM inference service |
| **app** | Local build from `Dockerfile` | Python pipeline execution |

Communication via the internal `dataforge_network` bridge. The app uses `OLLAMA_HOST=http://ollama:11434` to point to the service.

### Prerequisites

- Docker Engine 24+ with Docker Compose v2
- (Optional) GPU acceleration — depends on your hardware:
  - **NVIDIA**: `nvidia-container-toolkit` installed on the host
  - **AMD**: `rocm` and device configuration in compose
  - **Apple Silicon (macOS)**: Ollama uses Metal by default, the GPU is shared with the host with no extra configuration

### Basic Usage

```bash
# Start services (ollama + app)
docker compose up -d

# Tail logs from both services
docker compose logs -f
```

### Pulling Models

Ollama starts without models. You need to download them from the container:

```bash
docker compose exec localllm-dataforge-ollama ollama pull llama3.2
docker compose exec localllm-dataforge-ollama ollama pull llama3.1:8b
docker compose exec localllm-dataforge-ollama ollama pull qwen3:8b
```

### Running Pipelines

Once services are running, use `exec` (interactive mode):

```bash
# Example pipeline
docker compose exec localllm-dataforge-app python examples/demo_pipeline.py

# Real pipeline with Salony dataset (single generator)
docker compose exec localllm-dataforge-app python src/dataforge/use_cases/salony/scripts/salony_single_generator_pipeline.py \
  /app/data/outputs/result.csv \
  --model llama3.1:8b \
  --batch-size 4

# Dual generator pipeline
docker compose exec localllm-dataforge-app python src/dataforge/use_cases/salony/scripts/salony_dual_generator_pipeline.py \
  /app/data/outputs/result.csv \
  --model-a llama3.1:8b \
  --model-b qwen3:8b \
  --judge-model llama3.1:8b
```

For one-off commands without keeping the container running:

```bash
docker compose run --rm localllm-dataforge-app python examples/demo_pipeline.py
```

### GPU Acceleration

Ollama inside the container can use the host GPU. Configuration depends on your hardware.

#### NVIDIA (example)

If you have an NVIDIA GPU with CUDA and `nvidia-container-toolkit` installed, uncomment the GPU lines under the `ollama` service in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Verify Docker has access:

```bash
docker compose exec localllm-dataforge-ollama nvidia-smi
```

#### Other GPUs

The same principle applies for **AMD** (configuring ROCm devices in compose) or **Apple Silicon** (Ollama uses Metal automatically — on macOS the GPU is shared with the host, no extra Docker configuration needed).

> **⚠️ Important**: GPU acceleration is entirely optional. The pipeline works perfectly on CPU — it's just slower with large models. If you don't have a GPU or prefer not to configure it, simply ignore this section.

### Volumes

| Volume | Container Mount | Purpose |
|--------|-----------------|---------|
| `dataforge_ollama_models` | `/root/.ollama` | Downloaded models (persistent across restarts) |
| `dataforge_pipeline_cache` | `/app/.cache` | Pipeline cache (persistent) |
| `./data` (bind mount) | `/app/data` | Input/output data — shared with the host |

Input CSVs go in `data/raw/` and results are written to `data/outputs/`, accessible from the host immediately.

### Stopping

```bash
# Bring services down (volumes persist)
docker compose down

# Bring services down and remove volumes (deletes downloaded models and cache)
docker compose down -v
```

---

## 📖 Features

- 🐼 **Native pandas DataFrame support** — Work with familiar data structures
- 🦙 **Ollama local model integration** — Run LLMs locally without API costs
- 🔄 **Step-based processing architecture** — Modular and extensible design
- 💾 **Automatic caching** — Speed up iterations with intelligent caching
- 🔁 **Parallel batch processing** — Concurrent row execution via `ThreadPoolExecutor`
- 🛠️ **Easy to extend** — Create custom steps with simple API
- ⚖️ **LLM-as-a-judge validation** — Quality control with judge models
- 🔄 **Dual generator comparison** — Compare two models and pick the best
- 🔧 **JSON repair** — Automatic recovery from malformed LLM outputs

---

## 🏗️ Architecture

```
                  ┌─────────────────────────────────────┐
                  │         DataForgePipeline            │
                  │  (Sequential step orchestrator)      │
                  └─────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌──────────────────┐     ┌──────────────┐
│  transformers/ │     │     llm/         │     │  validators/ │
│  (load &       │     │  (generation &   │     │  (business   │
│   transform)   │     │   evaluation)    │     │   rules)     │
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

### Chaining Steps with `>>`

You can chain steps in two equivalent ways. The explicit API:

```python
pipeline = DataForgePipeline(name="example")
pipeline.add_step(LoadDataFrame(name="load", df=df))
pipeline.add_step(OllamaLLMStep(name="generate", ...))
```

Or with the `>>` operator (syntactic sugar for `add_step()`):

```python
pipeline = DataForgePipeline(name="example")

(pipeline
    >> LoadDataFrame(name="load", df=df)
    >> OllamaLLMStep(name="generate", ...)
)
```

Both are equivalent. The `>>` operator returns the pipeline, enabling fluent chaining.

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
    batch_size=8,
    num_workers=4         # concurrent requests within the batch
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

## 🏗️ Real-World Use Cases

The `src/dataforge/use_cases/` directory contains complete, runnable pipelines for real datasets.

### Salony Dataset

Pipelines for decomposing user stories from the [Salony](data/raw/salony_train.csv) dataset into development tasks.

| Script | Description |
|--------|-------------|
| **`salony_single_generator_pipeline.py`** | Full pipeline with a single generator + optional judge |
| **`salony_dual_generator_pipeline.py`** | Pipeline with two generators + ComparisonJudge to pick the best |

#### Single Generator

```bash
python src/dataforge/use_cases/salony/scripts/salony_single_generator_pipeline.py output.csv \
  --model llama3.1:8b \
  --batch-size 4 \
  --use-judge
```

#### Dual Generator

```bash
python src/dataforge/use_cases/salony/scripts/salony_dual_generator_pipeline.py output.csv \
  --model-a llama3.1:8b \
  --model-b qwen3:8b \
  --judge-model llama3.1:8b
```

#### Results Analysis

| Script | Description |
|--------|-------------|
| **`aggregate_metrics.py`** | Aggregated metrics: mean, std, pass rate, win rate (with LaTeX output) |
| **`criterion_breakdown.py`** | Score breakdown by criterion (coherence, completeness, etc.) |
| **`plots.py`** | Visualizations: boxplots, score comparison between generators |
| **`consolidate_results.py`** | Consolidate multiple test runs into a unified report with charts |
| **`fix_total_score_no_stage.py`** | Utility to recalculate totals when the judge skipped them |

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
| **OllamaLLMStep** | Generate text with Ollama models (with retry logic and exponential backoff) |
| **OllamaJudgeStep** | Validate generated content using LLM-as-a-judge (5 criteria, 0-50 scale) |
| **ComparisonJudgeStep** | Compare outputs from dual generators, score both, select the best |

### Business Validation (`dataforge.validators`)

| Step | Description |
|------|-------------|
| **ValidateUserStories** | Filter user stories matching Agile format ("As a... I want... so that...") |

### Utilities (`dataforge.utils`)

| Module | Description |
|--------|-------------|
| **CacheManager** | Automatic pipeline step caching with content-hash keys |
| **setup_logger** | Configure consistent logging across steps (timestamped, named loggers) |
| **batch_dataframe** | Split DataFrames into batches for memory-efficient processing |
| **split_dataframe** | Split DataFrames for parallel processing |
| **get_num_batches** | Calculate number of batches given a batch size |

### JSON Repair (`dataforge.transformers.json_repair`)

| Function | Description |
|----------|-------------|
| **clean_json_response** | Extract JSON from LLM responses (removes markdown fences, finds boundaries) |
| **repair_json** | Fix common JSON errors: escape newlines in strings, add missing commas, remove trailing commas |
| **parse_json_with_repair** | Parse with automatic fallback: try direct parse, then repair and retry |

---

## ⚖️ LLM-as-a-Judge: Validation Criteria

`OllamaJudgeStep` evaluates each generated task against 5 criteria. Each criterion scores 0 to 10, for a total of 0 to 50 points.

| Criterion | Range | What it measures |
|-----------|-------|------------------|
| **Coherence** | 0-10 | Are tasks directly related to the story? Any irrelevant tasks? |
| **Completeness** | 0-10 | Do they cover all necessary aspects? Is anything critical missing? |
| **Feasibility** | 0-10 | Are they technically achievable? Any impossible or illogical steps? |
| **Format** | 0-10 | Does each task have a clear title, description, and proper structure? |
| **Granularity** | 0-10 | Is the level of detail appropriate? Too broad or too atomic? |

**Output columns** generated by the judge:

| Column | Description |
|--------|-------------|
| `validacion_coherencia` | Score 0-10 |
| `validacion_completitud` | Score 0-10 |
| `validacion_viabilidad` | Score 0-10 |
| `validacion_formato` | Score 0-10 |
| `validacion_granularidad` | Score 0-10 |
| `validacion_total` | Total score (0-50) |
| `validacion_aprobado` | Boolean (`True` if total >= threshold) |
| `validacion_problemas` | List of critical issues detected |
| `validacion_recomendaciones` | Improvement suggestions |

```python
judge = OllamaJudgeStep(
    name="validate_tasks",
    model_name="llama3.1:8b",
    historia_usuario_column="input",
    tareas_generadas_column="tasks",
    approval_threshold=35.0,   # Minimum 35/50 to pass
    batch_size=2,
    generation_kwargs={"temperature": 0.2},  # Low temperature = consistent judgment
)
```

---

## 🔧 JSON Repair: Robust LLM Output Handling

LLMs frequently return malformed JSON. The `json_repair` module automatically solves the most common issues.

### Issues It Fixes

| Issue | Example | Fix |
|-------|---------|-----|
| Raw newlines in strings | `"desc": "line 1\nline 2"` | Escapes to `\n` |
| Missing commas between fields | `} "key"` | Adds `,` |
| Trailing commas | `},]` | Removes the extra comma |
| Markdown fences | `` ```json `` | Automatic cleanup |
| Control characters | tabs, null bytes | Escapes or removes |

### Usage

```python
from dataforge.transformers.json_repair import repair_json, parse_json_with_repair

# Clean and extract JSON from a response
cleaned = clean_json_response(llm_response)

# Fix common errors
repaired = repair_json(cleaned)

# Parse with automatic fallback (tries direct, then repairs)
result = parse_json_with_repair(llm_response)
if result is None:
    print("Could not parse even after repair")
```

---

## ⚡ Performance Tuning

### `batch_size` + `num_workers`: How to Tune Throughput

Each LLM step accepts two parameters that control performance:

| Parameter | Default (LLM / Judge) | What It Controls |
|-----------|----------------------|------------------|
| `batch_size` | 8 / 4 | Rows loaded per batch in memory. Also caps maximum concurrency. |
| `num_workers` | 1 | Concurrent requests to Ollama **within** each batch. |

**Golden rule:** `batch_size >= num_workers`. If `num_workers > batch_size`, extra workers are never used.

**Is `batch_size` a placebo?** No, but its role changes with `num_workers`:
- **Without parallelism** (`num_workers=1`): controls memory and cache granularity
- **With parallelism** (`num_workers > 1`): acts as a concurrency cap

### Hardware Guide

Recommended starting values for **7B-8B models** (llama3.1:8b, qwen3:8b, deepseek-r1:8b):

| Hardware | `num_workers` | `batch_size` | Notes |
|----------|:------------:|:------------:|-------|
| **CPU only** (8GB RAM) | 1 | 4-8 | CPU queues requests sequentially. Parallelism yields no real gain. |
| **CPU only** (16GB+ RAM) | 1-2 | 4-8 | With 2 workers it can help if you have multiple physical cores. |
| **GPU 4GB VRAM** | 1 | 2-4 | No room for more than 1 model in VRAM. Keep batches small. |
| **GPU 6GB VRAM** | 2 | 4 | Good starting point for 7B models with Q4 quantization. |
| **GPU 8GB VRAM** | 2-3 | 4-6 | Ollama can keep 1-2 concurrent GPU requests. |
| **GPU 12GB VRAM** | 4 | 8 | **Sweet spot** for 7B-8B. GPU fully utilized without saturation. |
| **GPU 16GB VRAM** | 4-6 | 8 | Can run 13B models while maintaining parallelism. |
| **GPU 24GB+ VRAM** | 6-8 | 8-12 | Multiple requests fit in VRAM. Increase gradually and monitor. |

### Recommendations by Model Size

| Model | Examples | `num_workers` | Notes |
|-------|----------|:------------:|-------|
| **1B-3B** | `llama3.2:3b`, `qwen3:4b`, `phi4-mini` | 4-8 | Small models, lots of concurrency possible. Don't saturate VRAM. |
| **7B-8B** | `llama3.1:8b`, `qwen3:8b`, `deepseek-r1:8b` | 2-4 | The sweet spot. Good quality without consuming too much VRAM. |
| **13B-14B** | `llama3.1:14b`, `qwen3:14b` | 1-3 | Each request uses more VRAM. Reduce workers compared to 7B. |
| **30B+** | `qwen3:30b`, `llama3.1:70b` | 1-2 | Large models. Need 24GB+ VRAM. Minimal parallelism. |

### How to Tune

```python
# GPU 12GB VRAM + 7B-8B model → sweet spot
generator = OllamaLLMStep(
    name="generate",
    model_name="qwen3:8b",
    prompt_column="prompt",
    output_column="tasks",
    batch_size=8,
    num_workers=4,
)

# GPU 8GB VRAM + 13B model → conservative
judge = OllamaJudgeStep(
    name="validate",
    model_name="llama3.1:14b",
    historia_usuario_column="input",
    tareas_generadas_column="tasks",
    batch_size=4,
    num_workers=2,
)
```

### Fine-tuning with `ollama ps`

```bash
# While the pipeline is running, monitor:
ollama ps

# If you see a queue of unprocessed requests → increase num_workers
# If you see CUDA OOM errors → decrease num_workers or batch_size
# If output is slow but GPU at 100% → num_workers is sufficient
```

**Strategy:** Start with `num_workers=2` and increase by 1 while monitoring with `ollama ps`. When you hit memory errors or latency stops improving, that's your limit.

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
│       ├── use_cases/       #   🏗️ Real-world pipeline implementations
│       │   └── salony/      #       Salony dataset pipelines & analysis
│       └── utils/           #   🛠️ Shared utilities (cache, logging, batching)
├── tests/                   # Automated tests
├── docs/                    # Documentation
│   ├── USER_STORY_TO_TASKS.md
│   └── springer/            # Academic paper and research tests
├── README.md
└── AGENTS.md                # AI agent guide
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your code and tests
4. Update documentation
5. Submit a pull request

**Where to start?** Check out the pipelines in `src/dataforge/use_cases/` to see real examples of how complete pipelines are built.

---

## 📄 License

MIT License

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/alvaldes/LocalLLM-DataForge/issues)
- **Documentation**: See `examples/`, `docs/`, and `AGENTS.md`
- **Real pipelines**: Explore `src/dataforge/use_cases/salony/scripts/`
