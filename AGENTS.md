# Repository Guidelines

## How to Use This Guide

This guide provides comprehensive documentation for AI agents working with the LocalLLM-DataForge project. Start here to understand the project structure, development patterns, and best practices.

## Architecture Overview

LocalLLM-DataForge follows a **Data Pipeline Architecture** — the code is organized around the natural flow of data through the pipeline:

**Entrada (User Stories)** ➔ **Procesamiento LLM** ➔ **Validación/Reparación** ➔ **Salida (Tareas estructuradas)**

---

## Project Structure

```
LocalLLM-DataForge/
├── data/                          # 📊 Project data
│   ├── raw/                       #    Input datasets
│   │   ├── salony_train.csv
│   │   └── user_stories_sample.csv
│   └── outputs/                   #    Pipeline results and benchmarks
├── docs/
│   ├── USER_STORY_TO_TASKS.md
│   └── springer/                  # Research documentation and paper
│       ├── paper/
│       └── tests/
├── examples/                      # 🎯 Usage demonstrations
│   └── demo_pipeline.py
├── scripts/                       # 🔧 Execution and maintenance utilities
│   ├── clear_cache.py
│   └── run_pipeline.py
├── src/                           # 🌟 CENTRALIZED SOURCE CODE
│   └── dataforge/                 #    Main processing package
│       ├── __init__.py            #    Public API exports
│       ├── pipeline.py            #    Pipeline orchestrator
│       ├── base_step.py           #    Abstract base class for steps
│       ├── llm/                   #    🦙 LLM connectors & generation steps
│       │   ├── __init__.py
│       │   ├── ollama_step.py           # Basic LLM generation
│       │   ├── ollama_judge_step.py     # LLM-as-a-judge validation
│       │   └── comparison_judge_step.py # Dual generator comparison
│       ├── transformers/          #    🔄 Data transformation steps
│       │   ├── __init__.py
│       │   ├── load_dataframe.py        # Data loading
│       │   ├── add_column.py            # Column computation
│       │   ├── keep_columns.py          # Column selection
│       │   ├── explode_tasks.py         # Task splitting
│       │   └── json_repair.py           # JSON repair utility
│       ├── validators/            #    ✅ Business validation rules
│       │   ├── __init__.py
│       │   └── validate_user_stories.py # User story format validation
│       ├── use_cases/             #    🏗️ Client/project configurations
│       │   └── salony/
│       │       ├── __init__.py
│       │       ├── scripts/       # Salony-specific pipelines
│       │       └── config/
│       └── utils/                 #    🛠️ Shared utilities
│           ├── __init__.py
│           ├── cache.py           # Caching system
│           ├── logging.py         # Logging configuration
│           └── batching.py        # Batch processing
├── tests/                         # 🧪 Automated tests
│   ├── test_pipeline.py
│   ├── test_dual_generator.py
│   ├── test_json_repair.py
│   └── test_validate_user_stories.py
├── pyproject.toml
├── README.md
└── AGENTS.md                      # This file
```

---

## Development Setup

### Prerequisites

- Python 3.8+
- Ollama installed and running
- pip or uv package manager

### Installation

```bash
# Clone the repository
cd LocalLLM-DataForge

# Install the package in development mode (uses src/ layout)
pip install -e .

# Install all dependencies (including dev dependencies)
pip install -r requirements.txt

# Start Ollama server (required for LLM steps)
ollama serve

# Pull models used in examples
ollama pull llama3.2
ollama pull llama3.1:8b
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Run example pipeline
python examples/demo_pipeline.py

# Check package installation
python -c "import dataforge; print(dataforge.__version__)"
```

---

## Core Architecture

### DataForgePipeline

The main orchestrator (`src/dataforge/pipeline.py`) that:

- Executes steps sequentially with data flowing between them
- Manages caching using `CacheManager` for expensive LLM operations
- Supports the `>>` operator for chaining steps
- Handles logging and error recovery
- Validates column dependencies between steps

```python
from dataforge import DataForgePipeline

pipeline = DataForgePipeline(
    name="my-pipeline",
    log_level="INFO"  # DEBUG, INFO, WARNING, ERROR
)
```

### BaseStep

Abstract base class (`src/dataforge/base_step.py`) for all processing steps:

- Defines `inputs` and `outputs` properties for column validation
- Handles input/output column mappings and renaming
- Manages step lifecycle (load/unload resources)
- All steps inherit from this class

### Step Categories

#### Data Loading

- **LoadDataFrame**: Load data from DataFrame or CSV file

#### Data Transformation (`src/dataforge/transformers/`)

- **AddColumn**: Add computed column with custom function
- **KeepColumns**: Select specific columns from DataFrame
- **ExplodeTasks**: Split concatenated tasks into individual rows

#### LLM Generation (`src/dataforge/llm/`)

- **OllamaLLMStep**: Basic LLM text generation with Ollama models
- **OllamaJudgeStep**: LLM-as-a-judge validation of generated content
- **ComparisonJudgeStep**: Compares outputs from dual generators and selects the best

#### Business Validation (`src/dataforge/validators/`)

- **ValidateUserStories**: Validates user stories against Agile format ("As a... I want... so that...")

---

## Key Architecture Patterns

### Column-based Processing

Steps declare required input columns and produced output columns. The pipeline validates these dependencies automatically:

```python
class MyCustomStep(BaseStep):
    @property
    def inputs(self):
        return ["input_column"]

    @property
    def outputs(self):
        return ["output_column"]
```

### Caching System

Each step can be cached based on:

- Step configuration hash (parameters, model name, etc.)
- Input DataFrame content hash
- Step class name and version

Cache lives in `.cache/dataforge/{pipeline_name}/`

```python
# Use cache (default)
result = pipeline.run(use_cache=True)

# Skip cache
result = pipeline.run(use_cache=False)

# Clear cache
pipeline.clear_cache()
```

### Batch Processing

LLM steps process data in configurable batches to optimize Ollama performance and memory usage:

```python
ollama_step = OllamaLLMStep(
    name="generate",
    model_name="llama3.2",
    prompt_column="prompt",
    output_column="response",
    batch_size=5  # Process 5 rows at a time
)
```

### Error Handling & Retries

`OllamaLLMStep` includes built-in retry logic with exponential backoff:

```python
step = OllamaLLMStep(
    name="generate",
    model_name="llama3.2",
    prompt_column="input",
    output_column="output",
    batch_size=3,
    max_retries=3,     # Retry failed generations up to 3 times
    generation_kwargs={"temperature": 0.3, "num_predict": 1000},
)
```

### LLM-as-a-Judge Pattern

The `OllamaJudgeStep` implements quality validation for generated content:

```python
judge_step = OllamaJudgeStep(
    name="validate_content",
    model_name="llama3.1:8b",
    historia_usuario_column="input",
    tareas_generadas_column="generated_tasks",
    approval_threshold=35.0,  # Score out of 50
    batch_size=2
)
```

**Judge Validation Criteria** (0-50 points total):

- **Coherencia (0-10)**: Relevance to input requirements
- **Completitud (0-10)**: Coverage of all necessary aspects
- **Viabilidad (0-10)**: Technical feasibility
- **Formato (0-10)**: Proper structure and formatting
- **Granularidad (0-10)**: Appropriate level of detail

**Judge Output Columns**:

- `validacion_coherencia`, `validacion_completitud`, etc.: Individual criterion scores
- `validacion_total`: Total score (0-50)
- `validacion_aprobado`: Boolean approval status
- `validacion_problemas`: List of identified issues
- `validacion_recomendaciones`: Suggested improvements

### JSON Repair Utility

The `json_repair.py` module in `transformers/` handles common LLM JSON formatting issues:
- Raw newlines inside string values
- Missing commas between fields
- Trailing commas before closing braces
- Control characters in strings

```python
from dataforge.transformers.json_repair import repair_json, parse_json_with_repair

# Repair and parse in one call
result = parse_json_with_repair(llm_response)
```

---

## Pipeline Flow Example

```python
from dataforge import DataForgePipeline
from dataforge.transformers import LoadDataFrame, KeepColumns
from dataforge.llm import OllamaLLMStep

# Create pipeline
pipeline = DataForgePipeline(name="example")

# Chain steps using >> operator
(pipeline
    >> LoadDataFrame(name="load", df=data)       # Generator step
    >> OllamaLLMStep(
        name="generate",
        model_name="llama3.2",
        prompt_column="prompt",
        output_column="response"
    )
    >> KeepColumns(name="select", columns=["prompt", "response"])
)

# Run pipeline with caching
result = pipeline.run(use_cache=True)
```

---

## Creating Custom Steps

### Basic Custom Step

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
        # Your custom logic here
        return value.upper()
```

### Registering Custom Steps

Add your custom step to the appropriate category's `__init__.py`:

```python
# In src/dataforge/transformers/__init__.py
from .my_custom_step import MyCustomStep

__all__ = [
    # ... existing steps
    "MyCustomStep",
]
```

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=dataforge

# Run with coverage report
pytest tests/ --cov=dataforge --cov-report=html
```

### Testing Patterns

- **Pipeline tests** in `tests/test_pipeline.py`: Test end-to-end pipeline execution
- **JSON repair tests** in `tests/test_json_repair.py`: Test LLM output parsing
- **Validation tests** in `tests/test_validate_user_stories.py`: Test user story validation
- **Mock Ollama calls** for deterministic testing using fixtures
- **Use pytest fixtures** for common test data and configurations

### Example Test

```python
import pytest
import pandas as pd
from dataforge import DataForgePipeline
from dataforge.transformers import LoadDataFrame


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'input': ['test1', 'test2', 'test3']
    })

def test_pipeline_basic(sample_data):
    pipeline = DataForgePipeline(name="test")
    pipeline.add_step(LoadDataFrame(name="load", df=sample_data))

    result = pipeline.run(use_cache=False)

    assert len(result) == 3
    assert 'input' in result.columns
```

---

## Cache Management

### Cache Location

Caches are stored in `.cache/dataforge/{pipeline_name}/`

### Cache Keys

Cache keys include:

- Step configuration hash (all parameters)
- Input DataFrame content hash (data + columns)
- Step class name

### Cache Operations

```bash
# Clear cache for all pipelines
python scripts/clear_cache.py

# Clear cache programmatically
pipeline.clear_cache()

# Skip cache for a single run
result = pipeline.run(use_cache=False)
```

### When to Clear Cache

- After modifying step logic or parameters
- After changing input data structure
- When debugging unexpected results
- After upgrading Ollama models

---

## LLM Step Patterns

### Basic LLM Generation

```python
from dataforge.llm import OllamaLLMStep

step = OllamaLLMStep(
    name="generate",
    model_name="llama3.2",
    prompt_column="prompt",
    output_column="response",
    batch_size=5
)
```

### LLM Judge Validation

```python
from dataforge.llm import OllamaJudgeStep

judge = OllamaJudgeStep(
    name="validate",
    model_name="llama3.1:8b",
    historia_usuario_column="user_story",
    tareas_generadas_column="tasks",
    approval_threshold=35.0,  # Out of 50
    batch_size=2
)
```

### Dual Generator Comparison

```python
from dataforge.llm import ComparisonJudgeStep

comparator = ComparisonJudgeStep(
    name="compare",
    model_name="llama3.1:8b",
    input_column="user_story",
    output_a_column="tasks_model_a",
    output_b_column="tasks_model_b",
    prompt_template_func=create_prompt,
    batch_size=2
)
```

---

## Best Practices

### Pipeline Design

1. **Start simple**: Begin with basic steps, then add complexity
2. **Use caching**: Enable caching for expensive LLM operations
3. **Monitor progress**: Use appropriate log levels (INFO for production, DEBUG for development)
4. **Handle errors**: Use `max_retries` and `generation_kwargs` for robust LLM calls
5. **Validate outputs**: Use judge steps for quality control

### Step Development

1. **Inherit from BaseStep**: Always use the base class
2. **Declare dependencies**: Clearly define `inputs` and `outputs`
3. **Don't modify input**: Always copy DataFrame before modifying
4. **Add tests**: Write unit tests for custom steps
5. **Document behavior**: Add docstrings to explain step purpose

### Performance Optimization

1. **Tune batch size**: Balance memory usage and throughput
2. **Use caching**: Avoid re-running expensive operations
3. **Filter early**: Remove unnecessary rows before LLM steps
4. **Monitor resources**: Watch Ollama memory usage
5. **Profile pipelines**: Use logging to identify bottlenecks

### Testing

1. **Mock Ollama**: Don't hit real API in unit tests
2. **Use fixtures**: Share common test data across tests
3. **Test edge cases**: Empty DataFrames, missing columns, errors
4. **Integration tests**: Test full pipelines end-to-end
5. **Coverage**: Aim for >80% code coverage

---

## Common Workflows

### Creating a New Pipeline

1. Define your data structure (input DataFrame)
2. Choose or create necessary steps
3. Chain steps using `>>` operator
4. Run with caching enabled
5. Validate results
6. Add judge validation if needed

### Adding a New Step Type

1. Create new file in the appropriate category (`llm/`, `transformers/`, or `validators/`)
2. Inherit from `BaseStep`
3. Implement `inputs`, `outputs`, and `process()`
4. Add to the category's `__init__.py`
5. Write tests in `tests/`
6. Update documentation

### Debugging Pipeline Issues

1. Enable DEBUG logging
2. Run without cache (`use_cache=False`)
3. Check step inputs/outputs in logs
4. Verify column names match expectations
5. Test steps individually
6. Check Ollama server status and logs

---

## Troubleshooting

### Common Issues

**Issue**: Pipeline fails with "Column not found"

- **Solution**: Check `inputs` and `outputs` of each step match your DataFrame columns

**Issue**: Ollama connection errors

- **Solution**: Ensure `ollama serve` is running, check Ollama logs

**Issue**: Cache not invalidating

- **Solution**: Run `python scripts/clear_cache.py` or modify step parameters

**Issue**: Out of memory with large datasets

- **Solution**: Reduce batch size, process in chunks, or upgrade hardware

**Issue**: Judge validation scores too low

- **Solution**: Adjust approval threshold, improve prompts, use better judge model

**Issue**: LLM output JSON parsing fails

- **Solution**: Use `json_repair.py` utilities; check if output is truncated

---

## Version History

- **v0.3.0**: Architecture restructured to Data Pipeline pattern (`src/` layout, categorized steps, JSON repair utility)
- **v0.2.0**: Added dual generator support and comparison judge
- **v0.1.0**: Initial release with basic pipeline and Ollama integration

---

## Contributing

Contributions are welcome! To add features:

1. Fork the repository
2. Create a feature branch
3. Add your code and tests
4. Update documentation
5. Submit a pull request

---

## Support

- **Issues**: [GitHub Issues](https://github.com/alvaldes/LocalLLM-DataForge/issues)
- **Documentation**: See `examples/`, `docs/`, and `AGENTS.md`
- **Examples**: Run pipelines in `src/dataforge/use_cases/salony/scripts/`

---

## Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Distilabel (Inspiration)](https://github.com/argilla-io/distilabel)
- [pytest Documentation](https://docs.pytest.org/)

---

## Acknowledgments

Inspired by [Distilabel](https://github.com/argilla-io/distilabel) by Argilla.
