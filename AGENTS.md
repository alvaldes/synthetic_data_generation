# Repository Guidelines

## How to Use This Guide

This guide provides comprehensive documentation for AI agents working with the LocalLLM-DataForge project. Start here to understand the project structure, development patterns, and best practices.

## Available Skills

Use these skills for detailed patterns on-demand:

### Generic Skills (Any Project)

| Skill | Description | URL |
|-------|-------------|-----|
| `typescript` | Const types, flat interfaces, utility types | [SKILL.md](skills/typescript/SKILL.md) |
| `python` | Type hints, dataclasses, async patterns | [SKILL.md](skills/python/SKILL.md) |
| `pytest` | Fixtures, mocking, markers, parametrize | [SKILL.md](skills/pytest/SKILL.md) |

### LocalLLM-DataForge-Specific Skills

| Skill | Description | URL |
|-------|-------------|-----|
| `dataforge` | Project overview, architecture patterns | [SKILL.md](skills/dataforge/SKILL.md) |
| `pipeline-design` | Creating and chaining pipeline steps | [SKILL.md](skills/pipeline-design/SKILL.md) |
| `ollama-steps` | LLM generation with Ollama | [SKILL.md](skills/ollama-steps/SKILL.md) |
| `judge-validation` | LLM-as-a-judge validation patterns | [SKILL.md](skills/judge-validation/SKILL.md) |
| `cache-management` | Pipeline caching strategies | [SKILL.md](skills/cache-management/SKILL.md) |
| `custom-steps` | Creating custom pipeline steps | [SKILL.md](skills/custom-steps/SKILL.md) |
| `error-handling` | Robust error handling in pipelines | [SKILL.md](skills/error-handling/SKILL.md) |

### Auto-invoke Skills

When performing these actions, ALWAYS invoke the corresponding skill FIRST:

| Action | Skill |
|--------|-------|
| Creating new pipeline steps | `custom-steps` |
| Working with Ollama LLM generation | `ollama-steps` |
| Implementing LLM-as-a-judge validation | `judge-validation` |
| Designing multi-step pipelines | `pipeline-design` |
| Debugging cache issues | `cache-management` |
| Adding error handling and retries | `error-handling` |
| Writing tests with pytest | `pytest` |
| Working with type hints and dataclasses | `python` |
| General project architecture questions | `dataforge` |

---

## Project Overview

LocalLLM-DataForge is a DataFrame-based pipeline processing library inspired by Distilabel, designed for synthetic data generation using local Ollama LLMs. It provides a modular, extensible framework for building data processing pipelines with built-in caching, error handling, and LLM integration.

### Key Features

- 🐼 **Native pandas DataFrame support** - Work with familiar data structures
- 🦙 **Ollama local model integration** - Run LLMs locally without API costs
- 🔄 **Step-based processing architecture** - Modular and extensible design
- 💾 **Automatic caching** - Speed up iterations with intelligent caching
- 🔁 **Batch processing** - Efficient processing of large datasets
- 🛠️ **Easy to extend** - Create custom steps with simple API
- 🔒 **Robust error handling** - Continue processing even when individual rows fail
- ⚖️ **LLM-as-a-judge validation** - Quality control with judge models

---

## Project Structure

```
LocalLLM-DataForge/
├── dataforge/           # Core library
│   ├── __init__.py
│   ├── base_step.py          # Base step class
│   ├── pipeline.py           # Pipeline orchestrator
│   ├── steps/                # Built-in steps
│   │   ├── __init__.py
│   │   ├── load_dataframe.py      # Data loading
│   │   ├── ollama_step.py         # Basic LLM generation
│   │   ├── ollama_judge_step.py   # LLM-as-a-judge validation
│   │   ├── comparison_judge_step.py # Dual generator comparison
│   │   ├── keep_columns.py        # Column selection
│   │   └── add_column.py          # Column transformation
│   └── utils/                # Utilities
│       ├── cache.py          # Caching system
│       ├── logging.py        # Logging configuration
│       └── batching.py       # Batch processing
├── examples/                 # Example pipelines
│   ├── instruction_pipeline.py
│   ├── salony_pipeline.py
│   └── salony_dual_generator_pipeline.py
├── tests/                   # Test suite
│   ├── test_pipeline.py
│   └── test_dual_generator.py
├── scripts/                 # CLI scripts
│   ├── run_pipeline.py
│   └── clear_cache.py
├── docs/                    # Documentation
│   └── USER_STORY_TO_TASKS.md
├── pyproject.toml           # Project metadata
├── requirements.txt         # Dependencies
├── CLAUDE.md               # Claude-specific guidance
└── AGENTS.md               # This file
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

# Install the package in development mode
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
python examples/instruction_pipeline.py

# Check package installation
python -c "import dataforge; print(dataforge.__version__)"
```

---

## Core Architecture

### DataForgePipeline

The main orchestrator (`dataforge/pipeline.py`) that:

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

Abstract base class (`dataforge/base_step.py`) for all processing steps:

- Defines `inputs` and `outputs` properties for column validation
- Handles input/output column mappings and renaming
- Manages step lifecycle (load/unload resources)
- All steps inherit from this class

### Step Categories

#### Data Loading Steps

- **LoadDataFrame**: Load data from DataFrame or CSV file

#### Data Transformation Steps

- **AddColumn**: Add computed column with custom function
- **KeepColumns**: Select specific columns from DataFrame

#### LLM Generation Steps

- **OllamaLLMStep**: Basic LLM text generation with Ollama models
- **RobustOllamaStep**: Production-ready generation with error handling and retry logic

#### LLM Validation Steps

- **OllamaJudgeStep**: Validates generated content using LLM-as-a-judge pattern
- **ComparisonJudgeStep**: Compares outputs from dual generators

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

### Error Handling

`RobustOllamaStep` provides production-ready LLM processing with:

- Automatic retries for failed generations (configurable max_retries)
- Saving failed rows to CSV files for debugging
- Success/failure metrics tracking
- Continue-on-error mode to process entire dataset

```python
robust_step = RobustOllamaStep(
    name="generate",
    model_name="llama3.2",
    prompt_column="input",
    output_column="output",
    save_failures=True,
    failure_dir="./failures",
    continue_on_error=True,
    max_retries=3
)
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

### LLM-as-a-Judge Pattern

The `OllamaJudgeStep` implements quality validation:

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

---

## Pipeline Flow Example

```python
from dataforge import DataForgePipeline
from dataforge.steps import LoadDataFrame, OllamaLLMStep, KeepColumns

# Create pipeline
pipeline = DataForgePipeline(name="example")

# Chain steps using >> operator
(pipeline
    >> LoadDataFrame(name="load", df=data)           # Generator step
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

Add your custom step to `dataforge/steps/__init__.py`:

```python
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
- **Step-specific tests** in `tests/test_steps.py`: Test individual step functionality
- **Mock Ollama calls** for deterministic testing using fixtures
- **Use pytest fixtures** for common test data and configurations

### Example Test

```python
import pytest
import pandas as pd
from dataforge import DataForgePipeline
from dataforge.steps import LoadDataFrame

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
from dataforge.steps import OllamaLLMStep

step = OllamaLLMStep(
    name="generate",
    model_name="llama3.2",
    prompt_column="prompt",
    output_column="response",
    batch_size=5
)
```

### Production LLM Generation with Error Handling

```python
from dataforge.steps import RobustOllamaStep

step = RobustOllamaStep(
    name="generate",
    model_name="llama3.2",
    prompt_column="prompt",
    output_column="response",
    batch_size=3,
    save_failures=True,
    failure_dir="./failures",
    continue_on_error=True,
    max_retries=3
)

# After running, check metrics
print(f"Success: {step.success_count}")
print(f"Failures: {step.failure_count}")
print(step.get_failure_summary())
```

### LLM Judge Validation

```python
from dataforge.steps import OllamaJudgeStep

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
from dataforge.steps import ComparisonJudgeStep

comparator = ComparisonJudgeStep(
    name="compare",
    model_name="llama3.1:8b",
    historia_usuario_column="user_story",
    tareas_a_column="tasks_model_a",
    tareas_b_column="tasks_model_b",
    batch_size=2
)
```

---

## Best Practices

### Pipeline Design

1. **Start simple**: Begin with basic steps, then add complexity
2. **Use caching**: Enable caching for expensive LLM operations
3. **Monitor progress**: Use appropriate log levels (INFO for production, DEBUG for development)
4. **Handle errors**: Use `RobustOllamaStep` for production pipelines
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

1. Create new file in `dataforge/steps/`
2. Inherit from `BaseStep`
3. Implement `inputs`, `outputs`, and `process()`
4. Add to `dataforge/steps/__init__.py`
5. Write tests in `tests/test_steps.py`
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

---

## Version History

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
- **Documentation**: See `CLAUDE.md`, `examples/`, and `docs/`
- **Examples**: Run pipelines in `examples/` directory

---

## Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Distilabel (Inspiration)](https://github.com/argilla-io/distilabel)
- [pytest Documentation](https://docs.pytest.org/)

---

## Acknowledgments

Inspired by [Distilabel](https://github.com/argilla-io/distilabel) by Argilla.
