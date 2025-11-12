# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Install the package in development mode
pip install -e .

# Install all dependencies (including dev dependencies)
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=simple_pipeline
```

### Pipeline Management
```bash
# Clear cache for all pipelines
python scripts/clear_cache.py

# Run example pipelines
python examples/instruction_pipeline.py
python examples/advanced_pipeline.py
python examples/model_comparison.py

# Run Salony user story to tasks pipeline (with optional judge validation)
python examples/salony_pipeline.py output.csv
python examples/salony_pipeline.py output.csv --use-judge --judge-threshold 40
python examples/salony_pipeline.py output.csv --sample 10  # For testing
```

### Ollama Setup
```bash
# Start Ollama server (required for LLM steps)
ollama serve

# Pull models used in examples
ollama pull llama3.2
```

## Architecture Overview

This is a **DataFrame-based pipeline processing library** inspired by Distilabel, designed for synthetic data generation using local Ollama LLMs.

### Core Components

**SimplePipeline** (`simple_pipeline/pipeline.py`): Main orchestrator that:
- Executes steps sequentially with data flowing between them
- Manages caching using `CacheManager` for expensive LLM operations
- Supports the `>>` operator for chaining steps
- Handles logging and error recovery

**BaseStep** (`simple_pipeline/base_step.py`): Abstract base class for all processing steps:
- Defines `inputs` and `outputs` properties for column validation
- Handles input/output column mappings and renaming
- Manages step lifecycle (load/unload resources)
- All steps inherit from this class

**Step Categories**:
- **Data Loading**: `LoadDataFrame` - loads data from DataFrame or CSV
- **Data Transformation**: `FilterRows`, `SortRows`, `SampleRows`, `AddColumn`, `KeepColumns`
- **LLM Generation**: `OllamaLLMStep`, `RobustOllamaStep` (with error handling and retry logic)
- **LLM Validation**: `OllamaJudgeStep` - validates generated content using LLM-as-a-judge pattern

### Key Architecture Patterns

**Column-based Processing**: Steps declare required input columns and produced output columns. The pipeline validates these dependencies automatically.

**Caching System**: Each step can be cached based on:
- Step configuration hash
- Input DataFrame content hash
- Step class and parameters
Cache lives in `.cache/simple_pipeline/{pipeline_name}/`

**Error Handling**: `RobustOllamaStep` provides production-ready LLM processing with:
- Automatic retries for failed generations
- Saving failed rows to CSV files for debugging
- Success/failure metrics tracking
- Continue-on-error mode

**Batch Processing**: LLM steps process data in configurable batches to optimize Ollama performance.

### Pipeline Flow Example
```python
# Typical pipeline structure
pipeline = SimplePipeline(name="example")

(pipeline
    >> LoadDataFrame(name="load", df=data)           # Generator step
    >> FilterRows(name="filter", condition="...")     # Transform step
    >> OllamaLLMStep(name="generate", model="...")     # LLM step
    >> KeepColumns(name="select", columns=["..."])    # Output step
)

result = pipeline.run(use_cache=True)
```

## Important Implementation Notes

### Creating New Steps
- Inherit from `BaseStep`
- Define `inputs` and `outputs` properties as List[str]
- Implement `process(self, df: pd.DataFrame) -> pd.DataFrame`
- Add to `simple_pipeline/steps/__init__.py` for imports

### LLM Step Patterns
- Use `OllamaLLMStep` for basic LLM generation
- Use `RobustOllamaStep` for production pipelines with error handling
- Use `OllamaJudgeStep` for LLM-as-a-judge validation of generated content
- Always specify `prompt_column` (input) and `output_column` (generated text)
- Configure `batch_size` based on model size and memory constraints

### LLM Judge Validation Pattern
The `OllamaJudgeStep` implements LLM-as-a-judge for quality validation:

```python
# Basic judge validation setup
pipeline.add_step(
    OllamaJudgeStep(
        name="validate_content",
        model_name="llama3.1:8b",
        historia_usuario_column="input",
        tareas_generadas_column="generated_tasks",
        approval_threshold=35.0,  # Score out of 50
        batch_size=2
    )
)
```

**Judge Validation Criteria**:
- **Coherencia (0-10)**: Relevance to input requirements
- **Completitud (0-10)**: Coverage of all necessary aspects
- **Viabilidad (0-10)**: Technical feasibility
- **Formato (0-10)**: Proper structure and formatting
- **Granularidad (0-10)**: Appropriate level of detail

**Judge Output Columns**:
- `validacion_*`: Individual criterion scores
- `validacion_total`: Total score (0-50)
- `validacion_aprobado`: Boolean approval status
- `validacion_problemas`: List of identified issues
- `validacion_recomendaciones`: Suggested improvements

### Cache Management
- Cache keys include step configuration AND input data hash
- Use `pipeline.clear_cache()` or `scripts/clear_cache.py` to reset
- Cache is automatically used unless `use_cache=False` in `pipeline.run()`

### Testing Patterns
- Pipeline tests in `tests/test_pipeline.py`
- Step-specific tests in `tests/test_steps.py`
- Mock Ollama calls for deterministic testing