# LocalLLM-DataForge

A simplified Distilabel-inspired pipeline for synthetic dataset creation using pandas DataFrames and local Ollama models.

## 🚀 Quick Start

```bash
# 1. Clone/create the project
cd synthetic_data_generation

# 2. Install dependencies
pip install -e .

# 3. Make sure Ollama is running
ollama serve

# 4. Pull a model (if not already done)
ollama pull llama3.2

# 5. Run example
python examples/instruction_pipeline.py

# 6. Run tests
pytest tests/ -v

# 7. Clear cache
python scripts/clear_cache.py
```

## 📖 Features

- 🐼 **Native pandas DataFrame support** - Work with familiar data structures
- 🦙 **Ollama local model integration** - Run LLMs locally without API costs
- 🔄 **Step-based processing architecture** - Modular and extensible design
- 💾 **Automatic caching** - Speed up iterations with intelligent caching
- 🔁 **Batch processing** - Efficient processing of large datasets
- 🛠️ **Easy to extend** - Create custom steps with simple API
- 🔒 **Robust error handling** - Continue processing even when individual rows fail
- 📊 **Rich logging** - Track pipeline execution with detailed logs

## 🏗️ Architecture

```
DataForgePipeline
├── LoadDataFrame (load data)
├── FilterRows (filter data)
├── SortRows (sort data)
├── SampleRows (sample data)
├── OllamaLLMStep (generate with LLM)
├── RobustOllamaStep (generate with error handling)
├── AddColumn (transform)
└── KeepColumns (select columns)
```

## 📚 Examples

### Basic Pipeline

```python
import pandas as pd
from dataforge import DataForgePipeline
from dataforge.steps import LoadDataFrame, OllamaLLMStep

# Create data
df = pd.DataFrame({
    'topic': ['Python', 'Machine Learning', 'Web Development']
})

# Create pipeline
pipeline = DataForgePipeline(name="basic-example")

# Add steps
pipeline.add_step(LoadDataFrame(name="load", df=df))
pipeline.add_step(OllamaLLMStep(
    name="explain",
    model_name="llama3.2",
    prompt_column="topic",
    output_column="explanation",
    batch_size=3
))

# Run
result = pipeline.run()
print(result)
```

### Advanced Pipeline with Transformations

```python
from dataforge.steps import FilterRows, SortRows, RobustOllamaStep

pipeline = DataForgePipeline(name="advanced")

# Filter, sort, and generate
pipeline.add_step(LoadDataFrame(name="load", df=data))
pipeline.add_step(FilterRows(
    name="filter",
    filter_column="priority",
    condition="> 5"
))
pipeline.add_step(SortRows(name="sort", by="priority", ascending=False))
pipeline.add_step(RobustOllamaStep(
    name="generate",
    model_name="llama3.2",
    prompt_column="topic",
    output_column="content",
    save_failures=True,  # Save failed rows
    continue_on_error=True  # Don't stop on errors
))

result = pipeline.run(use_cache=True)
```

### Using the >> Operator

```python
# Chain steps with >> for cleaner syntax
pipeline = DataForgePipeline(name="chain")

(pipeline
    >> LoadDataFrame(name="load", df=data)
    >> FilterRows(name="filter", filter_func=lambda r: r['score'] > 5)
    >> OllamaLLMStep(name="generate", model_name="llama3.2", ...)
    >> KeepColumns(name="keep", columns=["topic", "generation"])
)

result = pipeline.run()
```

## 🔧 Available Steps

### Data Loading

- **LoadDataFrame**: Load data from DataFrame or CSV

### Data Transformation

- **FilterRows**: Filter rows by condition or function
- **SortRows**: Sort by one or more columns
- **SampleRows**: Take random sample (n rows or fraction)
- **AddColumn**: Add computed column
- **KeepColumns**: Select specific columns

### LLM Generation

- **OllamaLLMStep**: Generate text with Ollama models
- **RobustOllamaStep**: Generate with error handling and metrics

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
        # Your custom logic here
        return value.upper()
```

## 📊 Caching

Caching is automatic and based on:

- Step configuration
- Input data hash
- Step name and parameters

```python
# Use cache (default)
result = pipeline.run(use_cache=True)

# Skip cache
result = pipeline.run(use_cache=False)

# Clear cache
pipeline.clear_cache()
```

## 🐛 Error Handling

Use `RobustOllamaStep` for production pipelines:

```python
robust_step = RobustOllamaStep(
    name="generate",
    model_name="llama3.2",
    prompt_column="input",
    output_column="output",
    save_failures=True,  # Save failed rows to CSV
    failure_dir="./failures",  # Where to save failures
    continue_on_error=True,  # Don't stop on errors
    max_retries=3  # Retry failed generations
)

result = pipeline.run()

# Check metrics
print(f"Success rate: {robust_step.success_count} / {robust_step.success_count + robust_step.failure_count}")

# Get failure summary
summary = robust_step.get_failure_summary()
print(summary)
```

## 📝 Logging

Configure logging level:

```python
pipeline = DataForgePipeline(
    name="my-pipeline",
    log_level="DEBUG"  # DEBUG, INFO, WARNING, ERROR
)
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=dataforge
```

## 📁 Project Structure

```
synthetic_data_generation/
├── dataforge/           # Core library
│   ├── __init__.py
│   ├── base_step.py          # Base step class
│   ├── pipeline.py           # Pipeline orchestrator
│   ├── steps/                # Built-in steps
│   │   ├── load_dataframe.py
│   │   ├── ollama_step.py
│   │   ├── robust_ollama.py
│   │   ├── filter_rows.py
│   │   ├── sort_rows.py
│   │   ├── sample_rows.py
│   │   ├── keep_columns.py
│   │   └── add_column.py
│   └── utils/                # Utilities
│       ├── cache.py
│       ├── logging.py
│       └── batching.py
├── examples/                 # Example pipelines
│   ├── instruction_pipeline.py
│   ├── model_comparison.py
│   └── advanced_pipeline.py
├── notebooks/               # Jupyter notebooks
│   └── demo.ipynb
├── tests/                   # Test suite
│   ├── test_pipeline.py
│   └── test_steps.py
├── scripts/                 # CLI scripts
│   ├── run_pipeline.py
│   └── clear_cache.py
└── data/                    # Data directory
    └── .gitkeep
```

## 🤝 Contributing

Contributions are welcome! To add a new step:

1. Create a new file in `dataforge/steps/`
2. Inherit from `BaseStep`
3. Implement `inputs`, `outputs`, and `process()`
4. Add to `dataforge/steps/__init__.py`
5. Write tests in `tests/`

## 📄 License

MIT License

## 🙏 Acknowledgments

Inspired by [Distilabel](https://github.com/argilla-io/distilabel) by Argilla.

## 📞 Support

- Issues: [GitHub Issues](https://github.com/alvaldes/synthetic_data_generation/issues)
- Documentation: See `examples/` and `notebooks/`

## 🎯 Roadmap

- [ ] Support for async Ollama calls
- [ ] Multiprocessing for parallel step execution
- [ ] More built-in steps (GroupBy, Merge, etc.)
- [ ] Web UI for pipeline monitoring
- [ ] Export to Hugging Face datasets
- [ ] Integration with other LLM providers
