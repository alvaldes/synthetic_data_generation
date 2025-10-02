# Building a Distilabel-Inspired Pipeline: Implementation Guide

## Executive Summary

Distilabel is a sophisticated AI Feedback framework built on a **DAG-based pipeline architecture** with **multiprocessing execution**, **batch-oriented data flow**, and **modular Step abstractions**. This guide reverse-engineers its core concepts to create a simplified pipeline system specifically designed for **pandas DataFrames** and **Ollama local models**. The implementation captures Distilabel's essential patterns—pipeline context managers, step abstraction, batch processing, and LLM integration—while streamlining for DataFrame workflows and local inference.

---

## 1. Core Architecture: Understanding Distilabel's Design

### Pipeline as Directed Acyclic Graph

**Central concept**: Distilabel represents pipelines internally as a DAG where **Steps are nodes** and **connections are edges**. This architecture enables parallel execution, prevents circular dependencies, and provides clear data flow semantics.

**Key architectural decisions**:
- **Multiprocessing over threading**: Each step runs in a separate subprocess to bypass Python's GIL and achieve true parallelism
- **Queue-based communication**: Steps communicate via multiprocessing queues with a shared output queue and per-step input queues
- **Batch-oriented processing**: Data flows as configurable-size batches (lists of dictionaries) rather than row-by-row
- **Stage-based execution**: Steps can load/unload in stages for memory efficiency with large models

### Three-Layer Component Model

**1. Pipeline Layer** - Orchestration and lifecycle management
- Creates and manages the DAG structure
- Handles validation, execution, and state management
- Implements caching and recovery mechanisms
- Coordinates batch flow between steps

**2. Step Layer** - Processing abstraction
- **Step**: Base class for data transformation (receives batches, processes, yields results)
- **GeneratorStep**: Root nodes that generate or load data without predecessors
- **GlobalStep**: Aggregation nodes that receive all data at once
- **Task**: Specialized Step subclass that integrates LLMs

**3. LLM Layer** - Model abstraction
- Unified interface across providers (OpenAI, Anthropic, Hugging Face, **Ollama**, etc.)
- Synchronous `generate()` and asynchronous `agenerate()` methods
- Structured output support via Pydantic schemas
- Automatic batching for async LLMs

### Data Flow Model

**Batch Structure**:
```python
_Batch = {
    "seq_no": int,              # Sequence number for ordering
    "step_name": str,           # Originating step
    "last_batch": bool,         # Final batch flag
    "data": List[Dict],         # Actual data (list of row dictionaries)
    "data_path": Optional[str], # Filesystem path for large data
}
```

**Row Structure**: Each dictionary represents one dataset row where keys are column names and values are the data. Steps add, remove, or modify keys as data flows through the pipeline.

**Processing Flow**:
```
GeneratorStep → Batch → Step.process() → Transform → Yield → 
BatchManager → Route to Downstream Steps → Next Step Input Queue
```

---

## 2. Simplified Architecture Design for DataFrames + Ollama

### Design Philosophy

**Simplifications from Distilabel**:
- Replace multiprocessing with sequential execution (simpler, easier to debug)
- Use pandas DataFrames as the native data structure instead of list-of-dicts batches
- Focus on Ollama integration exclusively (remove provider abstraction complexity)
- Implement in-memory processing without filesystem buffering
- Remove DAG validation, just use simple linear or branching flows

**Retained core concepts**:
- Step abstraction with inputs/outputs contracts
- Pipeline context manager pattern
- Batch processing for efficiency
- Caching and recovery mechanisms
- Column mapping between steps

### Core Architecture Blueprint

```
SimplePipeline
├── Steps: List[BaseStep]
├── Connections: Dict[str, List[str]]
├── Cache: Dict[str, pd.DataFrame]
└── execute() → pd.DataFrame

BaseStep (Abstract)
├── inputs: List[str]
├── outputs: List[str]
├── process(df: pd.DataFrame) → pd.DataFrame
└── load() / unload()

OllamaLLMStep (Extends BaseStep)
├── model_name: str
├── client: ollama.Client
├── prompt_template: Callable
├── batch_size: int
└── process(df: pd.DataFrame) → pd.DataFrame
```

---

## 3. Implementation Blueprint

### Class Hierarchy and Structure

#### Base Step Class

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import pandas as pd

class BaseStep(ABC):
    """Abstract base class for all pipeline steps."""
    
    def __init__(
        self,
        name: str,
        input_mappings: Optional[Dict[str, str]] = None,
        output_mappings: Optional[Dict[str, str]] = None,
        cache: bool = True
    ):
        self.name = name
        self.input_mappings = input_mappings or {}
        self.output_mappings = output_mappings or {}
        self.cache = cache
        self._loaded = False
    
    @property
    @abstractmethod
    def inputs(self) -> List[str]:
        """Required input column names."""
        pass
    
    @property
    @abstractmethod
    def outputs(self) -> List[str]:
        """Output column names this step produces."""
        pass
    
    def load(self) -> None:
        """Initialize resources (e.g., load models, connect to APIs)."""
        self._loaded = True
    
    def unload(self) -> None:
        """Cleanup resources."""
        self._loaded = False
    
    def _apply_input_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns according to input_mappings."""
        if self.input_mappings:
            df = df.rename(columns=self.input_mappings)
        return df
    
    def _apply_output_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns according to output_mappings."""
        if self.output_mappings:
            df = df.rename(columns=self.output_mappings)
        return df
    
    def _validate_inputs(self, df: pd.DataFrame) -> None:
        """Check required input columns exist."""
        missing = set(self.inputs) - set(df.columns)
        if missing:
            raise ValueError(
                f"Step '{self.name}' missing required inputs: {missing}. "
                f"Available: {list(df.columns)}"
            )
    
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the DataFrame and return transformed result."""
        pass
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the step with validation and mappings."""
        if not self._loaded:
            self.load()
        
        df = self._apply_input_mappings(df)
        self._validate_inputs(df)
        df = self.process(df)
        df = self._apply_output_mappings(df)
        
        return df
```

#### Ollama LLM Integration

```python
import ollama
from typing import Callable, List, Dict, Any, Optional
from tqdm import tqdm
import time

class OllamaLLMStep(BaseStep):
    """Step that calls Ollama for LLM inference on DataFrame rows."""
    
    def __init__(
        self,
        name: str,
        model_name: str,
        prompt_column: str,
        output_column: str = "generation",
        system_prompt: Optional[str] = None,
        prompt_template: Optional[Callable[[Dict], str]] = None,
        batch_size: int = 8,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        ollama_host: str = "http://localhost:11434",
        max_retries: int = 3,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.model_name = model_name
        self.prompt_column = prompt_column
        self.output_column = output_column
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        self.generation_kwargs = generation_kwargs or {}
        self.ollama_host = ollama_host
        self.max_retries = max_retries
        self.client = None
    
    @property
    def inputs(self) -> List[str]:
        return [self.prompt_column]
    
    @property
    def outputs(self) -> List[str]:
        return [self.output_column, "model_name"]
    
    def load(self) -> None:
        """Initialize Ollama client."""
        self.client = ollama.Client(host=self.ollama_host)
        super().load()
    
    def unload(self) -> None:
        """Cleanup Ollama client."""
        self.client = None
        super().unload()
    
    def _format_prompt(self, row: Dict[str, Any]) -> str:
        """Format prompt using template or direct column."""
        if self.prompt_template:
            return self.prompt_template(row)
        return row[self.prompt_column]
    
    def _generate_with_retry(
        self,
        prompt: str,
        retry_count: int = 0
    ) -> Optional[str]:
        """Call Ollama with retry logic."""
        try:
            messages = []
            
            if self.system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=False,
                **self.generation_kwargs
            )
            
            return response['message']['content']
            
        except Exception as e:
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                time.sleep(wait_time)
                return self._generate_with_retry(prompt, retry_count + 1)
            else:
                print(f"Error generating for prompt after {self.max_retries} retries: {e}")
                return None
    
    def _process_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of rows through Ollama."""
        results = []
        
        for _, row in batch_df.iterrows():
            prompt = self._format_prompt(row.to_dict())
            generation = self._generate_with_retry(prompt)
            results.append({
                self.output_column: generation,
                "model_name": self.model_name
            })
        
        # Add results as new columns
        result_df = pd.DataFrame(results, index=batch_df.index)
        return pd.concat([batch_df, result_df], axis=1)
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame in batches through Ollama."""
        results = []
        
        # Process in batches
        num_batches = (len(df) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=len(df), desc=f"Processing {self.name}") as pbar:
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(df))
                batch = df.iloc[start_idx:end_idx]
                
                processed_batch = self._process_batch(batch)
                results.append(processed_batch)
                
                pbar.update(len(batch))
        
        return pd.concat(results, ignore_index=False)
```

#### Simple Data Loader Step

```python
class LoadDataFrame(BaseStep):
    """Generator step that loads a pandas DataFrame."""
    
    def __init__(
        self,
        name: str,
        df: Optional[pd.DataFrame] = None,
        csv_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.df = df
        self.csv_path = csv_path
    
    @property
    def inputs(self) -> List[str]:
        return []  # Generator has no inputs
    
    @property
    def outputs(self) -> List[str]:
        return list(self.df.columns) if self.df is not None else []
    
    def load(self) -> None:
        """Load DataFrame from CSV if path provided."""
        if self.csv_path and self.df is None:
            self.df = pd.read_csv(self.csv_path)
        super().load()
    
    def process(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Return the loaded DataFrame."""
        return self.df.copy()
```

#### Column Management Steps

```python
class KeepColumns(BaseStep):
    """Keep only specified columns."""
    
    def __init__(self, name: str, columns: List[str], **kwargs):
        super().__init__(name, **kwargs)
        self.columns = columns
    
    @property
    def inputs(self) -> List[str]:
        return self.columns
    
    @property
    def outputs(self) -> List[str]:
        return self.columns
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.columns].copy()


class AddColumn(BaseStep):
    """Add a computed column using a function."""
    
    def __init__(
        self,
        name: str,
        input_columns: List[str],
        output_column: str,
        func: Callable,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.input_columns = input_columns
        self.output_column = output_column
        self.func = func
    
    @property
    def inputs(self) -> List[str]:
        return self.input_columns
    
    @property
    def outputs(self) -> List[str]:
        return [self.output_column]
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.output_column] = df.apply(
            lambda row: self.func(*[row[col] for col in self.input_columns]),
            axis=1
        )
        return df
```

#### Pipeline Class

```python
from typing import List, Optional, Dict, Any
from pathlib import Path
import pickle
import hashlib
import json

class SimplePipeline:
    """Simplified pipeline for DataFrame processing with LLMs."""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        cache_dir: str = ".cache/simple_pipeline"
    ):
        self.name = name
        self.description = description
        self.cache_dir = Path(cache_dir) / name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.steps: List[BaseStep] = []
        self._step_outputs: Dict[str, pd.DataFrame] = {}
    
    def add_step(self, step: BaseStep) -> 'SimplePipeline':
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self
    
    def __rshift__(self, step: BaseStep) -> 'SimplePipeline':
        """Allow step1 >> step2 syntax."""
        return self.add_step(step)
    
    def _get_cache_key(self, step: BaseStep, input_hash: str) -> str:
        """Generate cache key for a step."""
        step_config = {
            "name": step.name,
            "class": step.__class__.__name__,
            "inputs": step.inputs,
            "outputs": step.outputs,
        }
        step_str = json.dumps(step_config, sort_keys=True)
        combined = f"{step_str}_{input_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_df_hash(self, df: pd.DataFrame) -> str:
        """Generate hash of DataFrame for caching."""
        return hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()[:16]
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load cached DataFrame."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _save_to_cache(self, cache_key: str, df: pd.DataFrame) -> None:
        """Save DataFrame to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
    
    def run(
        self,
        input_df: Optional[pd.DataFrame] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Execute the pipeline."""
        
        # Start with input or first step
        if input_df is not None:
            current_df = input_df.copy()
        else:
            # First step should be a generator (like LoadDataFrame)
            if not self.steps:
                raise ValueError("Pipeline has no steps")
            
            first_step = self.steps[0]
            print(f"Loading data from {first_step.name}...")
            current_df = first_step(pd.DataFrame())  # Empty df for generators
            self._step_outputs[first_step.name] = current_df
            steps_to_process = self.steps[1:]
        
        # Process remaining steps
        for step in (self.steps if input_df else steps_to_process):
            print(f"\nExecuting step: {step.name}")
            
            # Check cache
            input_hash = self._get_df_hash(current_df)
            cache_key = self._get_cache_key(step, input_hash)
            
            if use_cache and step.cache:
                cached_df = self._load_from_cache(cache_key)
                if cached_df is not None:
                    print(f"  ✓ Loaded from cache")
                    current_df = cached_df
                    self._step_outputs[step.name] = current_df
                    continue
            
            # Execute step
            try:
                current_df = step(current_df)
                self._step_outputs[step.name] = current_df
                
                # Save to cache
                if use_cache and step.cache:
                    self._save_to_cache(cache_key, current_df)
                
                print(f"  ✓ Complete ({len(current_df)} rows, {len(current_df.columns)} columns)")
                
            except Exception as e:
                print(f"  ✗ Error in step {step.name}: {e}")
                raise
            finally:
                step.unload()
        
        return current_df
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
```

---

## 4. Complete Working Example

### Example: Instruction Dataset Generation Pipeline

```python
import pandas as pd
from typing import Dict

# Sample data
data = pd.DataFrame({
    'topic': ['Python', 'Machine Learning', 'Web Development', 'Data Science'],
    'difficulty': ['beginner', 'intermediate', 'beginner', 'advanced']
})

# Create pipeline
pipeline = SimplePipeline(
    name="instruction-generation",
    description="Generate instructions and responses for programming topics"
)

# Step 1: Load data
load_step = LoadDataFrame(
    name="load_data",
    df=data
)

# Step 2: Generate instruction using Ollama
def instruction_prompt_template(row: Dict) -> str:
    return f"Generate a {row['difficulty']} level instruction or question about {row['topic']}."

generate_instruction = OllamaLLMStep(
    name="generate_instruction",
    model_name="llama3.2",
    prompt_column="topic",
    output_column="instruction",
    prompt_template=instruction_prompt_template,
    system_prompt="You are an expert educator creating programming exercises.",
    batch_size=2,
    generation_kwargs={
        "temperature": 0.7,
        "num_predict": 100
    }
)

# Step 3: Generate response using Ollama
def response_prompt_template(row: Dict) -> str:
    return f"Provide a detailed answer to this instruction: {row['instruction']}"

generate_response = OllamaLLMStep(
    name="generate_response",
    model_name="llama3.2",
    prompt_column="instruction",
    output_column="response",
    prompt_template=response_prompt_template,
    system_prompt="You are a helpful programming instructor.",
    batch_size=2,
    generation_kwargs={
        "temperature": 0.6,
        "num_predict": 300
    }
)

# Step 4: Keep relevant columns
keep_cols = KeepColumns(
    name="keep_columns",
    columns=["topic", "difficulty", "instruction", "response"]
)

# Build pipeline
pipeline.add_step(load_step)
pipeline.add_step(generate_instruction)
pipeline.add_step(generate_response)
pipeline.add_step(keep_cols)

# Execute
if __name__ == "__main__":
    result_df = pipeline.run(use_cache=True)
    
    print("\n" + "="*50)
    print("FINAL DATASET")
    print("="*50)
    print(result_df)
    
    # Save result
    result_df.to_csv("instruction_dataset.csv", index=False)
    print("\n✓ Saved to instruction_dataset.csv")
```

### Example: Multi-Model Comparison Pipeline

```python
# Compare outputs from different Ollama models

data = pd.DataFrame({
    'prompt': [
        "Explain quantum computing in simple terms",
        "What are the benefits of functional programming?",
        "How does a neural network work?"
    ]
})

pipeline = SimplePipeline(name="model-comparison")

# Load data
load_step = LoadDataFrame(name="load", df=data)

# Generate with Llama 3.2
gen_llama = OllamaLLMStep(
    name="llama3",
    model_name="llama3.2",
    prompt_column="prompt",
    output_column="llama_response",
    batch_size=3,
    output_mappings={"model_name": "llama_model"}
)

# Generate with Mistral
gen_mistral = OllamaLLMStep(
    name="mistral",
    model_name="mistral",
    prompt_column="prompt",
    output_column="mistral_response",
    batch_size=3,
    output_mappings={"model_name": "mistral_model"}
)

# Note: For true parallel execution like Distilabel,
# you'd need to implement branching logic. For simplicity,
# this runs sequentially but keeps all columns.

pipeline.add_step(load_step)
pipeline.add_step(gen_llama)
pipeline.add_step(gen_mistral)

result = pipeline.run()
result.to_csv("model_comparison.csv", index=False)
```

---

## 5. Advanced Patterns

### Custom Task Step with Structured Output

```python
from pydantic import BaseModel
from typing import Optional
import json

class SentimentOutput(BaseModel):
    sentiment: str
    confidence: float
    reasoning: str

class StructuredOllamaStep(OllamaLLMStep):
    """Ollama step with structured JSON output parsing."""
    
    def __init__(self, schema: BaseModel, **kwargs):
        super().__init__(**kwargs)
        self.schema = schema
    
    def _generate_with_retry(self, prompt: str, retry_count: int = 0) -> Optional[Dict]:
        """Override to parse JSON output."""
        
        # Add JSON instruction to prompt
        schema_str = json.dumps(self.schema.model_json_schema(), indent=2)
        full_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{schema_str}"
        
        raw_response = super()._generate_with_retry(full_prompt, retry_count)
        
        if raw_response:
            try:
                # Extract JSON from response (handle markdown code blocks)
                json_str = raw_response
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                
                parsed = json.loads(json_str)
                # Validate with Pydantic
                validated = self.schema(**parsed)
                return validated.model_dump()
            except Exception as e:
                print(f"Failed to parse JSON: {e}")
                return None
        
        return None

# Usage
sentiment_step = StructuredOllamaStep(
    name="sentiment_analysis",
    schema=SentimentOutput,
    model_name="llama3.2",
    prompt_column="text",
    output_column="sentiment_data",
    system_prompt="Analyze sentiment and respond with valid JSON only."
)
```

### Parallel Branch Processing (Manual)

```python
# While our simplified pipeline is sequential, you can manually
# process branches and merge results

def run_parallel_branches(df: pd.DataFrame) -> pd.DataFrame:
    """Manually execute parallel branches."""
    
    # Branch 1: Generate with model A
    branch1 = OllamaLLMStep(
        name="model_a",
        model_name="llama3.2",
        prompt_column="instruction",
        output_column="response_a"
    )
    
    # Branch 2: Generate with model B  
    branch2 = OllamaLLMStep(
        name="model_b",
        model_name="mistral",
        prompt_column="instruction",
        output_column="response_b"
    )
    
    # Process both (sequentially, but keeping all columns)
    df = branch1(df)
    df = branch2(df)
    
    return df

# Use in pipeline
pipeline = SimplePipeline(name="parallel-demo")
pipeline.add_step(LoadDataFrame(name="load", df=input_df))

# Add custom processing
result = pipeline.run()
result = run_parallel_branches(result)
```

### Error Handling and Recovery

```python
class RobustOllamaStep(OllamaLLMStep):
    """Enhanced OllamaStep with dead letter queue."""
    
    def __init__(self, save_failures: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.save_failures = save_failures
        self.failed_rows = []
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process with failure tracking."""
        results = []
        
        for idx, row in df.iterrows():
            try:
                prompt = self._format_prompt(row.to_dict())
                generation = self._generate_with_retry(prompt)
                
                result_row = row.copy()
                result_row[self.output_column] = generation
                result_row["model_name"] = self.model_name
                result_row["error"] = None
                results.append(result_row)
                
            except Exception as e:
                # Add to failed rows
                failed_row = row.copy()
                failed_row[self.output_column] = None
                failed_row["model_name"] = self.model_name
                failed_row["error"] = str(e)
                results.append(failed_row)
                self.failed_rows.append((idx, row, e))
        
        result_df = pd.DataFrame(results)
        
        # Save failed rows for later reprocessing
        if self.save_failures and self.failed_rows:
            failed_df = pd.DataFrame([row for _, row, _ in self.failed_rows])
            failed_df.to_csv(f"{self.name}_failed.csv", index=False)
            print(f"  ⚠ {len(self.failed_rows)} rows failed, saved to {self.name}_failed.csv")
        
        return result_df
```

---

## 6. Ollama Integration Best Practices

### Model Selection and Configuration

**For instruction following**: Use `llama3.2`, `mistral`, or `gemma2`  
**For code generation**: Use `codellama`, `deepseek-coder`, or `qwen2.5-coder`  
**For structured output**: Use `llama3.2` with JSON mode

### Generation Parameters

```python
generation_kwargs = {
    "temperature": 0.7,      # Creativity (0=deterministic, 1=creative)
    "top_p": 0.9,            # Nucleus sampling
    "top_k": 40,             # Top-k sampling
    "num_predict": 512,      # Max tokens to generate
    "repeat_penalty": 1.1,   # Penalize repetition
    "stop": ["\n\n", "###"]  # Stop sequences
}
```

### Prompt Engineering for Ollama

```python
def effective_prompt_template(row: Dict) -> str:
    """Structure prompts for better Ollama performance."""
    return f"""Task: Generate a response based on the following.

Topic: {row['topic']}
Difficulty: {row['difficulty']}

Instructions:
1. Be concise and clear
2. Match the difficulty level
3. Provide practical examples

Response:"""

# Use system prompts effectively
system_prompt = """You are an expert AI assistant. Follow instructions precisely 
and provide high-quality, accurate responses."""
```

### Batch Size Tuning

- **Small prompts (\<500 tokens)**: batch_size = 16-32
- **Medium prompts (500-1500 tokens)**: batch_size = 8-16
- **Large prompts (\>1500 tokens)**: batch_size = 4-8
- **Very large models (70B+)**: batch_size = 1-4

### Memory Management

```python
# For large datasets, process in chunks
def process_large_dataset(df: pd.DataFrame, chunk_size: int = 1000):
    """Process very large DataFrames in chunks."""
    
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    results = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        result = pipeline.run(input_df=chunk, use_cache=False)
        results.append(result)
        
        # Save intermediate results
        result.to_parquet(f"chunk_{i}.parquet")
    
    return pd.concat(results, ignore_index=True)
```

---

## 7. Comparison: Simplified vs Full Distilabel

| Feature | Distilabel | Simplified Implementation |
|---------|------------|---------------------------|
| **Execution** | Multiprocessing (parallel) | Sequential |
| **Data Format** | List of dicts (batches) | pandas DataFrame |
| **DAG** | Full validation, complex routing | Simple linear/manual branching |
| **Caching** | Comprehensive with signatures | Basic pickle caching |
| **LLM Support** | 10+ providers | Ollama only |
| **Complexity** | Production-ready, feature-rich | Minimal, easy to understand |
| **Performance** | High (parallel steps) | Lower (sequential) but simpler |
| **Debugging** | More difficult (multiprocess) | Easy (single process) |
| **Use Case** | Large-scale production pipelines | Prototyping, small-medium datasets |

---

## 8. Migration Path to Full Distilabel

### When to Upgrade

Consider full Distilabel when:
- Dataset \> 100K rows requiring parallel processing
- Need multiple LLM providers (OpenAI, Anthropic, etc.)
- Require distributed execution across machines
- Need advanced features (offline batch generation, streaming to Hub)
- Production deployment with high reliability requirements

### Conversion Pattern

```python
# Your simplified code:
pipeline = SimplePipeline(name="test")
pipeline.add_step(LoadDataFrame(name="load", df=df))
pipeline.add_step(OllamaLLMStep(name="gen", model_name="llama3.2", ...))
result = pipeline.run()

# Equivalent Distilabel code:
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.llms import OllamaLLM
from distilabel.steps.tasks import TextGeneration

with Pipeline(name="test") as pipeline:
    load = LoadDataFromDicts(data=df.to_dict('records'))
    gen = TextGeneration(
        llm=OllamaLLM(model="llama3.2", host="http://localhost:11434")
    )
    load >> gen

distiset = pipeline.run()
result_df = distiset["default"]["train"].to_pandas()
```

---

## 9. Testing and Debugging

### Unit Testing Steps

```python
import pytest

def test_ollama_step():
    """Test OllamaLLMStep with mock client."""
    
    # Mock Ollama response
    class MockClient:
        def chat(self, **kwargs):
            return {'message': {'content': 'test response'}}
    
    step = OllamaLLMStep(
        name="test",
        model_name="llama3.2",
        prompt_column="text",
        output_column="output"
    )
    step.client = MockClient()
    step._loaded = True
    
    # Test data
    df = pd.DataFrame({'text': ['test prompt']})
    result = step.process(df)
    
    assert 'output' in result.columns
    assert result['output'].iloc[0] == 'test response'

def test_pipeline_execution():
    """Test full pipeline."""
    
    df = pd.DataFrame({'text': ['test']})
    
    pipeline = SimplePipeline(name="test")
    pipeline.add_step(LoadDataFrame(name="load", df=df))
    
    result = pipeline.run()
    assert len(result) == 1
    assert 'text' in result.columns
```

### Debugging Tips

```python
# Enable verbose output
import logging
logging.basicConfig(level=logging.DEBUG)

# Inspect intermediate outputs
pipeline.run()
for step_name, output in pipeline._step_outputs.items():
    print(f"\n{step_name}:")
    print(output.head())

# Test single step
step = OllamaLLMStep(...)
test_df = df.head(1)  # Single row
result = step(test_df)
print(result)
```

---

## 10. Complete Production-Ready Template

```python
"""
Production-ready template for DataFrame + Ollama pipelines.
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_instruction_pipeline(
    input_csv: str,
    output_dir: str,
    model_name: str = "llama3.2",
    batch_size: int = 8
) -> pd.DataFrame:
    """
    Create and execute a complete instruction dataset pipeline.
    
    Args:
        input_csv: Path to input CSV with topics
        output_dir: Directory for outputs
        model_name: Ollama model to use
        batch_size: Batch size for processing
        
    Returns:
        Final DataFrame with generated data
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} rows")
    
    # Create pipeline
    pipeline = SimplePipeline(
        name=f"instruction-gen-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        description="Generate instruction dataset",
        cache_dir=str(output_path / ".cache")
    )
    
    # Add steps
    pipeline.add_step(LoadDataFrame(name="load", df=df))
    
    pipeline.add_step(OllamaLLMStep(
        name="generate_instruction",
        model_name=model_name,
        prompt_column="topic",
        output_column="instruction",
        system_prompt="Generate clear, educational instructions.",
        batch_size=batch_size,
        generation_kwargs={"temperature": 0.7}
    ))
    
    pipeline.add_step(OllamaLLMStep(
        name="generate_response",
        model_name=model_name,
        prompt_column="instruction",
        output_column="response",
        system_prompt="Provide detailed, accurate responses.",
        batch_size=batch_size,
        generation_kwargs={"temperature": 0.6}
    ))
    
    pipeline.add_step(KeepColumns(
        name="keep_columns",
        columns=["topic", "instruction", "response"]
    ))
    
    # Execute
    logger.info("Starting pipeline execution")
    result = pipeline.run(use_cache=True)
    
    # Save results
    output_csv = output_path / "generated_dataset.csv"
    result.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(result)} rows to {output_csv}")
    
    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "input_rows": len(df),
        "output_rows": len(result),
        "batch_size": batch_size
    }
    
    import json
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return result

# Usage
if __name__ == "__main__":
    result = create_instruction_pipeline(
        input_csv="topics.csv",
        output_dir="output",
        model_name="llama3.2",
        batch_size=8
    )
    
    print("\nPipeline complete!")
    print(f"Generated {len(result)} instruction-response pairs")
```

---

## Conclusion

This implementation guide provides a **simplified but functional** Distilabel-inspired pipeline architecture specifically optimized for **pandas DataFrames and Ollama local models**. The design captures Distilabel's core patterns—step abstraction, pipeline orchestration, batch processing, and caching—while eliminating complexity around multiprocessing, DAG validation, and multi-provider abstractions.

**Key strengths of this approach**:
- Simple to understand and debug (single process execution)
- Native DataFrame support (no conversion overhead)
- Focused on Ollama for local inference
- Production-ready patterns (error handling, caching, logging)
- Easy to extend with custom steps

**When to use**: Prototyping, small-to-medium datasets (up to 100K rows), local development, learning LLM pipeline concepts

**When to upgrade to full Distilabel**: Large-scale production workloads, need for multiple LLM providers, distributed execution requirements, datasets exceeding 100K rows requiring parallel processing

The implementation maintains **conceptual compatibility** with Distilabel, making migration straightforward when scaling requirements demand the full framework's capabilities.