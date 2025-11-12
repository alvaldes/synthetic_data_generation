# Dual Generator Pipeline with Judge Selection

This feature implements a **dual generator pipeline** that uses two different LLM generators to create task breakdowns from user stories, then uses an LLM judge to compare both outputs and select the best one.

## 🏗️ Architecture

```
Input User Story
       ↓
   Generator A (e.g., llama3.1:8b, temp=0.3)
       ↓
   Generator B (e.g., mistral, temp=0.7)
       ↓
   LLM Judge (compares A vs B)
       ↓
   Selected Best Output
```

## 🚀 Quick Start

### Basic Usage

```bash
# Run with default models
python examples/salony_dual_generator_pipeline.py output.csv --sample 5

# Use different models for each generator
python examples/salony_dual_generator_pipeline.py output.csv \
  --model-a llama3.1:8b \
  --model-b mistral \
  --judge-model llama3.1:8b

# Customize temperatures for different generation styles
python examples/salony_dual_generator_pipeline.py output.csv \
  --temperature-a 0.3 \
  --temperature-b 0.8 \
  --sample 10
```

### Test the Pipeline

```bash
# Quick test with sample data
python test_dual_generator.py
```

## 📊 Output Format

The pipeline produces a CSV with these columns:

| Column | Description |
|--------|-------------|
| `input` | Original user story |
| `selected_output` | **The winning generator's output** |
| `judge_winner` | Which generator won (A or B) |
| `judge_score_a` | Score for Generator A (0-50) |
| `judge_score_b` | Score for Generator B (0-50) |
| `judge_reason` | Why the judge selected the winner |

## 🔧 Implementation Details

### Key Components

1. **`ComparisonJudgeStep`** - New step that compares two generator outputs
2. **`salony_dual_generator_pipeline.py`** - Complete pipeline implementation
3. **Scoring Criteria** - Judge evaluates on 5 dimensions (0-10 each):
   - Completeness
   - Clarity
   - Actionability
   - Logical Structure
   - Granularity

### Judge Prompt Strategy

The judge receives:
- Original user story
- Both generator outputs (A and B)
- Structured evaluation criteria
- Request for JSON-formatted decision

The judge returns structured scores and selects the winner based on total score.

## 🎯 Use Cases

### Model Comparison
```bash
# Compare different model families
python examples/salony_dual_generator_pipeline.py comparison_llama_vs_mistral.csv \
  --model-a llama3.1:8b \
  --model-b mistral
```

### Temperature Analysis
```bash
# Compare conservative vs creative outputs
python examples/salony_dual_generator_pipeline.py temp_analysis.csv \
  --model-a llama3.1:8b --temperature-a 0.2 \
  --model-b llama3.1:8b --temperature-b 0.9
```

### Production Quality Selection
```bash
# Automatically select best quality outputs
python examples/salony_dual_generator_pipeline.py production_tasks.csv \
  --model-a llama3.1:8b \
  --model-b mistral \
  --batch-size 2
```

## 📈 Performance Characteristics

- **Processing**: ~3x slower than single generator (2 generations + 1 judge)
- **Quality**: Higher quality outputs through competitive selection
- **Caching**: Supports full pipeline caching for efficiency
- **Batching**: Configurable batch sizes for memory optimization

## 🛠️ Configuration Options

```bash
# All available options
python examples/salony_dual_generator_pipeline.py --help
```

Key parameters:
- `--model-a`, `--model-b`: Generator models
- `--judge-model`: Judge model (defaults to model-a)
- `--temperature-a`, `--temperature-b`: Generation temperatures
- `--batch-size`: Concurrent processing batch size
- `--sample`: Limit to N stories (for testing)

## 🧪 Testing

Run the test suite:

```bash
python test_dual_generator.py
```

This creates sample user stories and runs them through the complete pipeline to verify:
- Both generators produce outputs
- Judge properly compares and selects winner
- Final output contains only the selected best result

## 🔍 Example Output

```
User Story: As a user, I want to login so I can access my account.

Generator A (score: 42):
1. Create login form UI
2. Implement authentication logic
3. Add session management

Generator B (score: 38):
1. Design login interface
2. Build authentication system
3. Setup user sessions

Winner: Generator A
Reason: More specific and actionable task descriptions

Selected Output: [Generator A's full response]
```

## ⚡ Performance Tips

1. **Use caching** (`--cache` enabled by default)
2. **Start with small samples** (`--sample 5`) for testing
3. **Adjust batch sizes** based on available memory
4. **Choose complementary models** (e.g., different model families or temperatures)

## 🤝 Integration

This pipeline is fully compatible with the existing simple-pipeline architecture:

- Uses standard `BaseStep` patterns
- Supports all caching mechanisms
- Follows established error handling
- Maintains consistent logging format

The `ComparisonJudgeStep` can be reused in other pipelines that need to compare multiple generator outputs.