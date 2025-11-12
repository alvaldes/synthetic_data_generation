#!/usr/bin/env python3
"""
Dual Generator Salony Pipeline with Judge Selection

This pipeline takes user stories from the Salony dataset and generates tasks using TWO different
LLM generators, then uses a judge to compare both outputs and select the best one.

Architecture:
1. Load user stories from CSV
2. Generate tasks with Generator A (e.g., llama3.1:8b)
3. Generate tasks with Generator B (e.g., mistral or different parameters)
4. Judge compares both outputs and scores them
5. Select and output only the best result from the winning generator

Usage:
    python salony_dual_generator_pipeline.py output.csv
    python salony_dual_generator_pipeline.py output.csv --model-a llama3.1:8b --model-b mistral
    python salony_dual_generator_pipeline.py output.csv --sample 10
"""

import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional
import ollama

from simple_pipeline import SimplePipeline
from simple_pipeline.steps import LoadDataFrame, OllamaLLMStep, AddColumn, ComparisonJudgeStep, KeepColumns


def create_task_generation_prompt(row: Dict) -> str:
    """
    Creates the prompt for generating tasks from a Salony dataset user story.

    Args:
        row: DataFrame row with 'input' column containing the story

    Returns:
        Formatted prompt
    """
    user_story = row['input'].strip()

    prompt = f"""Below is an instruction that describes a task, paired with an input that provides a user story.

Write a response that appropriately completes the request.


Instruction:

Break this user story into smaller development tasks to help the developers implement it efficiently. You can divide this user story into as many tasks as needed, depending on its complexity. Each task must be unique, actionable, and non-overlapping.

Use the following format for the response:

1. summary: ‹task summary 1›
description: ‹task description 1›
2. summary: ‹task summary 2›
description: ‹task description 2›

N. summary: ‹task summary N›
description: ‹task description N›


Input:

{user_story}


Response:"""

    return prompt


def create_comparison_judge_prompt(row: Dict) -> str:
    """
    Creates the prompt for the judge to compare two generator outputs.

    Args:
        row: DataFrame row with user story and both generator outputs

    Returns:
        Formatted judge prompt
    """
    user_story = row['input'].strip()
    tasks_a = row['tasks_generator_a'].strip()
    tasks_b = row['tasks_generator_b'].strip()

    prompt = f"""You are an expert software development manager evaluating two different breakdowns of a user story into development tasks.

Your job is to compare both breakdowns and determine which one is better overall.

USER STORY:
{user_story}

BREAKDOWN A:
{tasks_a}

BREAKDOWN B:
{tasks_b}

Evaluate both breakdowns based on these criteria (0-10 points each):

1. COMPLETENESS: How well does the breakdown cover all aspects of the user story?
2. CLARITY: How clear and understandable are the task descriptions?
3. ACTIONABILITY: How actionable and specific are the tasks for developers?
4. LOGICAL_STRUCTURE: How well organized and logically sequenced are the tasks?
5. GRANULARITY: Are tasks appropriately sized (not too big, not too small)?

Respond in this exact JSON format:
{{
  "breakdown_a": {{
    "completeness": [score_0_to_10],
    "clarity": [score_0_to_10],
    "actionability": [score_0_to_10],
    "logical_structure": [score_0_to_10],
    "granularity": [score_0_to_10],
    "total_score": [sum_of_all_scores],
    "strengths": "[brief_description_of_strengths]",
    "weaknesses": "[brief_description_of_weaknesses]"
  }},
  "breakdown_b": {{
    "completeness": [score_0_to_10],
    "clarity": [score_0_to_10],
    "actionability": [score_0_to_10],
    "logical_structure": [score_0_to_10],
    "granularity": [score_0_to_10],
    "total_score": [sum_of_all_scores],
    "strengths": "[brief_description_of_strengths]",
    "weaknesses": "[brief_description_of_weaknesses]"
  }},
  "winner": "[A_or_B]",
  "reason": "[brief_explanation_of_why_the_winner_is_better]"
}}"""

    return prompt


def validate_inputs(
    model_a: str,
    model_b: str,
    judge_model: str,
    batch_size: int,
    temperature_a: float,
    temperature_b: float
) -> None:
    """Validate input parameters."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got: {batch_size}")
    if not (0.0 <= temperature_a <= 2.0):
        raise ValueError(f"temperature_a must be between 0.0 and 2.0, got: {temperature_a}")
    if not (0.0 <= temperature_b <= 2.0):
        raise ValueError(f"temperature_b must be between 0.0 and 2.0, got: {temperature_b}")


def load_and_validate_data(input_csv: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load and validate the input CSV data with robust error handling."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Dataset not found: {input_csv}")

    try:
        df = pd.read_csv(input_csv)
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {input_csv}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file {input_csv}: {e}")

    # Remove unnamed index columns
    if len(df.columns) > 0 and df.columns[0] in ['Unnamed: 0', '']:
        df = df.iloc[:, 1:]

    # Validate required column exists
    if 'input' not in df.columns:
        raise ValueError(
            f"CSV must have an 'input' column with user stories. "
            f"Found columns: {list(df.columns)}"
        )

    # Apply sampling if requested
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got: {sample_size}")
        df = df.head(sample_size)
        logging.info(f"Using sample of {sample_size} rows for testing")

    # Clean data
    initial_count = len(df)
    df = df.dropna(subset=['input'])
    df['input'] = df['input'].astype(str).str.strip()
    df = df[df['input'] != '']

    final_count = len(df)
    if final_count < initial_count:
        logging.warning(f"Removed {initial_count - final_count} rows with missing/empty input data")

    if final_count == 0:
        raise ValueError("No valid user stories found after data cleaning")

    return df


def run_dual_generator_pipeline(
    output_csv: str,
    input_csv: Optional[str] = None,
    model_a: str = "llama3.1:8b",
    model_b: str = "qwen3:8b",
    judge_model: str = "llama3.1:8b",
    batch_size: int = 2,
    temperature_a: float = 0.3,
    temperature_b: float = 0.7,
    num_predict: int = 1000,
    sample_size: Optional[int] = None,
    use_cache: bool = True
):
    """
    Execute dual generator pipeline with judge selection.

    Args:
        output_csv: Path to save the results
        input_csv: Optional path to input CSV file (defaults to data/salony_train.csv)
        model_a: First Ollama model for task generation
        model_b: Second Ollama model for task generation
        judge_model: Ollama model for judging and selection
        batch_size: Number of stories to process simultaneously
        temperature_a: Generation temperature for model A
        temperature_b: Generation temperature for model B
        num_predict: Maximum tokens to generate
        sample_size: If specified, process only N stories (for testing)
        use_cache: Whether to use caching
    """

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Validate inputs
    validate_inputs(model_a, model_b, judge_model, batch_size, temperature_a, temperature_b)

    # Determine input file path
    if input_csv is None:
        input_path = Path(__file__).parent.parent / "data" / "salony_train.csv"
    else:
        input_path = Path(input_csv)

    logging.info(f"Loading data from: {input_path}")

    # Test Ollama connection and models
    try:
        client = ollama.Client()
        models = client.list()
        available_models = [m['model'] for m in models['models']]

        for model_name in [model_a, model_b, judge_model]:
            if model_name not in available_models:
                logging.warning(f"Model {model_name} not found locally. Attempting to pull...")
                client.pull(model_name)

    except Exception as e:
        if "connection refused" in str(e).lower():
            raise ConnectionError("Ollama server not running. Please start with: ollama serve")
        raise ConnectionError(f"Cannot connect to Ollama: {e}")

    # Load and validate data
    df = load_and_validate_data(input_path, sample_size)
    logging.info(f"Loaded {len(df)} user stories")

    # Configure pipeline
    pipeline_name = "salony-dual-generator-pipeline"

    logging.info(f"Configuring dual generator pipeline:")
    logging.info(f"  Generator A: {model_a} (temp={temperature_a})")
    logging.info(f"  Generator B: {model_b} (temp={temperature_b})")
    logging.info(f"  Judge: {judge_model}")
    logging.info(f"  Batch size: {batch_size}")

    pipeline = SimplePipeline(
        name=pipeline_name,
        description="Dual generator pipeline with judge selection for Salony dataset"
    )

    # Add data loading step
    pipeline.add_step(
        LoadDataFrame(name="load", df=df)
    )

    # Add Generator A
    pipeline.add_step(
        OllamaLLMStep(
            name="generator_a",
            model_name=model_a,
            prompt_column="input",
            output_column="tasks_generator_a",
            prompt_template=create_task_generation_prompt,
            system_prompt="You are an expert software development lead who excels at breaking down user stories into clear, actionable development tasks.",
            batch_size=batch_size,
            generation_kwargs={
                "temperature": temperature_a,
                "num_predict": num_predict
            },
        )
    )

    # Add Generator B
    pipeline.add_step(
        OllamaLLMStep(
            name="generator_b",
            model_name=model_b,
            prompt_column="input",
            output_column="tasks_generator_b",
            prompt_template=create_task_generation_prompt,
            system_prompt="You are an expert software development lead who excels at breaking down user stories into clear, actionable development tasks.",
            batch_size=batch_size,
            generation_kwargs={
                "temperature": temperature_b,
                "num_predict": num_predict
            },
        )
    )

    # Add Comparison Judge Step
    pipeline.add_step(
        ComparisonJudgeStep(
            name="comparison_judge",
            model_name=judge_model,
            input_column="input",
            output_a_column="tasks_generator_a",
            output_b_column="tasks_generator_b",
            prompt_template_func=create_comparison_judge_prompt,
            system_prompt="You are an expert software development manager who evaluates and compares different task breakdowns to determine which is superior.",
            batch_size=max(1, batch_size // 2),  # Smaller batches for complex judge operations
            generation_kwargs={
                "temperature": 0.2,  # More consistent judging
                "num_predict": 1000
            }
        )
    )

    # Add final column selection - keep only essential outputs
    pipeline.add_step(
        KeepColumns(
            name="final_selection",
            columns=[
                "input",                    # Original user story
                "tasks_generator_a",        # Output from generator A
                "tasks_generator_b",        # Output from generator B
                # "selected_output",          # Best output chosen by judge
                "judge_score_a",            # Score for generator A
                "judge_score_b",            # Score for generator B
                "judge_winner",             # Which generator won (A or B)
                "judge_reason"              # Why the winner was chosen
            ]
        )
    )

    # Execute pipeline
    logging.info("Executing complete dual generator pipeline with judge selection...")
    try:
        result_df = pipeline.run(use_cache=use_cache)
    except ollama.ResponseError as e:
        raise RuntimeError(f"Ollama API error: {e.error}")
    except Exception as e:
        raise RuntimeError(f"Pipeline execution failed: {e}")

    logging.info(f"Successfully processed {len(result_df)} user stories with dual generators and judge selection")

    # Show statistics
    if len(result_df) > 0:
        winner_counts = result_df['judge_winner'].value_counts()
        avg_score_a = result_df['judge_score_a'].mean()
        avg_score_b = result_df['judge_score_b'].mean()

        logging.info(f"\n=== JUDGE SELECTION STATISTICS ===")
        logging.info(f"Generator A ({model_a}) wins: {winner_counts.get('A', 0)} ({winner_counts.get('A', 0)/len(result_df)*100:.1f}%)")
        logging.info(f"Generator B ({model_b}) wins: {winner_counts.get('B', 0)} ({winner_counts.get('B', 0)/len(result_df)*100:.1f}%)")
        logging.info(f"Average score A: {avg_score_a:.1f}/50")
        logging.info(f"Average score B: {avg_score_b:.1f}/50")

        # Preview sample results
        logging.info(f"\n=== SAMPLE RESULT ===")
        sample_idx = 0
        sample = result_df.iloc[sample_idx]
        logging.info(f"User Story: {sample['input'][:150]}...")
        logging.info(f"Winner: Generator {sample['judge_winner']} (Score A: {sample['judge_score_a']}, Score B: {sample['judge_score_b']})")
        logging.info(f"Reason: {sample['judge_reason']}")

        # Show the winning output based on judge_winner
        if sample['judge_winner'] == 'B':
            winning_output = sample['tasks_generator_b']
            logging.info(f"Winning Output (Generator B): {winning_output[:300]}...")
        else:
            winning_output = sample['tasks_generator_a']
            logging.info(f"Winning Output (Generator A): {winning_output[:300]}...")

    # Save results
    logging.info(f"\nSaving final results to: {output_csv}")
    try:
        result_df.to_csv(output_csv, index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save CSV: {e}")

    logging.info("✅ Dual generator pipeline with judge selection completed successfully!")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate development tasks using dual generators with judge selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python salony_dual_generator_pipeline.py output.csv
  python salony_dual_generator_pipeline.py output.csv --model-a llama3.1:8b --model-b mistral
  python salony_dual_generator_pipeline.py output.csv --sample 5 --temperature-a 0.3 --temperature-b 0.7
  python salony_dual_generator_pipeline.py output.csv --input-csv custom_data.csv

Default dataset: data/salony_train.csv
        """
    )

    parser.add_argument(
        'output_csv',
        help='Path to save the generated tasks'
    )

    parser.add_argument(
        '--input-csv',
        help='Path to input CSV file (default: data/salony_train.csv)'
    )

    parser.add_argument(
        '--model-a',
        default='llama3.1:8b',
        help='First Ollama model for task generation (default: llama3.1:8b)'
    )

    parser.add_argument(
        '--model-b',
        default='qwen3:8b',
        help='Second Ollama model for task generation (default: qwen3:8b)'
    )

    parser.add_argument(
        '--judge-model',
        default='llama3.1:8b',
        help='Ollama model for judge comparison (default: llama3.1:8b)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Stories to process simultaneously (default: 2)'
    )

    parser.add_argument(
        '--temperature-a',
        type=float,
        default=0.3,
        help='Generation temperature for model A (default: 0.3)'
    )

    parser.add_argument(
        '--temperature-b',
        type=float,
        default=0.7,
        help='Generation temperature for model B (default: 0.7)'
    )

    parser.add_argument(
        '--num-predict',
        type=int,
        default=1000,
        help='Maximum tokens to generate (default: 1000)'
    )

    parser.add_argument(
        '--sample',
        type=int,
        help='Number of stories to process (useful for testing)'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching'
    )

    args = parser.parse_args()

    try:
        run_dual_generator_pipeline(
            output_csv=args.output_csv,
            input_csv=args.input_csv,
            model_a=args.model_a,
            model_b=args.model_b,
            judge_model=args.judge_model,
            batch_size=args.batch_size,
            temperature_a=args.temperature_a,
            temperature_b=args.temperature_b,
            num_predict=args.num_predict,
            sample_size=args.sample,
            use_cache=not args.no_cache
        )
        return 0
    except (ValueError, FileNotFoundError, ConnectionError, RuntimeError) as e:
        logging.error(f"Error: {e}")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())