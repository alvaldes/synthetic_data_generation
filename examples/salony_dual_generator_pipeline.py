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
import time
from pathlib import Path
from typing import Dict, Optional
import ollama

from dataforge import DataForgePipeline
from dataforge.steps import (
    LoadDataFrame,
    OllamaLLMStep,
    AddColumn,
    ComparisonJudgeStep,
    KeepColumns,
    ExplodeTasks,
    ValidateUserStories,
)


def create_task_generation_prompt(row: Dict) -> str:
    """
    Creates the prompt for generating tasks from a Salony dataset user story.

    Args:
        row: DataFrame row with 'input' column containing the story

    Returns:
        Formatted prompt
    """
    user_story = row["input"].strip()

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
    user_story = row["input"].strip()
    tasks_a = row["tasks_generator_a"].strip()
    tasks_b = row["tasks_generator_b"].strip()

    prompt = f"""You are an expert software development manager evaluating two different breakdowns of a user story into development tasks.

Your job is to compare both breakdowns and determine which one is better overall.

USER STORY:
{user_story}

BREAKDOWN A:
{tasks_a}

BREAKDOWN B:
{tasks_b}

Evaluate both breakdowns based on these criteria (0-10 points each), aligned with ISO/IEC/IEEE 29148:2018 quality standards:

1. COHERENCE: Semantic correspondence between the generated tasks and the original user story. Does it maintain consistency with the user story intent? (0-10)
2. COMPLETENESS: Degree of coverage of implicit and explicit requirements. Does it cover all necessary aspects of the user story? (0-10)
3. FEASIBILITY: Technical feasibility of the proposed tasks. Are the tasks realistically implementable? (0-10)
4. FORMAT: Structure and clarity of the output. Is the response well-formatted, unambiguous, and easy to understand? (0-10)
5. GRANULARITY: Appropriateness of the decomposition level. Are tasks properly sized (not too big, not too small) and atomic? (0-10)

Respond in this exact JSON format:
{{
  "breakdown_a": {{
    "coherence": [score_0_to_10],
    "completeness": [score_0_to_10],
    "feasibility": [score_0_to_10],
    "format": [score_0_to_10],
    "granularity": [score_0_to_10],
    "total_score": [sum_of_all_scores],
    "strengths": "[brief_description_of_strengths]",
    "weaknesses": "[brief_description_of_weaknesses]"
  }},
  "breakdown_b": {{
    "coherence": [score_0_to_10],
    "completeness": [score_0_to_10],
    "feasibility": [score_0_to_10],
    "format": [score_0_to_10],
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
    temperature_b: float,
) -> None:
    """Validate input parameters."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got: {batch_size}")
    if not (0.0 <= temperature_a <= 2.0):
        raise ValueError(
            f"temperature_a must be between 0.0 and 2.0, got: {temperature_a}"
        )
    if not (0.0 <= temperature_b <= 2.0):
        raise ValueError(
            f"temperature_b must be between 0.0 and 2.0, got: {temperature_b}"
        )


def load_and_validate_data(
    input_csv: Path, sample_size: Optional[int] = None
) -> pd.DataFrame:
    """Load and validate the input CSV data with robust error handling."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Dataset not found: {input_csv}")

    try:
        df = pd.read_csv(input_csv)
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {input_csv}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file {input_csv}: {e}")

    # Validate required column exists
    if "input" not in df.columns:
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
    df = df.dropna(subset=["input"])
    df["input"] = df["input"].astype(str).str.strip()
    df = df[df["input"] != ""]

    final_count = len(df)
    if final_count < initial_count:
        logging.warning(
            f"Removed {initial_count - final_count} rows with missing/empty input data"
        )

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
    use_cache: bool = True,
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

    # Set up logging with both console and file output
    log_file = Path(output_csv).with_suffix(".log")

    # Configure logging to write to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),  # File output
        ],
    )

    logging.info(f"Logging to: {log_file}")

    # Validate inputs
    validate_inputs(
        model_a, model_b, judge_model, batch_size, temperature_a, temperature_b
    )

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
        available_models = [m["model"] for m in models["models"]]

        for model_name in [model_a, model_b, judge_model]:
            if model_name not in available_models:
                logging.warning(
                    f"Model {model_name} not found locally. Attempting to pull..."
                )
                client.pull(model_name)

    except Exception as e:
        if "connection refused" in str(e).lower():
            raise ConnectionError(
                "Ollama server not running. Please start with: ollama serve"
            )
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

    pipeline = DataForgePipeline(
        name=pipeline_name,
        description="Dual generator pipeline with judge selection for Salony dataset",
    )

    # Add data loading step
    pipeline.add_step(LoadDataFrame(name="load", df=df))

    # Validate user story format
    pipeline.add_step(
        ValidateUserStories(
            name="validate_format", story_column="input", case_sensitive=False
        )
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
                "num_predict": num_predict,
            },
            track_time=True,
            time_column="generation_time_a",
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
                "num_predict": num_predict,
            },
            track_time=True,
            time_column="generation_time_b",
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
            # Smaller batches for complex judge operations
            batch_size=max(1, batch_size // 2),
            generation_kwargs={
                "temperature": 0.2,  # More consistent judging
                "num_predict": 1000,
            },
        )
    )

    # Add US ID tracking (counter starting from 1)
    us_counter = {"count": 0}

    def get_us_id():
        us_counter["count"] += 1
        return us_counter["count"]

    pipeline.add_step(
        AddColumn(
            name="add_us_id", input_columns=[], output_column="us_id", func=get_us_id
        )
    )

    # Execute pipeline up to this point (before exploding tasks)
    logging.info("Executing dual generator pipeline with judge comparison...")
    pipeline_start_time = time.time()
    start_timestamp = pd.Timestamp.now().isoformat()

    try:
        result_df = pipeline.run(use_cache=use_cache)
    except ollama.ResponseError as e:
        raise RuntimeError(f"Ollama API error: {e.error}")
    except Exception as e:
        raise RuntimeError(f"Pipeline execution failed: {e}")

    end_timestamp = pd.Timestamp.now().isoformat()
    pipeline_end_time = time.time()
    total_pipeline_time = pipeline_end_time - pipeline_start_time

    logging.info(
        f"Successfully compared {len(result_df)} user stories with dual generators"
    )

    # Add pipeline-level timestamps to all rows
    result_df["start_timestamp"] = start_timestamp
    result_df["end_timestamp"] = end_timestamp

    # Calculate total_time per row (generation_time_a + generation_time_b + judge_time)
    result_df["total_time"] = (
        result_df["generation_time_a"]
        + result_df["generation_time_b"]
        + result_df["judge_time"]
    )

    # Create judge results CSV (one row per user story) with detailed scores
    judge_results_df = result_df[
        [
            "us_id",
            "input",
            # Total scores
            "judge_score_a_total",
            "judge_score_b_total",
            # Individual criteria scores for A
            "judge_score_a_coherence",
            "judge_score_a_completeness",
            "judge_score_a_feasibility",
            "judge_score_a_format",
            "judge_score_a_granularity",
            # Individual criteria scores for B
            "judge_score_b_coherence",
            "judge_score_b_completeness",
            "judge_score_b_feasibility",
            "judge_score_b_format",
            "judge_score_b_granuylarity",
            # Qualitative feedback
            "judge_strengths_a",
            "judge_weaknesses_a",
            "judge_strengths_b",
            "judge_weaknesses_b",
            # Decision
            "judge_winner",
            "judge_reason",
            # Timing metadata
            "generation_time_a",
            "generation_time_b",
            "judge_time",
            "total_time",
            "start_timestamp",
            "end_timestamp",
        ]
    ].copy()

    # Save judge results
    judge_output_path = Path(output_csv).stem + "_judge_results.csv"
    judge_output_full = str(Path(output_csv).parent / judge_output_path)

    logging.info(f"\nSaving judge results to: {judge_output_full}")
    try:
        judge_results_df.to_csv(judge_output_full, index=False)
        logging.info(f"✅ Judge results saved ({len(judge_results_df)} user stories)")
    except Exception as e:
        raise RuntimeError(f"Failed to save judge results CSV: {e}")

    # Now create tasks DataFrames by exploding both generators
    from dataforge.steps.explode_tasks import ExplodeTasks

    explode_step = ExplodeTasks(
        name="explode",
        tasks_column="tasks",
        output_column="task",
        group_by_column="us_id",  # ← RESET task_id POR CADA us_id
    )

    # Explode Generator A tasks
    logging.info("\nExploding Generator A tasks...")
    df_a = result_df[["us_id", "input", "tasks_generator_a"]].copy()
    df_a = df_a.rename(columns={"tasks_generator_a": "tasks"})
    df_a_exploded = explode_step.process(df_a)
    df_a_exploded["generator"] = "A"
    df_a_exploded["generator_model"] = model_a
    df_a_exploded = df_a_exploded.rename(columns={"task": "task_generator_a"})

    # Explode Generator B tasks
    logging.info("Exploding Generator B tasks...")
    df_b = result_df[["us_id", "input", "tasks_generator_b"]].copy()
    df_b = df_b.rename(columns={"tasks_generator_b": "tasks"})
    df_b_exploded = explode_step.process(df_b)
    df_b_exploded["generator"] = "B"
    df_b_exploded["generator_model"] = model_b
    df_b_exploded = df_b_exploded.rename(columns={"task": "task_generator_b"})

    # Merge both on us_id and task_id to align tasks side-by-side
    logging.info("Merging exploded tasks...")
    tasks_df = pd.merge(
        df_a_exploded[["us_id", "input", "task_id", "task_generator_a"]],
        df_b_exploded[["us_id", "input", "task_id", "task_generator_b"]],
        on=["us_id", "task_id"],
        how="outer",
        suffixes=("", "_b"),  # Mantener input de A, renombrar B si existe
    )

    # Resolver conflicto de columna 'input': usar input de A, si está vacío usar de B
    if "input_b" in tasks_df.columns:
        tasks_df["input"] = tasks_df["input"].fillna(tasks_df["input_b"])
        tasks_df = tasks_df.drop(columns=["input_b"])

    # Reorder columns
    tasks_df = tasks_df[
        ["us_id", "task_id", "input", "task_generator_a", "task_generator_b"]
    ]

    # Save tasks CSV
    logging.info(f"\nSaving tasks comparison to: {output_csv}")
    try:
        tasks_df.to_csv(output_csv, index=False)
        logging.info(
            f"✅ Tasks saved ({len(tasks_df)} tasks from {
                tasks_df['us_id'].nunique()
            } user stories)"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save tasks CSV: {e}")

    # Show statistics
    if len(result_df) > 0:
        winner_counts = result_df["judge_winner"].value_counts()
        avg_score_a = result_df["judge_score_a_total"].mean()
        avg_score_b = result_df["judge_score_b_total"].mean()

        logging.info(f"\n=== JUDGE SELECTION STATISTICS ===")
        logging.info(
            f"Generator A ({model_a}) wins: {winner_counts.get('A', 0)} ({
                winner_counts.get('A', 0) / len(result_df) * 100:.1f}%)"
        )
        logging.info(
            f"Generator B ({model_b}) wins: {winner_counts.get('B', 0)} ({
                winner_counts.get('B', 0) / len(result_df) * 100:.1f}%)"
        )
        logging.info(
            f"Average total scores: A={avg_score_a:.1f}/50, B={avg_score_b:.1f}/50"
        )

        # Show average scores by criteria
        logging.info(f"\n=== AVERAGE SCORES BY CRITERIA (out of 10) ===")
        criteria = [
            "coherencia",
            "completitud",
            "viabilidad",
            "formato",
            "granularidad",
        ]
        for criterion in criteria:
            avg_a = result_df[f"judge_score_a_{criterion}"].mean()
            avg_b = result_df[f"judge_score_b_{criterion}"].mean()
            logging.info(f"{criterion.capitalize():20s}: A={avg_a:.1f}, B={avg_b:.1f}")

        # Show timing statistics
        logging.info(f"\n=== TIMING STATISTICS (seconds) ===")
        avg_time_a = result_df["generation_time_a"].mean()
        avg_time_b = result_df["generation_time_b"].mean()
        avg_judge_time = result_df["judge_time"].mean()
        avg_total_time = result_df["total_time"].mean()

        logging.info(f"Average generation time A: {avg_time_a:.2f}s")
        logging.info(f"Average generation time B: {avg_time_b:.2f}s")
        logging.info(f"Average judge time:        {avg_judge_time:.2f}s")
        logging.info(f"Average total time:        {avg_total_time:.2f}s")
        logging.info(f"Total pipeline time:       {total_pipeline_time:.2f}s")

    logging.info(
        "\n✅ Dual generator pipeline with judge comparison completed successfully!"
    )
    logging.info(f"   - Tasks comparison: {output_csv}")
    logging.info(f"   - Judge results: {judge_output_full}")

    return tasks_df, judge_results_df


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
        """,
    )

    parser.add_argument("output_csv", help="Path to save the generated tasks")

    parser.add_argument(
        "--input-csv", help="Path to input CSV file (default: data/salony_train.csv)"
    )

    parser.add_argument(
        "--model-a",
        default="llama3.1:8b",
        help="First Ollama model for task generation (default: llama3.1:8b)",
    )

    parser.add_argument(
        "--model-b",
        default="qwen3:8b",
        help="Second Ollama model for task generation (default: qwen3:8b)",
    )

    parser.add_argument(
        "--judge-model",
        default="llama3.1:8b",
        help="Ollama model for judge comparison (default: llama3.1:8b)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Stories to process simultaneously (default: 2)",
    )

    parser.add_argument(
        "--temperature-a",
        type=float,
        default=0.3,
        help="Generation temperature for model A (default: 0.3)",
    )

    parser.add_argument(
        "--temperature-b",
        type=float,
        default=0.7,
        help="Generation temperature for model B (default: 0.7)",
    )

    parser.add_argument(
        "--num-predict",
        type=int,
        default=1000,
        help="Maximum tokens to generate (default: 1000)",
    )

    parser.add_argument(
        "--sample", type=int, help="Number of stories to process (useful for testing)"
    )

    parser.add_argument("--no-cache", action="store_true", help="Disable caching")

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
            use_cache=not args.no_cache,
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

