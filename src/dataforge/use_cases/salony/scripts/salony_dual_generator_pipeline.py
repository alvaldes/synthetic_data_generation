#!/usr/bin/env python3
"""
Dual Generator Salony Pipeline with Judge Selection

This pipeline takes user stories from the Salony dataset and generates tasks
using TWO different LLM generators, then uses a judge to compare both outputs
and select the best one.

Architecture:
1. Load user stories from CSV
2. Generate tasks with Generator A
3. Generate tasks with Generator B
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
from dataforge.transformers import LoadDataFrame, AddColumn, KeepColumns, ExplodeTasks
from dataforge.llm import OllamaLLMStep, ComparisonJudgeStep
from dataforge.validators import ValidateUserStories
from dataforge.config import get_settings, PromptLoader


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def create_task_generation_prompt(row: Dict) -> str:
    """Render the task generation template with the user story."""
    return PromptLoader.render(
        "task_generation.j2",
        {"user_story": row.get("input", "").strip()},
    )


def create_comparison_judge_prompt(row: Dict) -> str:
    """Render the comparison judge template with both generator outputs."""
    return PromptLoader.render(
        "comparison_judge.j2",
        {
            "user_story": row.get("input", "").strip(),
            "tasks_a": row.get("tasks_generator_a", "").strip(),
            "tasks_b": row.get("tasks_generator_b", "").strip(),
        },
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

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

    if "input" not in df.columns:
        raise ValueError(
            f"CSV must have an 'input' column with user stories. "
            f"Found columns: {list(df.columns)}"
        )

    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got: {sample_size}")
        df = df.head(sample_size)
        logging.info(f"Using sample of {sample_size} rows for testing")

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


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_dual_generator_pipeline(
    output_csv: str,
    input_csv: Optional[str] = None,
    model_a: Optional[str] = None,
    model_b: Optional[str] = None,
    judge_model: Optional[str] = None,
    batch_size: Optional[int] = None,
    temperature_a: Optional[float] = None,
    temperature_b: Optional[float] = None,
    num_predict: Optional[int] = None,
    sample_size: Optional[int] = None,
    use_cache: bool = True,
):
    """
    Execute dual generator pipeline with judge selection.

    Parameters
    ----------
    output_csv : str
        Path to save the results.
    input_csv : str, optional
        Path to input CSV (defaults to ``<paths.raw_dir>/salony_train.csv``).
    model_a : str, optional
        First Ollama model for task generation (default from config).
    model_b : str, optional
        Second Ollama model (default from config or ``qwen3:8b``).
    judge_model : str, optional
        Ollama model for judging (defaults to *model_a*).
    batch_size : int, optional
        Stories to process simultaneously (default from config).
    temperature_a : float, optional
        Generation temperature for model A (default from config).
    temperature_b : float, optional
        Generation temperature for model B (default: 0.7).
    num_predict : int, optional
        Maximum tokens to generate (default from config).
    sample_size : int, optional
        If specified, process only N stories.
    use_cache : bool
        Whether to use caching.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (tasks_df, judge_results_df)
    """
    # --- Resolve config ---------------------------------------------------
    cfg = get_settings(use_case="salony")
    llm_cfg = cfg.llm
    comp_cfg = cfg.comparison_judge

    model_a = model_a or llm_cfg.default_model
    model_b = model_b or "qwen3:8b"
    judge_model = judge_model or model_a
    batch_size = batch_size if batch_size is not None else llm_cfg.batch_size
    temperature_a = temperature_a if temperature_a is not None else llm_cfg.temperature
    temperature_b = temperature_b if temperature_b is not None else 0.7
    num_predict = num_predict if num_predict is not None else llm_cfg.num_predict

    # --- Setup ------------------------------------------------------------
    log_file = Path(output_csv).with_suffix(".log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )
    logging.info(f"Logging to: {log_file}")

    validate_inputs(
        model_a, model_b, judge_model, batch_size, temperature_a, temperature_b
    )

    if input_csv is None:
        input_path = Path(cfg.paths.raw_dir) / "salony_train.csv"
    else:
        input_path = Path(input_csv)

    logging.info(f"Loading data from: {input_path}")

    # --- Ollama connectivity check ----------------------------------------
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

    # --- Load data --------------------------------------------------------
    df = load_and_validate_data(input_path, sample_size)
    logging.info(f"Loaded {len(df)} user stories")

    # --- Build pipeline ---------------------------------------------------
    pipeline = DataForgePipeline(
        name="salony-dual-generator-pipeline",
        description="Dual generator pipeline with judge selection for Salony dataset",
    )

    pipeline.add_step(LoadDataFrame(name="load", df=df))

    pipeline.add_step(
        ValidateUserStories(
            name="validate_format", story_column="input", case_sensitive=False
        )
    )

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

    pipeline.add_step(
        ComparisonJudgeStep(
            name="comparison_judge",
            model_name=judge_model,
            input_column="input",
            output_a_column="tasks_generator_a",
            output_b_column="tasks_generator_b",
            prompt_template_func=create_comparison_judge_prompt,
            system_prompt="You are an expert software development manager who evaluates and compares different task breakdowns to determine which is superior.",
            batch_size=max(1, batch_size // 2),
            generation_kwargs={
                "temperature": 0.2,
                "num_predict": 1000,
            },
        )
    )

    # US ID tracking
    us_counter = {"count": 0}

    def get_us_id():
        us_counter["count"] += 1
        return us_counter["count"]

    pipeline.add_step(
        AddColumn(
            name="add_us_id", input_columns=[], output_column="us_id", func=get_us_id
        )
    )

    # --- Execute ----------------------------------------------------------
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

    # --- Annotate timestamps ----------------------------------------------
    result_df["start_timestamp"] = start_timestamp
    result_df["end_timestamp"] = end_timestamp

    result_df["total_time"] = (
        result_df["generation_time_a"]
        + result_df["generation_time_b"]
        + result_df["judge_time"]
    )

    # --- Judge results CSV ------------------------------------------------
    judge_results_df = result_df[
        [
            "us_id", "input",
            "judge_score_a_total", "judge_score_b_total",
            "judge_score_a_coherence", "judge_score_a_completeness",
            "judge_score_a_feasibility", "judge_score_a_format",
            "judge_score_a_granularity",
            "judge_score_b_coherence", "judge_score_b_completeness",
            "judge_score_b_feasibility", "judge_score_b_format",
            "judge_score_b_granularity",
            "judge_strengths_a", "judge_weaknesses_a",
            "judge_strengths_b", "judge_weaknesses_b",
            "judge_winner", "judge_reason",
            "generation_time_a", "generation_time_b",
            "judge_time", "total_time",
            "start_timestamp", "end_timestamp",
        ]
    ].copy()

    judge_output_stem = Path(output_csv).stem + "_judge_results.csv"
    judge_output_full = str(Path(output_csv).parent / judge_output_stem)

    logging.info(f"\nSaving judge results to: {judge_output_full}")
    try:
        judge_results_df.to_csv(judge_output_full, index=False)
        logging.info(f"✅ Judge results saved ({len(judge_results_df)} user stories)")
    except Exception as e:
        raise RuntimeError(f"Failed to save judge results CSV: {e}")

    # --- Explode tasks from both generators -------------------------------
    explode_step = ExplodeTasks(
        name="explode",
        tasks_column="tasks",
        output_column="task",
        group_by_column="us_id",
    )

    df_a = result_df[["us_id", "input", "tasks_generator_a"]].copy()
    df_a = df_a.rename(columns={"tasks_generator_a": "tasks"})
    df_a_exploded = explode_step.process(df_a)
    df_a_exploded["generator"] = "A"
    df_a_exploded["generator_model"] = model_a
    df_a_exploded = df_a_exploded.rename(columns={"task": "task_generator_a"})

    df_b = result_df[["us_id", "input", "tasks_generator_b"]].copy()
    df_b = df_b.rename(columns={"tasks_generator_b": "tasks"})
    df_b_exploded = explode_step.process(df_b)
    df_b_exploded["generator"] = "B"
    df_b_exploded["generator_model"] = model_b
    df_b_exploded = df_b_exploded.rename(columns={"task": "task_generator_b"})

    logging.info("Merging exploded tasks...")
    tasks_df = pd.merge(
        df_a_exploded[["us_id", "input", "task_id", "task_generator_a"]],
        df_b_exploded[["us_id", "input", "task_id", "task_generator_b"]],
        on=["us_id", "task_id"],
        how="outer",
        suffixes=("", "_b"),
    )

    if "input_b" in tasks_df.columns:
        tasks_df["input"] = tasks_df["input"].fillna(tasks_df["input_b"])
        tasks_df = tasks_df.drop(columns=["input_b"])

    tasks_df = tasks_df[
        ["us_id", "task_id", "input", "task_generator_a", "task_generator_b"]
    ]

    logging.info(f"\nSaving tasks comparison to: {output_csv}")
    try:
        tasks_df.to_csv(output_csv, index=False)
        logging.info(
            f"✅ Tasks saved ({len(tasks_df)} tasks from "
            f"{tasks_df['us_id'].nunique()} user stories)"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save tasks CSV: {e}")

    # --- Statistics -------------------------------------------------------
    if len(result_df) > 0:
        winner_counts = result_df["judge_winner"].value_counts()
        avg_score_a = result_df["judge_score_a_total"].mean()
        avg_score_b = result_df["judge_score_b_total"].mean()

        logging.info(f"\n=== JUDGE SELECTION STATISTICS ===")
        logging.info(
            f"Generator A ({model_a}) wins: {winner_counts.get('A', 0)} "
            f"({winner_counts.get('A', 0) / len(result_df) * 100:.1f}%)"
        )
        logging.info(
            f"Generator B ({model_b}) wins: {winner_counts.get('B', 0)} "
            f"({winner_counts.get('B', 0) / len(result_df) * 100:.1f}%)"
        )
        logging.info(
            f"Average total scores: A={avg_score_a:.1f}/50, B={avg_score_b:.1f}/50"
        )

        cfg_display = {
            "coherence": "Coherencia",
            "completeness": "Completitud",
            "feasibility": "Viabilidad",
            "format": "Formato",
            "granularity": "Granularidad",
        }
        logging.info(f"\n=== AVERAGE SCORES BY CRITERIA (out of 10) ===")
        for criterion, display_name in cfg_display.items():
            avg_a = result_df[f"judge_score_a_{criterion}"].mean()
            avg_b = result_df[f"judge_score_b_{criterion}"].mean()
            logging.info(f"{display_name:20s}: A={avg_a:.1f}, B={avg_b:.1f}")

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

    logging.info("\n✅ Dual generator pipeline with judge comparison completed!")
    logging.info(f"   - Tasks comparison: {output_csv}")
    logging.info(f"   - Judge results: {judge_output_full}")

    return tasks_df, judge_results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate development tasks using dual generators with judge selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python salony_dual_generator_pipeline.py output.csv
  python salony_dual_generator_pipeline.py output.csv --model-a llama3.1:8b --model-b mistral
  python salony_dual_generator_pipeline.py output.csv --sample 5 --temperature-a 0.3 --temperature-b 0.7
        """,
    )

    parser.add_argument("output_csv", help="Path to save the generated tasks")
    parser.add_argument("--input-csv", help="Path to input CSV file")
    parser.add_argument(
        "--model-a", default="llama3.1:8b",
        help="First Ollama model for task generation (default: llama3.1:8b)",
    )
    parser.add_argument(
        "--model-b", default="qwen3:8b",
        help="Second Ollama model for task generation (default: qwen3:8b)",
    )
    parser.add_argument(
        "--judge-model", default="llama3.1:8b",
        help="Ollama model for judge comparison (default: llama3.1:8b)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Stories to process simultaneously (default: 2)",
    )
    parser.add_argument(
        "--temperature-a", type=float, default=0.3,
        help="Generation temperature for model A (default: 0.3)",
    )
    parser.add_argument(
        "--temperature-b", type=float, default=0.7,
        help="Generation temperature for model B (default: 0.7)",
    )
    parser.add_argument(
        "--num-predict", type=int, default=1000,
        help="Maximum tokens to generate (default: 1000)",
    )
    parser.add_argument(
        "--sample", type=int,
        help="Number of stories to process (useful for testing)",
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
