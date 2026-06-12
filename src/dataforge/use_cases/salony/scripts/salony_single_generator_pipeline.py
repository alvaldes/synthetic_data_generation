#!/usr/bin/env python3
"""
Script to generate development tasks from Salony dataset user stories.

This pipeline takes user stories from the salony_train.csv dataset and
breaks them down into smaller, actionable development tasks.

Usage:
    python salony_single_generator_pipeline.py
    python salony_single_generator_pipeline.py --output results.csv
    python salony_single_generator_pipeline.py --model llama3.1:8b --batch-size 4
    python salony_single_generator_pipeline.py --sample 10
"""

import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional
import ollama

from dataforge import DataForgePipeline
from dataforge.transformers import LoadDataFrame, AddColumn, ExplodeTasks
from dataforge.llm import OllamaLLMStep, OllamaJudgeStep
from dataforge.validators import ValidateUserStories
from dataforge.config import get_settings, PromptLoader


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def create_task_generation_prompt(row: Dict) -> str:
    """
    Render the task generation prompt template with the user story.

    Template loaded from ``config/prompts/task_generation.j2``.
    """
    return PromptLoader.render(
        "task_generation.j2",
        {"user_story": row.get("input", "").strip()},
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_inputs(
    model_name: str,
    batch_size: int,
    temperature: float,
    num_predict: int,
    judge_threshold: Optional[float] = None,
) -> None:
    """Validate input parameters."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got: {batch_size}")
    if not (0.0 <= temperature <= 2.0):
        raise ValueError(
            f"temperature must be between 0.0 and 2.0, got: {temperature}"
        )
    if num_predict <= 0:
        raise ValueError(f"num_predict must be positive, got: {num_predict}")
    if judge_threshold is not None and not (0.0 <= judge_threshold <= 50.0):
        raise ValueError(
            f"judge_threshold must be between 0.0 and 50.0, got: {judge_threshold}"
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

def run_salony_pipeline(
    output_csv: Optional[str] = None,
    input_csv: Optional[str] = None,
    model_name: Optional[str] = None,
    judge_model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
    sample_size: Optional[int] = None,
    use_judge: bool = False,
    judge_threshold: Optional[float] = None,
    use_cache: bool = True,
):
    """
    Execute Salony pipeline to generate development tasks from user stories.

    Parameters
    ----------
    output_csv : str, optional
        Path to save the results (defaults to ``<paths.output_dir>/salony_tasks.csv``).
    input_csv : str, optional
        Path to input CSV (defaults to ``<paths.raw_dir>/salony_train.csv``).
    model_name : str, optional
        Ollama model for task generation (default from config).
    judge_model_name : str, optional
        Model for validation (defaults to *model_name*).
    batch_size : int, optional
        Stories to process simultaneously (default from config).
    temperature : float, optional
        Generation temperature (default from config).
    num_predict : int, optional
        Maximum tokens to generate (default from config).
    sample_size : int, optional
        Process only N stories (for testing).
    use_judge : bool
        Whether to enable LLM judge validation.
    judge_threshold : float, optional
        Approval threshold for judge (0-50, default from config).
    use_cache : bool
        Whether to use caching.

    Returns
    -------
    pd.DataFrame
    """
    # --- Resolve config ---------------------------------------------------
    cfg = get_settings(use_case="salony")
    llm_cfg = cfg.llm
    judge_cfg = cfg.judge

    model_name = model_name or llm_cfg.default_model
    batch_size = batch_size if batch_size is not None else llm_cfg.batch_size
    temperature = temperature if temperature is not None else llm_cfg.temperature
    num_predict = num_predict if num_predict is not None else llm_cfg.num_predict
    judge_threshold = (
        judge_threshold if judge_threshold is not None
        else judge_cfg.approval_threshold
    )

    # Resolve output path
    if output_csv is None:
        output_csv = str(Path(cfg.paths.output_dir) / "salony_tasks.csv")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    # --- Setup ------------------------------------------------------------
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    validate_inputs(
        model_name,
        batch_size,
        temperature,
        num_predict,
        judge_threshold if use_judge else None,
    )

    if use_judge and judge_model_name is None:
        judge_model_name = model_name

    # Determine input file path
    if input_csv is None:
        input_path = Path(cfg.paths.raw_dir) / "salony_train.csv"
    else:
        input_path = Path(input_csv)

    logging.info(f"Loading data from: {input_path}")

    # --- Ollama connectivity check ----------------------------------------
    try:
        client = ollama.Client()
        try:
            models = client.list()
            available_models = [m.model for m in models.models]
            if model_name not in available_models:
                logging.warning(
                    f"Model {model_name} not found locally. Attempting to pull..."
                )
                client.pull(model_name)
        except ollama.ResponseError as e:
            if "connection refused" in str(e).lower():
                raise ConnectionError(
                    "Ollama server not running. Please start with: ollama serve"
                )
            raise
    except ConnectionError:
        raise ConnectionError(
            "Cannot connect to Ollama. Ensure it's running with: ollama serve"
        )

    # --- Load data --------------------------------------------------------
    df = load_and_validate_data(input_path, sample_size)
    logging.info(f"Loaded {len(df)} user stories")

    # --- Build pipeline ---------------------------------------------------
    pipeline_name = "salony-tasks-pipeline"
    if use_judge:
        pipeline_name += "-with-judge"

    logging.info(f"Configuring pipeline: {model_name}, batch_size={batch_size}")
    if use_judge:
        logging.info(
            f"Judge validation enabled: {judge_model_name}, threshold={judge_threshold}"
        )

    pipeline = DataForgePipeline(
        name=pipeline_name,
        description="Pipeline for generating and validating development tasks from Salony dataset",
    )

    pipeline.add_step(LoadDataFrame(name="load", df=df))

    pipeline.add_step(
        ValidateUserStories(
            name="validate_format", story_column="input", case_sensitive=False
        )
    )

    # US ID tracking (counter starting from 1)
    us_counter = {"count": 0}

    def get_us_id():
        us_counter["count"] += 1
        return us_counter["count"]

    pipeline.add_step(
        AddColumn(
            name="add_us_id",
            input_columns=[],
            output_column="us_id",
            func=get_us_id,
        )
    )

    pipeline.add_step(
        AddColumn(
            name="add_generator_model",
            input_columns=[],
            output_column="generator_model_name",
            func=lambda: model_name,
        )
    )

    pipeline.add_step(
        OllamaLLMStep(
            name="generate_tasks",
            model_name=model_name,
            prompt_column="input",
            output_column="tasks",
            prompt_template=create_task_generation_prompt,
            system_prompt="You are an expert software development lead who excels at breaking down user stories into clear, actionable development tasks.",
            batch_size=batch_size,
            generation_kwargs={"temperature": temperature, "num_predict": num_predict},
        )
    )

    pipeline.add_step(
        ExplodeTasks(name="explode_tasks", tasks_column="tasks", output_column="task")
    )

    if use_judge:
        pipeline.add_step(
            AddColumn(
                name="add_judge_model",
                input_columns=[],
                output_column="judge_model_name",
                func=lambda: judge_model_name,
            )
        )

        pipeline.add_step(
            OllamaJudgeStep(
                name="validate_tasks",
                model_name=judge_model_name,
                historia_usuario_column="input",
                tareas_generadas_column="task",
                approval_threshold=judge_threshold,
                batch_size=max(1, batch_size // 2),
                generation_kwargs={
                    "temperature": 0.2,
                    "num_predict": 800,
                },
            )
        )

    # --- Execute ----------------------------------------------------------
    logging.info("Executing pipeline...")
    try:
        result_df = pipeline.run(use_cache=use_cache)
    except ollama.ResponseError as e:
        raise RuntimeError(f"Ollama API error: {e.error}")
    except Exception as e:
        raise RuntimeError(f"Pipeline execution failed: {e}")

    if "us_id" in result_df.columns:
        cols = ["us_id"] + [col for col in result_df.columns if col != "us_id"]
        result_df = result_df[cols]

    logging.info("Saving results...")
    try:
        result_df.to_csv(output_csv, index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save CSV: {e}")

    logging.info(f"Results saved to: {output_csv}")
    logging.info(f"Processed {len(result_df)} tasks successfully")

    if use_judge and "validacion_aprobado" in result_df.columns:
        approved = result_df["validacion_aprobado"].sum()
        total = len(result_df)
        avg_score = result_df["validacion_total"].mean()
        logging.info(
            f"Validation: {approved}/{total} approved "
            f"({approved / total * 100:.1f}%)"
        )
        logging.info(f"Average score: {avg_score:.1f}/50")

    return result_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate development tasks from user stories in the Salony dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python salony_single_generator_pipeline.py
  python salony_single_generator_pipeline.py --output results.csv
  python salony_single_generator_pipeline.py --model mistral
  python salony_single_generator_pipeline.py --use-judge --judge-threshold 40
  python salony_single_generator_pipeline.py --batch-size 4 --temperature 0.6
  python salony_single_generator_pipeline.py --sample 10
        """,
    )

    parser.add_argument(
        "--output", dest="output_csv", nargs="?", default=None,
        help="Path to save the generated tasks (default: data/outputs/salony_tasks.csv)",
    )
    parser.add_argument("--input-csv", help="Path to input CSV file")
    parser.add_argument(
        "--model", default="llama3.1:8b",
        help="Ollama model for task generation (default: llama3.1:8b)",
    )
    parser.add_argument("--use-judge", action="store_true", help="Enable LLM judge validation")
    parser.add_argument(
        "--judge-model",
        help="Ollama model for judge validation (defaults to same as --model)",
    )
    parser.add_argument(
        "--judge-threshold", type=float, default=35.0,
        help="Approval threshold for judge (0-50, default: 35.0)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Stories to process simultaneously (default: 2)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Generation temperature (default: 0.3)",
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
        run_salony_pipeline(
            output_csv=args.output_csv,
            input_csv=args.input_csv,
            model_name=args.model,
            judge_model_name=args.judge_model,
            batch_size=args.batch_size,
            temperature=args.temperature,
            num_predict=args.num_predict,
            sample_size=args.sample,
            use_judge=args.use_judge,
            judge_threshold=args.judge_threshold,
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
