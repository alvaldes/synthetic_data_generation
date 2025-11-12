#!/usr/bin/env python3
"""
Script to generate development tasks from Salony dataset user stories.

This pipeline takes user stories from the salony_train.csv dataset and breaks them down
into smaller, actionable development tasks.

Usage:
    python salony_pipeline.py output.csv
    python salony_pipeline.py output.csv --model llama3.1:8b --batch-size 4
    python salony_pipeline.py output.csv --sample 10
"""

import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional
import ollama

from simple_pipeline import SimplePipeline
from simple_pipeline.steps import LoadDataFrame, OllamaLLMStep, OllamaJudgeStep, AddColumn


def create_task_generation_prompt(row: Dict) -> str:
    """
    Crea el prompt para generar tareas a partir de una historia de usuario del dataset Salony.
    
    Args:
        row: Fila del DataFrame con la columna 'input' que contiene la historia
    
    Returns:
        Prompt formateado
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


def validate_inputs(
    model_name: str,
    batch_size: int,
    temperature: float,
    num_predict: int,
    judge_threshold: float = None
) -> None:
    """Validate input parameters."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got: {batch_size}")
    if not (0.0 <= temperature <= 2.0):
        raise ValueError(f"temperature must be between 0.0 and 2.0, got: {temperature}")
    if num_predict <= 0:
        raise ValueError(f"num_predict must be positive, got: {num_predict}")
    if judge_threshold is not None and not (0.0 <= judge_threshold <= 50.0):
        raise ValueError(f"judge_threshold must be between 0.0 and 50.0, got: {judge_threshold}")


def load_and_validate_data(input_csv: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load and validate the input CSV data with robust error handling."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Dataset not found: {input_csv}")

    try:
        # Use pandas error handling best practices
        df = pd.read_csv(input_csv)
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {input_csv}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file {input_csv}: {e}")

    # Remove unnamed index columns (common issue with exported CSVs)
    if len(df.columns) > 0 and df.columns[0] in ['Unnamed: 0', '']:
        df = df.iloc[:, 1:]

    # Validate required column exists
    if 'input' not in df.columns:
        raise ValueError(
            f"CSV must have an 'input' column with user stories. "
            f"Found columns: {list(df.columns)}"
        )

    # Apply sampling if requested (for testing)
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got: {sample_size}")
        df = df.head(sample_size)
        logging.info(f"Using sample of {sample_size} rows for testing")

    # Clean data using pandas best practices
    initial_count = len(df)

    # Handle missing data with pandas errors='coerce' pattern
    df = df.dropna(subset=['input'])
    df['input'] = df['input'].astype(str).str.strip()

    # Remove empty strings after stripping
    df = df[df['input'] != '']

    final_count = len(df)
    if final_count < initial_count:
        logging.warning(f"Removed {initial_count - final_count} rows with missing/empty input data")

    if final_count == 0:
        raise ValueError("No valid user stories found after data cleaning")

    return df


def run_salony_pipeline(
    output_csv: str,
    input_csv: Optional[str] = None,
    model_name: str = "llama3.1:8b",
    judge_model_name: Optional[str] = None,
    batch_size: int = 2,
    temperature: float = 0.3,
    num_predict: int = 1000,
    sample_size: Optional[int] = None,
    use_judge: bool = False,
    judge_threshold: float = 35.0,
    use_cache: bool = True
):
    """
    Execute Salony pipeline to generate development tasks from user stories.

    Args:
        output_csv: Path to save the results
        input_csv: Optional path to input CSV file (defaults to data/salony_train.csv)
        model_name: Ollama model for task generation
        judge_model_name: Optional model for validation (defaults to same as model_name)
        batch_size: Number of stories to process simultaneously
        temperature: Generation temperature
        num_predict: Maximum tokens to generate
        sample_size: If specified, process only N stories (for testing)
        use_judge: Whether to enable LLM judge validation
        judge_threshold: Approval threshold for judge (0-50)
        use_cache: Whether to use caching
    """

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Validate inputs first
    validate_inputs(model_name, batch_size, temperature, num_predict,
                   judge_threshold if use_judge else None)

    # Set default judge model
    if use_judge and judge_model_name is None:
        judge_model_name = model_name

    # Determine input file path
    if input_csv is None:
        input_path = Path(__file__).parent.parent / "data" / "salony_train.csv"
    else:
        input_path = Path(input_csv)

    logging.info(f"Loading data from: {input_path}")

    # Test Ollama connection early
    try:
        client = ollama.Client()
        # Test if model is available
        try:
            models = client.list()
            available_models = [m['name'] for m in models['models']]
            if model_name not in available_models:
                logging.warning(f"Model {model_name} not found locally. Attempting to pull...")
                client.pull(model_name)
        except ollama.ResponseError as e:
            if "connection refused" in str(e).lower():
                raise ConnectionError("Ollama server not running. Please start with: ollama serve")
            raise
    except ConnectionError:
        raise ConnectionError("Cannot connect to Ollama. Ensure it's running with: ollama serve")

    # Load and validate data
    df = load_and_validate_data(input_path, sample_size)
    logging.info(f"Loaded {len(df)} user stories")

    # Configure pipeline
    pipeline_name = "salony-tasks-pipeline"
    if use_judge:
        pipeline_name += "-with-judge"

    logging.info(f"Configuring pipeline: {model_name}, batch_size={batch_size}")
    if use_judge:
        logging.info(f"Judge validation enabled: {judge_model_name}, threshold={judge_threshold}")

    pipeline = SimplePipeline(
        name=pipeline_name,
        description="Pipeline for generating and validating development tasks from Salony dataset"
    )

    # Add data loading step
    pipeline.add_step(
        LoadDataFrame(name="load", df=df)
    )

    # Add generator model tracking
    pipeline.add_step(
        AddColumn(
            name="add_generator_model",
            input_columns=[],
            output_column="generator_model_name",
            func=lambda: model_name
        )
    )

    # Add task generation step
    pipeline.add_step(
        OllamaLLMStep(
            name="generate_tasks",
            model_name=model_name,
            prompt_column="input",
            output_column="tasks",
            prompt_template=create_task_generation_prompt,
            system_prompt="You are an expert software development lead who excels at breaking down user stories into clear, actionable development tasks.",
            batch_size=batch_size,
            generation_kwargs={
                "temperature": temperature,
                "num_predict": num_predict
            },
        )
    )

    # Add judge validation if enabled
    if use_judge:
        pipeline.add_step(
            AddColumn(
                name="add_judge_model",
                input_columns=[],
                output_column="judge_model_name",
                func=lambda: judge_model_name
            )
        )

        pipeline.add_step(
            OllamaJudgeStep(
                name="validate_tasks",
                model_name=judge_model_name,
                historia_usuario_column="input",
                tareas_generadas_column="tasks",
                approval_threshold=judge_threshold,
                batch_size=max(1, batch_size // 2),  # Smaller batches for judge
                generation_kwargs={
                    "temperature": 0.2,  # Lower temperature for more consistent judging
                    "num_predict": 800
                }
            )
        )

    # Execute pipeline
    logging.info("Executing pipeline...")
    try:
        result_df = pipeline.run(use_cache=use_cache)
    except ollama.ResponseError as e:
        raise RuntimeError(f"Ollama API error: {e.error}")
    except Exception as e:
        raise RuntimeError(f"Pipeline execution failed: {e}")

    # Save results
    logging.info("Saving results...")
    try:
        result_df.to_csv(output_csv, index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save CSV: {e}")

    logging.info(f"Results saved to: {output_csv}")
    logging.info(f"Processed {len(result_df)} user stories successfully")

    # Show validation statistics if judge was used
    if use_judge and 'validacion_aprobado' in result_df.columns:
        approved = result_df['validacion_aprobado'].sum()
        total = len(result_df)
        avg_score = result_df['validacion_total'].mean()
        logging.info(f"Validation: {approved}/{total} approved ({approved/total*100:.1f}%)")
        logging.info(f"Average score: {avg_score:.1f}/50")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate development tasks from user stories in the Salony dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python salony_pipeline.py output.csv
  python salony_pipeline.py output.csv --model mistral
  python salony_pipeline.py output.csv --use-judge --judge-threshold 40
  python salony_pipeline.py output.csv --input-csv custom_data.csv
  python salony_pipeline.py output.csv --batch-size 4 --temperature 0.6
  python salony_pipeline.py output.csv --sample 10  # For testing

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
        '--model',
        default='llama3.1:8b',
        help='Ollama model for task generation (default: llama3.1:8b)'
    )

    parser.add_argument(
        '--use-judge',
        action='store_true',
        help='Enable LLM judge validation'
    )

    parser.add_argument(
        '--judge-model',
        help='Ollama model for judge validation (defaults to same as --model)'
    )

    parser.add_argument(
        '--judge-threshold',
        type=float,
        default=35.0,
        help='Approval threshold for judge (0-50, default: 35.0)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Stories to process simultaneously (default: 2)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Generation temperature (default: 0.3)'
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
