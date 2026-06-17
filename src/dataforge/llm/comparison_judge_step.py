import json
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple
import ollama
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..base_step import BaseStep
from ..config import get_settings, PromptLoader

_thread_local = threading.local()
from ..utils.batching import batch_dataframe, get_num_batches
from ..transformers.json_repair import clean_json_response, repair_json, parse_json_with_repair


class ComparisonJudgeStep(BaseStep):
    """
    Step that compares outputs from two generators using an LLM judge and selects the best one.

    The judge prompt is loaded from :file:`config/prompts/comparison_judge.j2`.
    Default parameters (model, batch size, temperature, column prefix) come
    from the centralised :ref:`config <DataForgeSettings>`.
    """

    PROMPT_TEMPLATE = "comparison_judge.j2"

    def __init__(
        self,
        name: str,
        model_name: Optional[str] = None,
        input_column: Optional[str] = None,
        output_a_column: Optional[str] = None,
        output_b_column: Optional[str] = None,
        prompt_template_func: Optional[Callable] = None,
        system_prompt: Optional[str] = None,
        batch_size: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        num_workers: Optional[int] = None,
        max_zero_retries: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        cfg = get_settings()
        llm_cfg = cfg.llm
        comp_cfg = cfg.comparison_judge

        self.model_name = model_name or comp_cfg.model or llm_cfg.default_model
        self.input_column = input_column
        self.output_a_column = output_a_column
        self.output_b_column = output_b_column
        self.prompt_template_func = prompt_template_func or self._default_prompt_func
        self.system_prompt = system_prompt or (
            "You are an expert evaluator who compares different outputs "
            "and selects the best one based on quality criteria."
        )
        self.batch_size = batch_size if batch_size is not None else comp_cfg.batch_size
        self.generation_kwargs = generation_kwargs or {
            "temperature": comp_cfg.temperature,
            "num_predict": comp_cfg.num_predict,
        }
        self.num_workers = num_workers if num_workers is not None else comp_cfg.num_workers
        self.max_zero_retries = max_zero_retries if max_zero_retries is not None else comp_cfg.max_zero_retries

        self.client = None

    # ------------------------------------------------------------------
    # Default prompt (uses template file)
    # ------------------------------------------------------------------

    def _default_prompt_func(self, row: Dict[str, Any]) -> str:
        """Render the comparison judge template with row data."""
        return PromptLoader.render(
            self.PROMPT_TEMPLATE,
            {
                "user_story": str(row.get(self.input_column, "")),
                "tasks_a": str(row.get(self.output_a_column, "")),
                "tasks_b": str(row.get(self.output_b_column, "")),
            },
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def inputs(self) -> List[str]:
        return [self.input_column, self.output_a_column, self.output_b_column]

    @property
    def outputs(self) -> List[str]:
        p = get_settings().comparison_judge.column_prefix
        return [
            f"{p}score_a_total",
            f"{p}score_b_total",
            f"{p}score_a_coherence",
            f"{p}score_a_completeness",
            f"{p}score_a_feasibility",
            f"{p}score_a_format",
            f"{p}score_a_granularity",
            f"{p}score_b_coherence",
            f"{p}score_b_completeness",
            f"{p}score_b_feasibility",
            f"{p}score_b_format",
            f"{p}score_b_granularity",
            f"{p}strengths_a",
            f"{p}weaknesses_a",
            f"{p}strengths_b",
            f"{p}weaknesses_b",
            f"{p}winner",
            f"{p}reason",
            "selected_output",
            f"{p}time",
        ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Initialize Ollama client."""
        super().load()
        try:
            self.client = ollama.Client()
            self.client.list()
            logging.info(f"Connected to Ollama for judge model: {self.model_name}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")

    def unload(self) -> None:
        """Clean up resources."""
        self.client = None
        super().unload()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _has_zero_score(self, result: Dict) -> bool:
        """Check if any individual criterion has a score of 0 in either breakdown."""
        score_fields = ["coherence", "completeness", "feasibility", "format", "granularity"]
        for breakdown_key in ["breakdown_a", "breakdown_b"]:
            breakdown = result.get(breakdown_key, {})
            if isinstance(breakdown, dict):
                for field in score_fields:
                    if field in breakdown and breakdown[field] == 0:
                        return True
        return False

    def _parse_judge_response(self, raw_response: str) -> Dict:
        """Parse the judge response JSON with robust error handling and repair."""
        result = parse_json_with_repair(raw_response, logger=logging.getLogger(__name__))

        if result is not None:
            try:
                return self._validate_and_normalize_judge_result(result)
            except (ValueError, KeyError) as e:
                logging.warning(f"Judge result validation failed: {e}")

        return {
            "breakdown_a": {
                "total_score": -5,
                "coherence": -1,
                "completeness": -1,
                "feasibility": -1,
                "format": -1,
                "granularity": -1,
                "strengths": "Unable to parse",
                "weaknesses": "Parse error",
            },
            "breakdown_b": {
                "total_score": -5,
                "coherence": -1,
                "completeness": -1,
                "feasibility": -1,
                "format": -1,
                "granularity": -1,
                "strengths": "Unable to parse",
                "weaknesses": "Parse error",
            },
            "winner": "A",
            "reason": "Failed to parse judge response",
            "parse_error": True,
        }

    def _validate_and_normalize_judge_result(self, result: Dict) -> Dict:
        """Validate and normalize the parsed judge result."""
        required_top_level = ["breakdown_a", "breakdown_b", "winner", "reason"]
        for field in required_top_level:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        score_fields = ["coherence", "completeness", "feasibility", "format", "granularity"]

        for breakdown_key in ["breakdown_a", "breakdown_b"]:
            breakdown = result[breakdown_key]

            # Validate individual score fields exist
            for field in score_fields:
                if field not in breakdown:
                    raise ValueError(f"Missing score field '{field}' in {breakdown_key}")

            # Always calculate total in code — override LLM's value
            code_total = sum(breakdown[field] for field in score_fields)

            if "total_score" in breakdown and breakdown["total_score"] != code_total:
                logging.warning(
                    f"Discrepancia en total_score para {breakdown_key}: "
                    f"LLM={breakdown['total_score']}, código={code_total}. "
                    f"Usando valor del código."
                )

            breakdown["total_score"] = code_total

        if result["winner"] not in ["A", "B"]:
            result["winner"] = "A"
            result["reason"] = "Invalid winner designation, defaulting to A"

        return result

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _get_thread_client(self) -> ollama.Client:
        """Get thread-local Ollama client for parallel execution."""
        if not hasattr(_thread_local, "client"):
            _thread_local.client = ollama.Client()
        return _thread_local.client

    def _judge_comparison(
        self, input_text: str, output_a: str, output_b: str,
        client: Optional[ollama.Client] = None,
        zero_retry_count: int = 0,
    ) -> Dict:
        """Use LLM to judge which output is better.

        Args:
            input_text: The original input/user story.
            output_a: Output from generator A.
            output_b: Output from generator B.
            client: Optional Ollama client instance. Falls back to self.client if None.
            zero_retry_count: Current retry count for zero-score retries.
        """
        client = client or self.client
        row_data = {
            self.input_column: input_text,
            self.output_a_column: output_a,
            self.output_b_column: output_b,
        }

        prompt = self.prompt_template_func(row_data)

        if zero_retry_count > 0:
            prompt += (
                f"\n\n**⚠️ RETRY {zero_retry_count} — CRITICAL:**\n"
                "In your previous attempt you gave a score of 0 in at least one criterion.\n"
                "EACH criterion MUST have a value between 1 and 10. 0 is NOT allowed.\n"
                "Please re-evaluate carefully, ensuring every score is between 1 and 10.\n"
                "If a criterion truly does not apply, use 1 as the minimum.\n"
            )

        try:
            response = client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                options=self.generation_kwargs,
            )

            raw_response = response["message"]["content"]
            judgment = self._parse_judge_response(raw_response)

            # Zero-score retry
            if (
                "parse_error" not in judgment
                and zero_retry_count < self.max_zero_retries
                and self._has_zero_score(judgment)
            ):
                logging.warning(
                    f"Detected zero score in judge comparison "
                    f"(retry {zero_retry_count + 1}/{self.max_zero_retries}). "
                    f"Retrying with corrected prompt."
                )
                return self._judge_comparison(
                    input_text, output_a, output_b,
                    client=client, zero_retry_count=zero_retry_count + 1,
                )

            return judgment

        except Exception as e:
            logging.error(f"Judge evaluation failed: {e}")
            return {
                "breakdown_a": {
                    "total_score": -5,
                    "coherence": -1,
                    "completeness": -1,
                    "feasibility": -1,
                    "format": -1,
                    "granularity": -1,
                    "strengths": "Error occurred",
                    "weaknesses": str(e),
                },
                "breakdown_b": {
                    "total_score": -5,
                    "coherence": -1,
                    "completeness": -1,
                    "feasibility": -1,
                    "format": -1,
                    "granularity": -1,
                    "strengths": "Error occurred",
                    "weaknesses": str(e),
                },
                "winner": "A",
                "reason": f"Judge evaluation failed: {e}",
                "parse_error": True,
            }

    def _judge_comparison_parallel(
        self, input_text: str, output_a: str, output_b: str
    ) -> Tuple[Dict, float]:
        """Wrapper for parallel execution: uses thread-local client and tracks time."""
        client = self._get_thread_client()
        start = time.time()
        judgment = self._judge_comparison(input_text, output_a, output_b, client=client)
        judge_time = time.time() - start
        return judgment, judge_time

    def _process_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of rows for comparison judging."""
        results = []

        def ensure_scalar(value):
            if not isinstance(value, (int, float)):
                logging.warning(f"ensure_scalar encountered non-scalar value: {value}")
                return 0
            return value

        p = get_settings().comparison_judge.column_prefix

        if self.num_workers <= 1:
            # Sequential path — uses self.client
            for _, row in batch_df.iterrows():
                input_text = str(row[self.input_column])
                output_a = str(row[self.output_a_column])
                output_b = str(row[self.output_b_column])

                start_time = time.time()
                judgment = self._judge_comparison(input_text, output_a, output_b)
                judge_time = time.time() - start_time

                breakdown_a = judgment["breakdown_a"]
                breakdown_b = judgment["breakdown_b"]

                if not isinstance(breakdown_a, dict) or not isinstance(breakdown_b, dict):
                    logging.error("Invalid breakdown structure in judgment!")
                    continue

                results.append({
                    f"{p}score_a_total": ensure_scalar(breakdown_a.get("total_score", 0)),
                    f"{p}score_b_total": ensure_scalar(breakdown_b.get("total_score", 0)),
                    f"{p}score_a_coherence": breakdown_a.get("coherence", 0),
                    f"{p}score_a_completeness": breakdown_a.get("completeness", 0),
                    f"{p}score_a_feasibility": breakdown_a.get("feasibility", 0),
                    f"{p}score_a_format": breakdown_a.get("format", 0),
                    f"{p}score_a_granularity": breakdown_a.get("granularity", 0),
                    f"{p}score_b_coherence": breakdown_b.get("coherence", 0),
                    f"{p}score_b_completeness": breakdown_b.get("completeness", 0),
                    f"{p}score_b_feasibility": breakdown_b.get("feasibility", 0),
                    f"{p}score_b_format": breakdown_b.get("format", 0),
                    f"{p}score_b_granularity": breakdown_b.get("granularity", 0),
                    f"{p}strengths_a": breakdown_a.get("strengths", "N/A"),
                    f"{p}weaknesses_a": breakdown_a.get("weaknesses", "N/A"),
                    f"{p}strengths_b": breakdown_b.get("strengths", "N/A"),
                    f"{p}weaknesses_b": breakdown_b.get("weaknesses", "N/A"),
                    f"{p}winner": judgment["winner"],
                    f"{p}reason": judgment["reason"],
                    "selected_output": output_b if judgment["winner"] == "B" else output_a,
                    f"{p}time": judge_time,
                })
        else:
            # Parallel path — thread-local clients via ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                future_to_idx: Dict[Any, int] = {}
                # Store row data for ordered result assembly
                row_data_cache: Dict[int, Any] = {}
                for idx, (_, row) in enumerate(batch_df.iterrows()):
                    input_text = str(row[self.input_column])
                    output_a = str(row[self.output_a_column])
                    output_b = str(row[self.output_b_column])
                    row_data_cache[idx] = (output_a, output_b)
                    future = pool.submit(
                        self._judge_comparison_parallel, input_text, output_a, output_b
                    )
                    future_to_idx[future] = idx

                ordered: List[Optional[Dict]] = [None] * len(batch_df)

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    judgment, judge_time = future.result()
                    cached_a, cached_b = row_data_cache[idx]

                    breakdown_a = judgment["breakdown_a"]
                    breakdown_b = judgment["breakdown_b"]

                    if not isinstance(breakdown_a, dict) or not isinstance(breakdown_b, dict):
                        logging.error(f"Invalid breakdown structure in judgment for row {idx}!")
                        continue

                    ordered[idx] = {
                        f"{p}score_a_total": ensure_scalar(breakdown_a.get("total_score", 0)),
                        f"{p}score_b_total": ensure_scalar(breakdown_b.get("total_score", 0)),
                        f"{p}score_a_coherence": breakdown_a.get("coherence", 0),
                        f"{p}score_a_completeness": breakdown_a.get("completeness", 0),
                        f"{p}score_a_feasibility": breakdown_a.get("feasibility", 0),
                        f"{p}score_a_format": breakdown_a.get("format", 0),
                        f"{p}score_a_granularity": breakdown_a.get("granularity", 0),
                        f"{p}score_b_coherence": breakdown_b.get("coherence", 0),
                        f"{p}score_b_completeness": breakdown_b.get("completeness", 0),
                        f"{p}score_b_feasibility": breakdown_b.get("feasibility", 0),
                        f"{p}score_b_format": breakdown_b.get("format", 0),
                        f"{p}score_b_granularity": breakdown_b.get("granularity", 0),
                        f"{p}strengths_a": breakdown_a.get("strengths", "N/A"),
                        f"{p}weaknesses_a": breakdown_a.get("weaknesses", "N/A"),
                        f"{p}strengths_b": breakdown_b.get("strengths", "N/A"),
                        f"{p}weaknesses_b": breakdown_b.get("weaknesses", "N/A"),
                        f"{p}winner": judgment["winner"],
                        f"{p}reason": judgment["reason"],
                        "selected_output": cached_b if judgment["winner"] == "B" else cached_a,
                        f"{p}time": judge_time,
                    }

                results = ordered

        result_df = batch_df.copy()
        for key in results[0].keys():
            result_df[key] = [r[key] for r in results]

        return result_df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the DataFrame in batches for comparison judging."""
        results = []

        for col in [self.input_column, self.output_a_column, self.output_b_column]:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                logging.warning(f"Column '{col}' has {missing_count} missing values")

        num_batches = get_num_batches(df, self.batch_size)
        logging.info(
            f"Processing {len(df)} comparisons in {num_batches} batches of {self.batch_size}"
        )

        for i, batch_df in enumerate(batch_dataframe(df, self.batch_size)):
            logging.info(f"Processing judge batch {i + 1}/{num_batches}")

            try:
                batch_result = self._process_batch(batch_df)
                results.append(batch_result)
            except Exception as e:
                logging.error(f"Batch {i + 1} failed: {e}")
                fallback_df = batch_df.copy()
                p = get_settings().comparison_judge.column_prefix
                bsz = len(batch_df)
                fallback_df[f"{p}score_a_total"] = [-5] * bsz
                fallback_df[f"{p}score_b_total"] = [-5] * bsz
                fallback_df[f"{p}score_a_coherence"] = [-1] * bsz
                fallback_df[f"{p}score_a_completeness"] = [-1] * bsz
                fallback_df[f"{p}score_a_feasibility"] = [-1] * bsz
                fallback_df[f"{p}score_a_format"] = [-1] * bsz
                fallback_df[f"{p}score_a_granularity"] = [-1] * bsz
                fallback_df[f"{p}score_b_coherence"] = [-1] * bsz
                fallback_df[f"{p}score_b_completeness"] = [-1] * bsz
                fallback_df[f"{p}score_b_feasibility"] = [-1] * bsz
                fallback_df[f"{p}score_b_format"] = [-1] * bsz
                fallback_df[f"{p}score_b_granularity"] = [-1] * bsz
                fallback_df[f"{p}strengths_a"] = ["N/A"] * bsz
                fallback_df[f"{p}weaknesses_a"] = ["Error occurred"] * bsz
                fallback_df[f"{p}strengths_b"] = ["N/A"] * bsz
                fallback_df[f"{p}weaknesses_b"] = ["Error occurred"] * bsz
                fallback_df[f"{p}winner"] = ["A"] * bsz
                fallback_df[f"{p}reason"] = [f"Batch processing failed: {e}"] * bsz
                fallback_df["selected_output"] = batch_df[self.output_a_column].tolist()
                fallback_df[f"{p}time"] = [0.0] * bsz
                results.append(fallback_df)

        final_df = pd.concat(results, ignore_index=True)

        if len(final_df) > 0:
            p = get_settings().comparison_judge.column_prefix
            winner_counts = final_df[f"{p}winner"].value_counts()
            avg_score_a = final_df[f"{p}score_a_total"].mean()
            avg_score_b = final_df[f"{p}score_b_total"].mean()
            logging.info(
                f"Judge results: A wins: {winner_counts.get('A', 0)}, "
                f"B wins: {winner_counts.get('B', 0)}"
            )
            logging.info(f"Average scores: A={avg_score_a:.1f}, B={avg_score_b:.1f}")

        return final_df
