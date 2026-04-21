import json
import logging
import re
import pandas as pd
from typing import Dict, List, Optional, Any
import ollama
import time
from datetime import datetime

from ..base_step import BaseStep
from ..utils.batching import batch_dataframe, get_num_batches


class ComparisonJudgeStep(BaseStep):
    """
    Step that compares outputs from two generators using an LLM judge and selects the best one.

    This judge evaluates two different outputs for the same input and determines which one
    is superior based on multiple criteria. It returns the scores for both outputs and
    indicates which one should be selected.

    The step adds the following columns:
    - judge_score_a: Score for generator A output
    - judge_score_b: Score for generator B output
    - judge_winner: 'A' or 'B' indicating which generator won
    - judge_reason: Explanation of why the winner was selected
    - selected_output: The actual content from the winning generator
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        input_column: str,
        output_a_column: str,
        output_b_column: str,
        prompt_template_func,
        system_prompt: str = "You are an expert evaluator who compares different outputs and selects the best one based on quality criteria.",
        batch_size: int = 1,  # Smaller batches for judge due to longer prompts
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the comparison judge step.

        Args:
            name: Step name
            model_name: Ollama model for judging
            input_column: Column containing the original input/prompt
            output_a_column: Column containing output from generator A
            output_b_column: Column containing output from generator B
            prompt_template_func: Function to create judge prompt from row data
            system_prompt: System prompt for the judge
            batch_size: Number of rows to process in each batch
            generation_kwargs: Additional parameters for Ollama generation
        """
        super().__init__(name, **kwargs)
        self.model_name = model_name
        self.input_column = input_column
        self.output_a_column = output_a_column
        self.output_b_column = output_b_column
        self.prompt_template_func = prompt_template_func
        self.system_prompt = system_prompt
        self.batch_size = batch_size
        self.generation_kwargs = generation_kwargs or {}

        # Set default generation parameters optimized for judge tasks
        self.generation_kwargs.setdefault("temperature", 0.2)  # More consistent judging
        self.generation_kwargs.setdefault("num_predict", 1000)

        self.client = None

    @property
    def inputs(self) -> List[str]:
        return [self.input_column, self.output_a_column, self.output_b_column]

    @property
    def outputs(self) -> List[str]:
        return [
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
            "judge_score_b_granularity",
            # Qualitative feedback
            "judge_strengths_a",
            "judge_weaknesses_a",
            "judge_strengths_b",
            "judge_weaknesses_b",
            # Decision and reason
            "judge_winner",
            "judge_reason",
            "selected_output",
            # Timing metadata
            "judge_time",
        ]

    def load(self) -> None:
        """Initialize Ollama client."""
        super().load()
        try:
            self.client = ollama.Client()
            # Test connection
            self.client.list()
            logging.info(f"Connected to Ollama for judge model: {self.model_name}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")

    def unload(self) -> None:
        """Clean up resources."""
        self.client = None
        super().unload()

    def _clean_json_response(self, response: str) -> str:
        """Clean the response to extract valid JSON."""
        # Remove markdown if exists
        response = response.replace("```json", "").replace("```", "").strip()

        # Find JSON object boundaries
        start = response.find("{")
        end = response.rfind("}")

        if start != -1 and end != -1 and end > start:
            return response[start : end + 1]

        return response

    def _repair_json(self, json_str: str) -> str:
        """
        Attempt to repair common JSON errors from LLM output using a character-by-character
        state machine to properly handle string boundaries.

        Repairs:
        - Escapes newlines inside string values
        - Escapes control characters inside string values
        - Adds missing commas between fields
        - Removes trailing commas before closing braces/brackets
        """
        # First, extract JSON boundaries
        json_str = self._clean_json_response(json_str)

        # Character-by-character state machine
        result = []
        i = 0
        in_string = False
        escaped = False

        while i < len(json_str):
            char = json_str[i]

            if escaped:
                # Previous char was backslash, just copy and reset
                result.append(char)
                escaped = False
            elif char == "\\" and in_string:
                # Backslash inside string - escape it
                result.append(char)
                escaped = True
            elif char == '"':
                # Toggle string state
                in_string = not in_string
                result.append(char)
            elif in_string:
                # We're inside a string value
                if char in ("\n", "\r"):
                    # Escape raw newlines inside strings
                    if char == "\n":
                        result.append("\\n")
                    else:
                        result.append("\\r")
                elif char == "\t":
                    result.append("\\t")
                elif char == "\0":
                    # Remove null bytes
                    pass
                else:
                    result.append(char)
            else:
                # Outside string - normal JSON
                result.append(char)

            i += 1

        json_str = "".join(result)

        # Now apply simple comma fixes (outside string context now)
        # Fix missing commas between fields (only outside strings)
        json_str = re.sub(r'(\d+)\s*"(\w+)":\s*', r'\1, "\2": ', json_str)

        # Fix missing commas after } before "field":
        json_str = re.sub(r'}\s*"(\w+)":\s*', r'}, "\1": ', json_str)

        # Fix missing commas after ] before "field":
        json_str = re.sub(r']\s*"(\w+)":\s*', r'], "\1": ', json_str)

        # Fix missing comma after string value before next field
        # Pattern: "string value"\n"next_field": -> "string value",\n"next_field":
        json_str = re.sub(r'"\s*\n\s*"(\w+)":\s*', r'",\n"\1": ', json_str)

        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        # Normalize whitespace around colons and commas
        json_str = re.sub(r"\s*,\s*", ", ", json_str)
        json_str = re.sub(r"\s*:\s*", ": ", json_str)

        return json_str

    def _parse_judge_response(self, raw_response: str) -> Dict:
        """Parse the judge response JSON with robust error handling and repair."""
        cleaned_response = self._clean_json_response(raw_response)

        # Try parsing directly first
        try:
            result = json.loads(cleaned_response)
            return self._validate_and_normalize_judge_result(result)
        except json.JSONDecodeError as e:
            logging.warning(f"Initial JSON parse failed: {e}")
            logging.debug(f"Raw response (first 500 chars): {raw_response[:500]}")

        # Try repaired JSON
        try:
            repaired_response = self._repair_json(cleaned_response)
            logging.debug(f"Repaired JSON (first 500 chars): {repaired_response[:500]}")

            result = json.loads(repaired_response)
            logging.info("Successfully parsed JSON after repair")
            return self._validate_and_normalize_judge_result(result)

        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse repaired JSON: {e}")
            logging.warning(f"Raw response (first 1000 chars): {raw_response[:1000]}")
            logging.warning(
                f"Repaired response (first 1000 chars): {repaired_response[:1000]}"
            )

            # Return fallback structure
            return {
                "breakdown_a": {
                    "total_score": 25,
                    "strengths": "Unable to parse",
                    "weaknesses": "Parse error",
                },
                "breakdown_b": {
                    "total_score": 25,
                    "strengths": "Unable to parse",
                    "weaknesses": "Parse error",
                },
                "winner": "A",
                "reason": f"Failed to parse judge response: {e}",
            }

    def _validate_and_normalize_judge_result(self, result: Dict) -> Dict:
        """Validate and normalize the parsed judge result."""
        # Validate required structure
        required_top_level = ["breakdown_a", "breakdown_b", "winner", "reason"]
        for field in required_top_level:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Validate breakdown structure
        for breakdown_key in ["breakdown_a", "breakdown_b"]:
            breakdown = result[breakdown_key]
            if "total_score" not in breakdown:
                # Calculate total if missing
                score_fields = [
                    "coherence",
                    "completeness",
                    "feasibility",
                    "format",
                    "granularity",
                ]
                total = sum(breakdown.get(field, 0) for field in score_fields)
                breakdown["total_score"] = total

        # Ensure winner is valid
        if result["winner"] not in ["A", "B"]:
            # Default to A if invalid
            result["winner"] = "A"
            result["reason"] = "Invalid winner designation, defaulting to A"

        return result

    def _judge_comparison(self, input_text: str, output_a: str, output_b: str) -> Dict:
        """Use LLM to judge which output is better."""
        # Create row-like dict for prompt template
        row_data = {
            self.input_column: input_text,
            self.output_a_column: output_a,
            self.output_b_column: output_b,
        }

        prompt = self.prompt_template_func(row_data)

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                options=self.generation_kwargs,
            )

            raw_response = response["message"]["content"]
            return self._parse_judge_response(raw_response)

        except Exception as e:
            logging.error(f"Judge evaluation failed: {e}")
            # Return fallback
            return {
                "breakdown_a": {
                    "total_score": 25,
                    "strengths": "Error occurred",
                    "weaknesses": str(e),
                },
                "breakdown_b": {
                    "total_score": 25,
                    "strengths": "Error occurred",
                    "weaknesses": str(e),
                },
                "winner": "A",
                "reason": f"Judge evaluation failed: {e}",
            }

    def _process_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of rows for comparison judging."""
        results = []

        # Define score criteria
        score_criteria = [
            "coherence",
            "completeness",
            "feasibility",
            "format",
            "granularity",
        ]

        def ensure_scalar(value):
            """
            Ensure the value is scalar (either int or float). If not, log and default to 0.
            """
            if not isinstance(value, (int, float)):
                logging.warning(f"ensure_scalar encountered non-scalar value: {value}")
                return 0
            return value

        for _, row in batch_df.iterrows():
            input_text = str(row[self.input_column])
            output_a = str(row[self.output_a_column])
            output_b = str(row[self.output_b_column])

            # Track judge time
            start_time = time.time()

            # Get judge evaluation
            judgment = self._judge_comparison(input_text, output_a, output_b)

            judge_time = time.time() - start_time

            # Extract breakdown for A
            breakdown_a = judgment["breakdown_a"]
            breakdown_b = judgment["breakdown_b"]

            # Ensure breakdown structures are valid\nif not isinstance(breakdown_a, dict) or not isinstance(breakdown_b, dict):\n    logging.error("Invalid breakdown structure in judgment!")\n    continue\n# Sanitize total scores
            score_a_total = ensure_scalar(breakdown_a.get("total_score", 0))
            score_b_total = ensure_scalar(breakdown_b.get("total_score", 0))

            # Extract individual criteria scores for A
            score_a_coherence = breakdown_a.get("coherence", 0)
            score_a_completeness = breakdown_a.get("completeness", 0)
            score_a_feasibility = breakdown_a.get("feasibility", 0)
            score_a_format = breakdown_a.get("format", 0)
            score_a_granularity = breakdown_a.get("granularity", 0)

            # Extract individual criteria scores for B
            score_b_coherence = breakdown_b.get("coherence", 0)
            score_b_completeness = breakdown_b.get("completeness", 0)
            score_b_feasibility = breakdown_b.get("feasibility", 0)
            score_b_format = breakdown_b.get("format", 0)
            score_b_granularity = breakdown_b.get("granularity", 0)

            # Extract qualitative feedback
            strengths_a = breakdown_a.get("strengths", "N/A")
            weaknesses_a = breakdown_a.get("weaknesses", "N/A")
            strengths_b = breakdown_b.get("strengths", "N/A")
            weaknesses_b = breakdown_b.get("weaknesses", "N/A")

            winner = judgment["winner"]
            reason = judgment["reason"]

            # Select the winning output
            if winner == "B":
                selected_output = output_b
            else:  # Default to A
                selected_output = output_a

            results.append(
                {
                    # Total scores
                    "judge_score_a_total": score_a_total,
                    "judge_score_b_total": score_b_total,
                    # Individual scores for A
                    "judge_score_a_coherence": score_a_coherence,
                    "judge_score_a_completeness": score_a_completeness,
                    "judge_score_a_feasibility": score_a_feasibility,
                    "judge_score_a_format": score_a_format,
                    "judge_score_a_granularity": score_a_granularity,
                    # Individual scores for B
                    "judge_score_b_coherence": score_b_coherence,
                    "judge_score_b_completeness": score_b_completeness,
                    "judge_score_b_feasibility": score_b_feasibility,
                    "judge_score_b_format": score_b_format,
                    "judge_score_b_granularity": score_b_granularity,
                    # Qualitative feedback
                    "judge_strengths_a": strengths_a,
                    "judge_weaknesses_a": weaknesses_a,
                    "judge_strengths_b": strengths_b,
                    "judge_weaknesses_b": weaknesses_b,
                    # Decision
                    "judge_winner": winner,
                    "judge_reason": reason,
                    "selected_output": selected_output,
                    # Timing metadata
                    "judge_time": judge_time,
                }
            )

        # Create result DataFrame with all original columns plus new ones
        result_df = batch_df.copy()

        # Add all judge columns
        for key in results[0].keys():
            result_df[key] = [r[key] for r in results]

        return result_df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the DataFrame in batches for comparison judging."""
        results = []

        # Verify that columns exist and have valid data
        for col in [self.input_column, self.output_a_column, self.output_b_column]:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")

            # Check for missing data
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                logging.warning(f"Column '{col}' has {missing_count} missing values")

        num_batches = get_num_batches(df, self.batch_size)
        logging.info(
            f"Processing {len(df)} comparisons in {num_batches} batches of {
                self.batch_size
            }"
        )

        for i, batch_df in enumerate(batch_dataframe(df, self.batch_size)):
            logging.info(f"Processing judge batch {i + 1}/{num_batches}")

            try:
                batch_result = self._process_batch(batch_df)
                results.append(batch_result)
            except Exception as e:
                logging.error(f"Batch {i + 1} failed: {e}")
                # Create fallback results for failed batch
                batch_size = len(batch_df)
                fallback_df = batch_df.copy()
                # Total scores
                fallback_df["judge_score_a_total"] = [25] * batch_size
                fallback_df["judge_score_b_total"] = [25] * batch_size
                # Individual scores for A (distributed equally)
                fallback_df["judge_score_a_coherence"] = [5] * batch_size
                fallback_df["judge_score_a_completeness"] = [5] * batch_size
                fallback_df["judge_score_a_feasibility"] = [5] * batch_size
                fallback_df["judge_score_a_format"] = [5] * batch_size
                fallback_df["judge_score_a_granularity"] = [5] * batch_size
                # Individual scores for B (distributed equally)
                fallback_df["judge_score_b_coherence"] = [5] * batch_size
                fallback_df["judge_score_b_completeness"] = [5] * batch_size
                fallback_df["judge_score_b_feasibility"] = [5] * batch_size
                fallback_df["judge_score_b_format"] = [5] * batch_size
                fallback_df["judge_score_b_granularity"] = [5] * batch_size
                # Qualitative feedback
                fallback_df["judge_strengths_a"] = ["N/A"] * batch_size
                fallback_df["judge_weaknesses_a"] = ["Error occurred"] * batch_size
                fallback_df["judge_strengths_b"] = ["N/A"] * batch_size
                fallback_df["judge_weaknesses_b"] = ["Error occurred"] * batch_size
                # Decision
                fallback_df["judge_winner"] = ["A"] * batch_size
                fallback_df["judge_reason"] = [
                    f"Batch processing failed: {e}"
                ] * batch_size
                fallback_df["selected_output"] = batch_df[self.output_a_column].tolist()
                # Timing metadata
                fallback_df["judge_time"] = [0.0] * batch_size
                results.append(fallback_df)

        # Combine all results
        final_df = pd.concat(results, ignore_index=True)

        # Log statistics
        if len(final_df) > 0:
            winner_counts = final_df["judge_winner"].value_counts()
            avg_score_a = final_df["judge_score_a_total"].mean()
            avg_score_b = final_df["judge_score_b_total"].mean()

            logging.info(
                f"Judge results: A wins: {winner_counts.get('A', 0)}, B wins: {
                    winner_counts.get('B', 0)
                }"
            )
            logging.info(f"Average scores: A={avg_score_a:.1f}, B={avg_score_b:.1f}")

        return final_df
