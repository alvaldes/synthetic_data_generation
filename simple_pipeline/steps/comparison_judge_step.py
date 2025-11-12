import json
import logging
import pandas as pd
from typing import Dict, List, Optional, Any
import ollama

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
        **kwargs
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
            "judge_score_a",
            "judge_score_b",
            "judge_winner",
            "judge_reason",
            "selected_output"
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
        start = response.find('{')
        end = response.rfind('}')

        if start != -1 and end != -1 and end > start:
            return response[start:end+1]

        return response

    def _parse_judge_response(self, raw_response: str) -> Dict:
        """Parse the judge response JSON with robust error handling."""
        try:
            cleaned_response = self._clean_json_response(raw_response)
            result = json.loads(cleaned_response)

            # Validate required structure
            required_top_level = ['breakdown_a', 'breakdown_b', 'winner', 'reason']
            for field in required_top_level:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            # Validate breakdown structure
            for breakdown_key in ['breakdown_a', 'breakdown_b']:
                breakdown = result[breakdown_key]
                if 'total_score' not in breakdown:
                    # Calculate total if missing
                    score_fields = ['completeness', 'clarity', 'actionability', 'logical_structure', 'granularity']
                    total = sum(breakdown.get(field, 0) for field in score_fields)
                    breakdown['total_score'] = total

            # Ensure winner is valid
            if result['winner'] not in ['A', 'B']:
                # Default to A if invalid
                result['winner'] = 'A'
                result['reason'] = "Invalid winner designation, defaulting to A"

            return result

        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Failed to parse judge response: {e}")
            logging.warning(f"Raw response: {raw_response}")

            # Return fallback structure
            return {
                "breakdown_a": {"total_score": 25, "strengths": "Unable to parse", "weaknesses": "Parse error"},
                "breakdown_b": {"total_score": 25, "strengths": "Unable to parse", "weaknesses": "Parse error"},
                "winner": "A",
                "reason": f"Failed to parse judge response: {e}"
            }

    def _judge_comparison(self, input_text: str, output_a: str, output_b: str) -> Dict:
        """Use LLM to judge which output is better."""
        # Create row-like dict for prompt template
        row_data = {
            self.input_column: input_text,
            self.output_a_column: output_a,
            self.output_b_column: output_b
        }

        prompt = self.prompt_template_func(row_data)

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                options=self.generation_kwargs
            )

            raw_response = response['message']['content']
            return self._parse_judge_response(raw_response)

        except Exception as e:
            logging.error(f"Judge evaluation failed: {e}")
            # Return fallback
            return {
                "breakdown_a": {"total_score": 25, "strengths": "Error occurred", "weaknesses": str(e)},
                "breakdown_b": {"total_score": 25, "strengths": "Error occurred", "weaknesses": str(e)},
                "winner": "A",
                "reason": f"Judge evaluation failed: {e}"
            }

    def _process_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of rows for comparison judging."""
        results = []

        for _, row in batch_df.iterrows():
            input_text = str(row[self.input_column])
            output_a = str(row[self.output_a_column])
            output_b = str(row[self.output_b_column])

            # Get judge evaluation
            judgment = self._judge_comparison(input_text, output_a, output_b)

            # Extract scores and decision
            score_a = judgment['breakdown_a']['total_score']
            score_b = judgment['breakdown_b']['total_score']
            winner = judgment['winner']
            reason = judgment['reason']

            # Select the winning output
            if winner == 'B':
                selected_output = output_b
            else:  # Default to A
                selected_output = output_a

            results.append({
                "judge_score_a": score_a,
                "judge_score_b": score_b,
                "judge_winner": winner,
                "judge_reason": reason,
                "selected_output": selected_output
            })

        # Create result DataFrame with all original columns plus new ones
        result_df = batch_df.copy()

        # Add judge columns
        result_df["judge_score_a"] = [r['judge_score_a'] for r in results]
        result_df["judge_score_b"] = [r['judge_score_b'] for r in results]
        result_df["judge_winner"] = [r['judge_winner'] for r in results]
        result_df["judge_reason"] = [r['judge_reason'] for r in results]
        result_df["selected_output"] = [r['selected_output'] for r in results]

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
        logging.info(f"Processing {len(df)} comparisons in {num_batches} batches of {self.batch_size}")

        for i, batch_df in enumerate(batch_dataframe(df, self.batch_size)):
            logging.info(f"Processing judge batch {i+1}/{num_batches}")

            try:
                batch_result = self._process_batch(batch_df)
                results.append(batch_result)
            except Exception as e:
                logging.error(f"Batch {i+1} failed: {e}")
                # Create fallback results for failed batch
                batch_size = len(batch_df)
                fallback_df = batch_df.copy()
                fallback_df["judge_score_a"] = [25] * batch_size
                fallback_df["judge_score_b"] = [25] * batch_size
                fallback_df["judge_winner"] = ["A"] * batch_size
                fallback_df["judge_reason"] = [f"Batch processing failed: {e}"] * batch_size
                fallback_df["selected_output"] = batch_df[self.output_a_column].tolist()
                results.append(fallback_df)

        # Combine all results
        final_df = pd.concat(results, ignore_index=True)

        # Log statistics
        if len(final_df) > 0:
            winner_counts = final_df["judge_winner"].value_counts()
            avg_score_a = final_df["judge_score_a"].mean()
            avg_score_b = final_df["judge_score_b"].mean()

            logging.info(f"Judge results: A wins: {winner_counts.get('A', 0)}, B wins: {winner_counts.get('B', 0)}")
            logging.info(f"Average scores: A={avg_score_a:.1f}, B={avg_score_b:.1f}")

        return final_df