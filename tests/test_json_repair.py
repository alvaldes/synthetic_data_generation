# tests/test_json_repair.py

import pytest
import json

from dataforge.steps.comparison_judge_step import ComparisonJudgeStep


class TestComparisonJudgeStep:
    """Test suite for ComparisonJudgeStep JSON parsing and repair."""

    @pytest.fixture
    def judge_step(self):
        """Create a ComparisonJudgeStep instance for testing without loading Ollama."""
        def dummy_template(row_data):
            return "Compare these outputs"

        step = ComparisonJudgeStep(
            name="test_judge",
            model_name="testmodel",
            input_column="input",
            output_a_column="output_a",
            output_b_column="output_b",
            prompt_template_func=dummy_template,
        )
        # Don't call load() - we're testing parsing only
        return step

    # =============================================================================
    # Test cases from actual logs - edge cases that were failing
    # =============================================================================

    def test_multiline_string_with_comma_inside(self, judge_step):
        """LLM put a comma inside the string value and split across lines."""
        raw_response = '''{
  "breakdown_a": {
    "coherence": 10,
    "completeness": 8,
    "feasibility": 10,
    "format": 10,
    "granularity": 9,
    "total_score": 47,
    "strengths": "Comprehensive coverage of backend logic, testing, and user behavior implications. Tasks are well-structured and aligned with the user story's technical requirements.",
    "weaknesses": "Lacks explicit tasks for the editor interface to set publicity levels, which is a key part of the user story."
  },
  "breakdown_b": {
    "coherence": 9,
    "completeness": 8,
    "feasibility": 9,
    "format": 10,
    "granularity": 8,
    "total_score": 45,
    "strengths": "Detailed technical steps with clear backend focus, including authentication and documentation.",
    "weaknesses": "Lacks user-facing features and explicit testing beyond basic unit tests."
  },
  "winner": "A",
  "reason": "Breakdown A provides superior completeness and coherence by addressing edge cases, user validation, and integration with existing systems."
}'''

        result = judge_step._parse_judge_response(raw_response)
        assert result["winner"] == "A"
        assert result["breakdown_a"]["total_score"] == 47
        assert result["breakdown_b"]["total_score"] == 45

    def test_missing_commas_between_fields(self, judge_step):
        """Missing commas between string fields - common LLM error."""
        raw_response = '''{
  "breakdown_a": {
    "coherence": 9
    "completeness": 8
    "feasibility": 9
    "format": 10
    "granularity": 8
    "total_score": 44
    "strengths": "Clear focus on core functionality."
    "weaknesses": "Lacks edge case handling."
  },
  "breakdown_b": {
    "coherence": 8
    "completeness": 9
    "feasibility": 9
    "format": 10
    "granularity": 9
    "total_score": 45
    "strengths": "Comprehensive coverage of edge cases."
    "weaknesses": "Too verbose in some areas."
  },
  "winner": "B",
  "reason": "Breakdown B is more comprehensive."
}'''

        result = judge_step._parse_judge_response(raw_response)
        assert result["winner"] == "B"
        assert result["breakdown_a"]["coherence"] == 9
        assert result["breakdown_b"]["completeness"] == 9

    def test_unclosed_string_at_end(self, judge_step):
        """String value that was cut off by LLM."""
        raw_response = '''{
  "breakdown_a": {
    "coherence": 9,
    "completeness": 8,
    "feasibility": 9,
    "format": 9,
    "granularity": 8,
    "total_score": 43,
    "strengths": "Good focus on core requirements.",
    "weaknesses": "Missing validation and error handling."
  },
  "breakdown_b": {
    "coherence": 10,
    "completeness": 9,
    "feasibility": 9,
    "format": 10,
    "granularity": 9,
    "total_score": 47,
    "strengths": "Comprehensive coverage of technical and UX requirements"
  },
  "winner": "B",
  "reason": "Breakdown B provides better completeness."
}'''

        result = judge_step._parse_judge_response(raw_response)
        assert result["winner"] == "B"
        assert result["breakdown_b"]["total_score"] == 47

    def test_newlines_inside_strings(self, judge_step):
        """Newlines embedded inside string values - the main bug from test3."""
        raw_response = '''{
  "breakdown_a": {
    "coherence": 9,
    "completeness": 9,
    "feasibility": 10,
    "format": 10,
    "granularity": 9,
    "total_score": 47,
    "strengths": "Tasks are tightly aligned with the user story's core requirements (viewing trainee data and filtering). Clear separation of concerns with atomic tasks.",
    "weaknesses": "Lacks explicit mention of data loading or error handling, which are common in real-world implementations."
  },
  "breakdown_b": {
    "coherence": 7,
    "completeness": 8,
    "feasibility": 9,
    "format": 9,
    "granularity": 8,
    "total_score": 41,
    "strengths": "Includes additional features like editing and saving trainee data, which may anticipate future requirements.",
    "weaknesses": "Introduces functionality (editing trainee data) not mentioned in the user story, potentially diverging from the original scope."
  },
  "winner": "A",
  "reason": "Breakdown A maintains tight alignment with the user story requirements."
}'''

        result = judge_step._parse_judge_response(raw_response)
        assert result["winner"] == "A"
        assert result["breakdown_a"]["total_score"] == 47

    def test_raw_json_from_logs(self, judge_step):
        """Actual raw response that was failing in test3 - truncated B breakdown."""
        raw_response = '''{
  "breakdown_a": {
    "coherence": 9,
    "completeness": 8,
    "feasibility": 8,
    "format": 9,
    "granularity": 7,
    "total_score": 41,
    "strengths": "Comprehensive coverage of format selection, integration with tools, and documentation. Aligns well with the user story's intent of supporting text mining tools.",
    "weaknesses": "Some tasks (e.g., 'Design Data Structure') are vague and lack specificity. Integration with text mining tools may be outside the immediate scope of the user story."
  },
  "breakdown_b"
}'''

        result = judge_step._parse_judge_response(raw_response)
        # Should get fallback since B is truncated
        assert "breakdown_a" in result
        assert "breakdown_b" in result

    def test_typo_in_granularity_key(self, judge_step):
        """LLM wrote 'granuylarity' instead of 'granularity'."""
        raw_response = '''{
  "breakdown_a": {
    "coherence": 9,
    "completeness": 8,
    "feasibility": 9,
    "format": 10,
    "granuylarity": 8,
    "total_score": 44,
    "strengths": "Clear and well-structured tasks.",
    "weaknesses": "Could be more detailed."
  },
  "breakdown_b": {
    "coherence": 8,
    "completeness": 9,
    "feasibility": 9,
    "format": 10,
    "granularity": 9,
    "total_score": 45,
    "strengths": "More detailed and comprehensive.",
    "weaknesses": "Slightly too verbose."
  },
  "winner": "B",
  "reason": "B is more complete."
}'''

        result = judge_step._parse_judge_response(raw_response)
        # Should fallback to 0 for the typo'd key, compute total from correct keys
        assert "breakdown_a" in result
        assert "breakdown_b" in result

    def test_perfect_json_still_works(self, judge_step):
        """Valid JSON should parse correctly without repair."""
        raw_response = '''{
  "breakdown_a": {
    "coherence": 9,
    "completeness": 8,
    "feasibility": 9,
    "format": 10,
    "granularity": 8,
    "total_score": 44,
    "strengths": "Clear and well-structured tasks.",
    "weaknesses": "Could be more detailed."
  },
  "breakdown_b": {
    "coherence": 8,
    "completeness": 9,
    "feasibility": 9,
    "format": 10,
    "granularity": 9,
    "total_score": 45,
    "strengths": "More detailed and comprehensive.",
    "weaknesses": "Slightly too verbose."
  },
  "winner": "B",
  "reason": "B is more complete."
}'''

        result = judge_step._parse_judge_response(raw_response)
        assert result["winner"] == "B"
        assert result["breakdown_a"]["total_score"] == 44
        assert result["breakdown_b"]["total_score"] == 45
        assert result["breakdown_a"]["granularity"] == 8
        assert result["breakdown_b"]["granularity"] == 9

    def test_json_with_trailing_content(self, judge_step):
        """JSON followed by extra text explanation from LLM."""
        raw_response = '''{
  "breakdown_a": {
    "coherence": 9,
    "completeness": 8,
    "feasibility": 9,
    "format": 10,
    "granularity": 8,
    "total_score": 44,
    "strengths": "Well-structured tasks.",
    "weaknesses": "Missing some details."
  },
  "breakdown_b": {
    "coherence": 8,
    "completeness": 9,
    "feasibility": 9,
    "format": 10,
    "granularity": 9,
    "total_score": 45,
    "strengths": "Very comprehensive.",
    "weaknesses": "A bit verbose."
  },
  "winner": "B",
  "reason": "B covers more requirements."
}
Based on my analysis, Breakdown B is superior because it provides better completeness while maintaining good coherence and feasibility.'''

        result = judge_step._parse_judge_response(raw_response)
        assert result["winner"] == "B"
        # Should extract only the JSON part

    def test_empty_response_fallback(self, judge_step):
        """Empty or minimal response should return fallback."""
        raw_response = ""

        result = judge_step._parse_judge_response(raw_response)
        assert result["winner"] == "A"  # Default fallback
        assert result["breakdown_a"]["total_score"] == 25
        assert result["breakdown_b"]["total_score"] == 25

    def test_invalid_winner_fallback(self, judge_step):
        """Winner field with invalid value should default to A."""
        raw_response = '''{
  "breakdown_a": {
    "coherence": 9,
    "completeness": 8,
    "feasibility": 9,
    "format": 10,
    "granularity": 8,
    "total_score": 44
  },
  "breakdown_b": {
    "coherence": 8,
    "completeness": 9,
    "feasibility": 9,
    "format": 10,
    "granularity": 9,
    "total_score": 45
  },
  "winner": "INVALID",
  "reason": "Both breakdowns are similar."
}'''

        result = judge_step._parse_judge_response(raw_response)
        # Should default to A since winner is invalid
        assert result["winner"] == "A"


# =============================================================================
# Direct unit tests for _repair_json method
# =============================================================================

class TestRepairJson:
    """Test the _repair_json method directly."""

    @pytest.fixture
    def judge_step(self):
        def dummy_template(row_data):
            return "Compare these outputs"

        step = ComparisonJudgeStep(
            name="test_judge",
            model_name="testmodel",
            input_column="input",
            output_a_column="output_a",
            output_b_column="output_b",
            prompt_template_func=dummy_template,
        )
        return step

    def test_escape_newlines_in_strings(self, judge_step):
        """Raw newlines inside string values should be escaped."""
        raw = '{"strengths": "Hello\nWorld"}'
        repaired = judge_step._repair_json(raw)
        # Should now have escaped newline
        assert '\\n' in repaired or '\\n' in repaired

        # Should be valid JSON
        parsed = json.loads(repaired)
        assert parsed["strengths"] == "Hello\nWorld"

    def test_missing_commas_after_numbers(self, judge_step):
        """Missing commas after number values before next field."""
        raw = '{"score": 42 "next": "value"}'
        repaired = judge_step._repair_json(raw)
        parsed = json.loads(repaired)
        assert parsed["score"] == 42
        assert parsed["next"] == "value"

    def test_trailing_comma_before_closing_brace(self, judge_step):
        """Trailing comma before } should be removed."""
        raw = '{"a": 1, "b": 2,}'
        repaired = judge_step._repair_json(raw)
        parsed = json.loads(repaired)
        assert parsed == {"a": 1, "b": 2}

    def test_no_escaping_outside_strings(self, judge_step):
        """Normal JSON structure should not be affected."""
        raw = '{"coherence": 9, "completeness": 8}'
        repaired = judge_step._repair_json(raw)
        parsed = json.loads(repaired)
        assert parsed["coherence"] == 9
        assert parsed["completeness"] == 8

    def test_control_characters_removed(self, judge_step):
        """Null bytes and invalid control chars should be removed."""
        raw = '{"text": "Hello\x00World"}'
        repaired = judge_step._repair_json(raw)
        # Should be valid JSON without null char
        parsed = json.loads(repaired)
        assert "text" in parsed


# =============================================================================
# Run tests with: pytest tests/test_json_repair.py -v
# =============================================================================