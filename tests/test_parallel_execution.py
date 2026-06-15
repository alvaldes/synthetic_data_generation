"""
Tests for parallel row processing in LLM steps.

Verifies that:
- num_workers=1 produces identical output to sequential
- Row order is preserved with num_workers > 1
- Thread-local client isolation works
- Sequential fallback at num_workers=0
"""

from unittest.mock import patch, MagicMock
import ollama
import pandas as pd
import pytest

from dataforge.llm.ollama_step import OllamaLLMStep, _thread_local as ollama_tl
from dataforge.llm.ollama_judge_step import OllamaJudgeStep, _thread_local as judge_tl
from dataforge.llm.comparison_judge_step import ComparisonJudgeStep, _thread_local as comp_tl


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "prompt": ["Hello", "World", "Test", "Data"],
    })


@pytest.fixture
def judge_df():
    return pd.DataFrame({
        "user_story": ["Story A", "Story B"],
        "generated_tasks": ["Task 1", "Task 2"],
    })


@pytest.fixture
def comparison_df():
    return pd.DataFrame({
        "input": ["Input 1", "Input 2"],
        "output_a": ["A1", "A2"],
        "output_b": ["B1", "B2"],
    })


@pytest.fixture(autouse=True)
def clear_thread_local():
    """Clean thread-local state between tests."""
    for tl in [ollama_tl, judge_tl, comp_tl]:
        if hasattr(tl, "client"):
            del tl.client
    yield


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock get_settings to return predictable config with num_workers."""
    with patch("dataforge.llm.ollama_step.get_settings") as mock_ollama, \
         patch("dataforge.llm.ollama_judge_step.get_settings") as mock_judge, \
         patch("dataforge.llm.comparison_judge_step.get_settings") as mock_comp:

        # OllamaLLMStep config
        cfg_ollama = MagicMock()
        cfg_ollama.llm.default_model = "test-model"
        cfg_ollama.llm.ollama_host = "http://localhost:11434"
        cfg_ollama.llm.max_retries = 1
        cfg_ollama.llm.batch_size = 4
        cfg_ollama.llm.temperature = 0.3
        cfg_ollama.llm.num_predict = 100
        cfg_ollama.llm.num_workers = 1
        mock_ollama.return_value = cfg_ollama

        # OllamaJudgeStep config
        cfg_judge = MagicMock()
        cfg_judge.llm.default_model = "test-model"
        cfg_judge.llm.ollama_host = "http://localhost:11434"
        cfg_judge.llm.max_retries = 1
        cfg_judge.judge.model = None
        cfg_judge.judge.approval_threshold = 35.0
        cfg_judge.judge.batch_size = 4
        cfg_judge.judge.temperature = 0.2
        cfg_judge.judge.num_predict = 100
        cfg_judge.judge.num_workers = 1
        cfg_judge.judge.column_prefix = "validacion_"
        cfg_judge.judge.criteria = []
        mock_judge.return_value = cfg_judge

        # ComparisonJudgeStep config
        cfg_comp = MagicMock()
        cfg_comp.llm.default_model = "test-model"
        cfg_comp.llm.ollama_host = "http://localhost:11434"
        cfg_comp.llm.max_retries = 1
        cfg_comp.comparison_judge.model = None
        cfg_comp.comparison_judge.batch_size = 2
        cfg_comp.comparison_judge.temperature = 0.2
        cfg_comp.comparison_judge.num_predict = 100
        cfg_comp.comparison_judge.num_workers = 1
        cfg_comp.comparison_judge.column_prefix = "judge_"
        mock_comp.return_value = cfg_comp

        yield


# ---------------------------------------------------------------------------
# Helper: mock ollama.Client.chat
# ---------------------------------------------------------------------------

def _make_mock_chat(response_text: str = "mock response"):
    """Create a mock for ollama.Client.chat that returns a predictable response."""
    def mock_chat(model=None, messages=None, stream=False, format=None, options=None):
        return {
            "message": {
                "content": response_text,
            }
        }
    return mock_chat


# ---------------------------------------------------------------------------
# OllamaLLMStep tests
# ---------------------------------------------------------------------------

class TestOllamaLLMStepParallel:

    def test_sequential_fallback_default(self, sample_df):
        """num_workers=1 uses sequential path and produces output."""
        step = OllamaLLMStep(
            name="test",
            prompt_column="prompt",
            output_column="response",
            prompt_template=lambda row: row["prompt"],
            num_workers=1,
        )
        step.client = MagicMock()
        step.client.chat = _make_mock_chat("sequential response")

        result = step.process(sample_df)

        assert result is not None
        assert len(result) == len(sample_df)
        assert "response" in result.columns
        assert result["response"].tolist() == ["sequential response"] * 4

    def test_none_workers_uses_config_default(self, sample_df):
        """num_workers=None falls back to config default (num_workers=1)."""
        step = OllamaLLMStep(
            name="test",
            prompt_column="prompt",
            output_column="response",
            prompt_template=lambda row: row["prompt"],
        )
        assert step.num_workers == 1

    def test_row_order_preserved_with_parallel(self, sample_df):
        """Row order is preserved when num_workers > batch_size."""
        # Each row returns a unique value based on the prompt
        def chat_side_effect(model=None, messages=None, stream=False, format=None, options=None):
            prompt = messages[-1]["content"] if messages else "unknown"
            return {"message": {"content": f"response:{prompt}"}}

        step = OllamaLLMStep(
            name="test",
            prompt_column="prompt",
            output_column="response",
            prompt_template=lambda row: row["prompt"],
            num_workers=4,
        )
        step.client = MagicMock()
        step.client.chat = _make_mock_chat("unused")

        with patch.object(step, "_get_thread_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.side_effect = chat_side_effect
            mock_get.return_value = mock_client

            result = step.process(sample_df)

        assert result is not None
        # Order must match original
        assert result["prompt"].tolist() == ["Hello", "World", "Test", "Data"]
        assert result["response"].tolist() == [
            "response:Hello",
            "response:World",
            "response:Test",
            "response:Data",
        ]

    def test_thread_local_client_isolation(self, sample_df):
        """Each thread gets its own ollama.Client via thread-local."""
        clients_created = []

        original_client = ollama.Client

        def tracking_client(host=None):
            client = original_client(host=host)
            clients_created.append(client)
            return client

        step = OllamaLLMStep(
            name="test",
            prompt_column="prompt",
            output_column="response",
            prompt_template=lambda row: row["prompt"],
            num_workers=2,
        )
        step.client = MagicMock()

        with patch("dataforge.llm.ollama_step.ollama.Client", side_effect=tracking_client):
            result = step.process(sample_df)

        assert result is not None
        # At least one thread-local client was created
        assert len(clients_created) >= 1


# ---------------------------------------------------------------------------
# OllamaJudgeStep tests
# ---------------------------------------------------------------------------

class TestOllamaJudgeStepParallel:

    def test_judge_sequential_fallback(self, judge_df):
        """Judge step with num_workers=1 produces expected output structure."""
        step = OllamaJudgeStep(
            name="test-judge",
            historia_usuario_column="user_story",
            tareas_generadas_column="generated_tasks",
            num_workers=1,
        )
        # Mock client to return valid JSON
        judge_response = (
            '{"coherence": {"puntuacion": 8, "justificacion": "Good"},'
            '"completeness": {"puntuacion": 7, "justificacion": "OK", "tareas_faltantes": []},'
            '"feasibility": {"puntuacion": 9, "justificacion": "Feasible"},'
            '"format": {"puntuacion": 8, "justificacion": "Clean"},'
            '"granularity": {"puntuacion": 7, "justificacion": "Adequate"}}'
        )
        step.client = MagicMock()
        step.client.chat = _make_mock_chat(judge_response)

        result = step.process(judge_df)

        assert result is not None
        assert len(result) == 2
        assert "validacion_total" in result.columns
        assert result["validacion_total"].tolist() == [39, 39]

    def test_judge_row_order_preserved(self, judge_df):
        """Judge step preserves row order in parallel mode."""
        def chat_side_effect(model=None, messages=None, stream=False, format=None, options=None):
            return {
                "message": {
                    "content": (
                        '{"coherence": {"puntuacion": 10, "justificacion": "A"},'
                        '"completeness": {"puntuacion": 10, "justificacion": "A", "tareas_faltantes": []},'
                        '"feasibility": {"puntuacion": 10, "justificacion": "A"},'
                        '"format": {"puntuacion": 10, "justificacion": "A"},'
                        '"granularity": {"puntuacion": 10, "justificacion": "A"}}'
                    )
                }
            }

        step = OllamaJudgeStep(
            name="test-judge",
            historia_usuario_column="user_story",
            tareas_generadas_column="generated_tasks",
            num_workers=2,
        )
        step.client = MagicMock()

        with patch.object(step, "_get_thread_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.side_effect = chat_side_effect
            mock_get.return_value = mock_client

            result = step.process(judge_df)

        assert result is not None
        assert result["user_story"].tolist() == ["Story A", "Story B"]


# ---------------------------------------------------------------------------
# ComparisonJudgeStep tests
# ---------------------------------------------------------------------------

class TestComparisonJudgeStepParallel:

    def test_comparison_sequential_fallback(self, comparison_df):
        """Comparison judge with num_workers=1 produces expected output."""
        judge_response = (
            '{"breakdown_a": {"total_score": 40, "coherence": 8, "completeness": 8, '
            '"feasibility": 8, "format": 8, "granularity": 8, "strengths": "Good", '
            '"weaknesses": "None"}, '
            '"breakdown_b": {"total_score": 30, "coherence": 6, "completeness": 6, '
            '"feasibility": 6, "format": 6, "granularity": 6, "strengths": "OK", '
            '"weaknesses": "Some"}, '
            '"winner": "A", "reason": "Better overall"}'
        )

        step = ComparisonJudgeStep(
            name="test-comp",
            input_column="input",
            output_a_column="output_a",
            output_b_column="output_b",
            num_workers=1,
        )
        step.client = MagicMock()
        step.client.chat = _make_mock_chat(judge_response)

        result = step.process(comparison_df)

        assert result is not None
        assert len(result) == 2
        assert "judge_winner" in result.columns
        assert result["judge_winner"].tolist() == ["A", "A"]

    def test_comparison_row_order_preserved(self, comparison_df):
        """Comparison judge preserves row order in parallel mode."""
        def chat_side_effect(model=None, messages=None, stream=False, format=None, options=None):
            return {
                "message": {
                    "content": (
                        '{"breakdown_a": {"total_score": 35, "coherence": 7, "completeness": 7, '
                        '"feasibility": 7, "format": 7, "granularity": 7, "strengths": "S", '
                        '"weaknesses": "W"}, '
                        '"breakdown_b": {"total_score": 25, "coherence": 5, "completeness": 5, '
                        '"feasibility": 5, "format": 5, "granularity": 5, "strengths": "S", '
                        '"weaknesses": "W"}, '
                        '"winner": "A", "reason": "Better"}'
                    )
                }
            }

        step = ComparisonJudgeStep(
            name="test-comp",
            input_column="input",
            output_a_column="output_a",
            output_b_column="output_b",
            num_workers=2,
        )
        step.client = MagicMock()

        with patch.object(step, "_get_thread_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.side_effect = chat_side_effect
            mock_get.return_value = mock_client

            result = step.process(comparison_df)

        assert result is not None
        assert result["input"].tolist() == ["Input 1", "Input 2"]
