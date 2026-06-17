# tests/test_judge_validation.py

import json
import pytest

from dataforge.llm.ollama_judge_step import OllamaJudgeStep
from dataforge.llm.comparison_judge_step import ComparisonJudgeStep


# ---------------------------------------------------------------------------
# OllamaJudgeStep — _parse_validation_result
# ---------------------------------------------------------------------------

@pytest.fixture
def ollama_judge():
    step = OllamaJudgeStep(
        name="test-judge",
        historia_usuario_column="story",
        tareas_generadas_column="tasks",
    )
    step.logger.disabled = True
    return step


def test_parse_valid_json_returns_all_fields(ollama_judge):
    raw = json.dumps({
        "coherence": {"puntuacion": 8, "justificacion": "Good"},
        "completeness": {"puntuacion": 7, "justificacion": "OK", "tareas_faltantes": []},
        "feasibility": {"puntuacion": 9, "justificacion": "Feasible"},
        "format": {"puntuacion": 6, "justificacion": "Decent"},
        "granularity": {"puntuacion": 8, "justificacion": "Adequate"},
        "puntuacion_total": 50,
        "aprobado": True,
        "problemas_criticos": [],
        "recomendaciones": ["Good job"],
    })
    result = ollama_judge._parse_validation_result(raw)

    assert result["coherence"]["puntuacion"] == 8
    assert result["completeness"]["puntuacion"] == 7
    assert result["feasibility"]["puntuacion"] == 9
    assert result["format"]["puntuacion"] == 6
    assert result["granularity"]["puntuacion"] == 8


def test_parse_overrides_llm_total_with_code_total(ollama_judge):
    raw = json.dumps({
        "coherence": {"puntuacion": 8, "justificacion": ""},
        "completeness": {"puntuacion": 7, "justificacion": ""},
        "feasibility": {"puntuacion": 9, "justificacion": ""},
        "format": {"puntuacion": 6, "justificacion": ""},
        "granularity": {"puntuacion": 8, "justificacion": ""},
        "puntuacion_total": 99,
    })
    result = ollama_judge._parse_validation_result(raw)

    assert result["puntuacion_total"] == 38  # 8+7+9+6+8
    assert result["puntuacion_total"] != 99


def test_parse_calculates_total_when_llm_omits_it(ollama_judge):
    raw = json.dumps({
        "coherence": {"puntuacion": 5, "justificacion": ""},
        "completeness": {"puntuacion": 5, "justificacion": ""},
        "feasibility": {"puntuacion": 5, "justificacion": ""},
        "format": {"puntuacion": 5, "justificacion": ""},
        "granularity": {"puntuacion": 5, "justificacion": ""},
    })
    result = ollama_judge._parse_validation_result(raw)

    assert result["puntuacion_total"] == 25


def test_parse_determines_approval_when_omitted(ollama_judge):
    raw = json.dumps({
        "coherence": {"puntuacion": 10, "justificacion": ""},
        "completeness": {"puntuacion": 10, "justificacion": ""},
        "feasibility": {"puntuacion": 10, "justificacion": ""},
        "format": {"puntuacion": 10, "justificacion": ""},
        "granularity": {"puntuacion": 10, "justificacion": ""},
    })
    result = ollama_judge._parse_validation_result(raw)
    assert result["aprobado"] is True


def test_parse_approval_threshold_respected(ollama_judge):
    raw = json.dumps({
        "coherence": {"puntuacion": 1, "justificacion": ""},
        "completeness": {"puntuacion": 1, "justificacion": ""},
        "feasibility": {"puntuacion": 1, "justificacion": ""},
        "format": {"puntuacion": 1, "justificacion": ""},
        "granularity": {"puntuacion": 1, "justificacion": ""},
    })
    result = ollama_judge._parse_validation_result(raw)
    assert result["puntuacion_total"] == 5
    assert result["aprobado"] is False


def test_parse_invalid_json_returns_fallback_with_parse_error(ollama_judge):
    result = ollama_judge._parse_validation_result("not json at all")

    assert result["parse_error"] is True
    assert result["coherence"]["puntuacion"] == -1
    assert result["puntuacion_total"] == -5
    assert result["aprobado"] is False
    assert "Error en parsing" in result["problemas_criticos"][0]


def test_parse_missing_field_returns_fallback(ollama_judge):
    raw = json.dumps({
        "coherence": {"puntuacion": 8, "justificacion": ""},
    })
    result = ollama_judge._parse_validation_result(raw)

    assert result["parse_error"] is True
    assert result["puntuacion_total"] == -5


def test_parse_empty_puntuacion_returns_fallback(ollama_judge):
    raw = json.dumps({
        "coherence": {"puntuacion": 8, "justificacion": ""},
        "completeness": {"puntuacion": 7, "justificacion": ""},
        "feasibility": {"puntuacion": 9, "justificacion": ""},
        "format": {"puntuacion": 6, "justificacion": ""},
        "granularity": {"justificacion": "missing puntuacion"},
    })
    result = ollama_judge._parse_validation_result(raw)
    assert result["parse_error"] is True


def test_parse_sets_defaults_for_optional_fields(ollama_judge):
    raw = json.dumps({
        "coherence": {"puntuacion": 8, "justificacion": ""},
        "completeness": {"puntuacion": 7, "justificacion": ""},
        "feasibility": {"puntuacion": 9, "justificacion": ""},
        "format": {"puntuacion": 6, "justificacion": ""},
        "granularity": {"puntuacion": 8, "justificacion": ""},
    })
    result = ollama_judge._parse_validation_result(raw)
    assert result["problemas_criticos"] == []
    assert result["recomendaciones"] == []


# ---------------------------------------------------------------------------
# OllamaJudgeStep — _has_zero_score
# ---------------------------------------------------------------------------

def test_has_zero_score_returns_false_when_all_positive(ollama_judge):
    result = {
        "coherence": {"puntuacion": 8},
        "completeness": {"puntuacion": 7},
        "feasibility": {"puntuacion": 9},
        "format": {"puntuacion": 6},
        "granularity": {"puntuacion": 8},
    }
    assert ollama_judge._has_zero_score(result) is False


def test_has_zero_score_detects_zero(ollama_judge):
    result = {
        "coherence": {"puntuacion": 8},
        "completeness": {"puntuacion": 0},
        "feasibility": {"puntuacion": 9},
        "format": {"puntuacion": 6},
        "granularity": {"puntuacion": 8},
    }
    assert ollama_judge._has_zero_score(result) is True


def test_has_zero_score_ignores_sentinel_minus_one(ollama_judge):
    result = {
        "coherence": {"puntuacion": -1},
        "completeness": {"puntuacion": -1},
        "feasibility": {"puntuacion": -1},
        "format": {"puntuacion": -1},
        "granularity": {"puntuacion": -1},
    }
    assert ollama_judge._has_zero_score(result) is False


def test_has_zero_score_multiple_zeros(ollama_judge):
    result = {
        "coherence": {"puntuacion": 0},
        "completeness": {"puntuacion": 7},
        "feasibility": {"puntuacion": 0},
        "format": {"puntuacion": 6},
        "granularity": {"puntuacion": 0},
    }
    assert ollama_judge._has_zero_score(result) is True


# ---------------------------------------------------------------------------
# ComparisonJudgeStep — _validate_and_normalize_judge_result
# ---------------------------------------------------------------------------

@pytest.fixture
def comparison_judge():
    return ComparisonJudgeStep(
        name="test-compare",
        input_column="input",
        output_a_column="output_a",
        output_b_column="output_b",
    )


def test_normalize_valid_result_recalculates_total(comparison_judge):
    result = {
        "breakdown_a": {
            "coherence": 8, "completeness": 7, "feasibility": 9,
            "format": 6, "granularity": 8,
            "total_score": 99,
            "strengths": "a", "weaknesses": "b",
        },
        "breakdown_b": {
            "coherence": 5, "completeness": 5, "feasibility": 5,
            "format": 5, "granularity": 5,
            "total_score": 99,
            "strengths": "c", "weaknesses": "d",
        },
        "winner": "A",
        "reason": "A is better",
    }
    normalized = comparison_judge._validate_and_normalize_judge_result(result)

    assert normalized["breakdown_a"]["total_score"] == 38  # 8+7+9+6+8
    assert normalized["breakdown_a"]["total_score"] != 99
    assert normalized["breakdown_b"]["total_score"] == 25  # 5*5
    assert normalized["breakdown_b"]["total_score"] != 99


def test_normalize_calculates_total_when_omitted(comparison_judge):
    result = {
        "breakdown_a": {
            "coherence": 3, "completeness": 4, "feasibility": 5,
            "format": 6, "granularity": 7,
            "strengths": "", "weaknesses": "",
        },
        "breakdown_b": {
            "coherence": 1, "completeness": 2, "feasibility": 3,
            "format": 4, "granularity": 5,
            "strengths": "", "weaknesses": "",
        },
        "winner": "B",
        "reason": "B is more detailed",
    }
    normalized = comparison_judge._validate_and_normalize_judge_result(result)

    assert normalized["breakdown_a"]["total_score"] == 25
    assert normalized["breakdown_b"]["total_score"] == 15


def test_normalize_fixes_invalid_winner(comparison_judge):
    result = {
        "breakdown_a": {
            "coherence": 5, "completeness": 5, "feasibility": 5,
            "format": 5, "granularity": 5, "total_score": 25,
            "strengths": "", "weaknesses": "",
        },
        "breakdown_b": {
            "coherence": 5, "completeness": 5, "feasibility": 5,
            "format": 5, "granularity": 5, "total_score": 25,
            "strengths": "", "weaknesses": "",
        },
        "winner": "X",
        "reason": "Test",
    }
    normalized = comparison_judge._validate_and_normalize_judge_result(result)

    assert normalized["winner"] == "A"
    assert "Invalid winner" in normalized["reason"]


def test_normalize_missing_score_field_raises(comparison_judge):
    result = {
        "breakdown_a": {
            "coherence": 5, "completeness": 5, "feasibility": 5,
            "format": 5,
        },
        "breakdown_b": {
            "coherence": 5, "completeness": 5, "feasibility": 5,
            "format": 5, "granularity": 5, "total_score": 25,
            "strengths": "", "weaknesses": "",
        },
        "winner": "A",
        "reason": "Test",
    }
    with pytest.raises(ValueError, match="granularity"):
        comparison_judge._validate_and_normalize_judge_result(result)


def test_normalize_missing_breakdown_raises(comparison_judge):
    result = {
        "breakdown_a": {},
        "winner": "A",
        "reason": "Test",
    }
    with pytest.raises(ValueError, match="breakdown_b"):
        comparison_judge._validate_and_normalize_judge_result(result)


# ---------------------------------------------------------------------------
# ComparisonJudgeStep — _has_zero_score
# ---------------------------------------------------------------------------

def test_comparison_has_zero_score_false(comparison_judge):
    result = {
        "breakdown_a": {
            "coherence": 8, "completeness": 7, "feasibility": 9,
            "format": 6, "granularity": 8,
        },
        "breakdown_b": {
            "coherence": 5, "completeness": 5, "feasibility": 5,
            "format": 5, "granularity": 5,
        },
    }
    assert comparison_judge._has_zero_score(result) is False


def test_comparison_has_zero_score_in_a(comparison_judge):
    result = {
        "breakdown_a": {
            "coherence": 0, "completeness": 7, "feasibility": 9,
            "format": 6, "granularity": 8,
        },
        "breakdown_b": {
            "coherence": 5, "completeness": 5, "feasibility": 5,
            "format": 5, "granularity": 5,
        },
    }
    assert comparison_judge._has_zero_score(result) is True


def test_comparison_has_zero_score_in_b(comparison_judge):
    result = {
        "breakdown_a": {
            "coherence": 8, "completeness": 7, "feasibility": 9,
            "format": 6, "granularity": 8,
        },
        "breakdown_b": {
            "coherence": 5, "completeness": 5, "feasibility": 0,
            "format": 5, "granularity": 5,
        },
    }
    assert comparison_judge._has_zero_score(result) is True


def test_comparison_has_zero_score_ignores_sentinel(comparison_judge):
    result = {
        "breakdown_a": {
            "coherence": -1, "completeness": -1, "feasibility": -1,
            "format": -1, "granularity": -1,
        },
        "breakdown_b": {
            "coherence": -1, "completeness": -1, "feasibility": -1,
            "format": -1, "granularity": -1,
        },
    }
    assert comparison_judge._has_zero_score(result) is False


def test_comparison_has_zero_score_missing_breakdown(comparison_judge):
    result = {"winner": "A", "reason": "Test"}
    assert comparison_judge._has_zero_score(result) is False


# ---------------------------------------------------------------------------
# ComparisonJudgeStep — _parse_judge_response fallback
# ---------------------------------------------------------------------------

def test_parse_judge_response_invalid_json(comparison_judge):
    result = comparison_judge._parse_judge_response("not json")

    assert result["parse_error"] is True
    assert result["breakdown_a"]["total_score"] == -5
    assert result["breakdown_a"]["coherence"] == -1
    assert result["breakdown_b"]["granularity"] == -1
    assert result["winner"] == "A"


def test_parse_judge_response_validation_failure(comparison_judge):
    raw = json.dumps({"breakdown_a": {}, "winner": "A"})
    result = comparison_judge._parse_judge_response(raw)

    assert result["parse_error"] is True
    assert result["winner"] == "A"


# ---------------------------------------------------------------------------
# OllamaJudgeStep — _create_judge_prompt zero retry
# ---------------------------------------------------------------------------

def test_create_judge_prompt_no_retry(ollama_judge):
    prompt = ollama_judge._create_judge_prompt("story", "tasks", zero_retry_count=0)
    assert "REINTENTO" not in prompt
    assert "CRÍTICO" not in prompt


def test_create_judge_prompt_with_retry(ollama_judge):
    prompt = ollama_judge._create_judge_prompt("story", "tasks", zero_retry_count=1)
    assert "REINTENTO 1" in prompt
    assert "CRÍTICO" in prompt
    assert "0 NO está permitido" in prompt


def test_create_judge_prompt_retry_count_appears(ollama_judge):
    prompt = ollama_judge._create_judge_prompt("story", "tasks", zero_retry_count=2)
    assert "REINTENTO 2" in prompt
