"""
JSON repair utilities for fixing common LLM output formatting issues.

LLMs frequently produce malformed JSON with issues like:
- Raw newlines inside string values
- Missing commas between fields
- Trailing commas before closing braces
- Control characters in strings
- Truncated or incomplete JSON
- Typos in field names (e.g. ``granuylarity`` instead of ``granularity``)

This module provides robust repair functions to handle these cases,
plus fuzzy field-name normalisation so misspelled keys are matched
to the expected schema automatically.
"""

import json
import logging
import re
from collections import defaultdict
from threading import Lock
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Parse metrics  (thread-safe counters)
# ---------------------------------------------------------------------------

class ParseMetrics:
    """Thread-safe counters for JSON parse outcomes across the pipeline."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.success: int = 0
        self.repaired: int = 0
        self.failed: int = 0

    def record_success(self) -> None:
        with self._lock:
            self.success += 1

    def record_repaired(self) -> None:
        with self._lock:
            self.repaired += 1

    def record_failure(self) -> None:
        with self._lock:
            self.failed += 1

    @property
    def total(self) -> int:
        return self.success + self.repaired + self.failed

    def log_summary(self, logger: logging.Logger) -> None:
        total = self.total
        if total == 0:
            return
        logger.info(
            f"[ParseMetrics] total={total} "
            f"success={self.success} ({self.success/total*100:.0f}%) "
            f"repaired={self.repaired} ({self.repaired/total*100:.0f}%) "
            f"failed={self.failed} ({self.failed/total*100:.0f}%)"
        )


# Singleton so all steps share the same counters for a process run
_global_metrics = ParseMetrics()


def get_parse_metrics() -> ParseMetrics:
    return _global_metrics


# ---------------------------------------------------------------------------
# Fuzzy key normalisation  (Levenshtein-based)
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    # Optimisation: difference in length → bail early if too large
    if abs(m - n) > 3:
        return max(m, n)
    dp = list(range(n + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            dp[j], prev = min(
                prev + cost,       # substitution
                dp[j] + 1,         # insertion
                dp[j - 1] + 1,     # deletion
            ), dp[j]
    return dp[n]


def fuzzy_normalize_dict(
    d: Dict[str, Any],
    expected_keys: list[str],
    threshold: int = 2,
) -> Dict[str, Any]:
    """Normalise top-level keys of *d* that are close to an *expected_keys*
    entry via Levenshtein distance.

    Example: ``{"granuylarity": 8}`` → ``{"granularity": 8}`` when
    *expected_keys* contains ``"granularity"``.

    Unmatched keys are preserved as-is (they may be valid unknown fields).
    """
    if not expected_keys:
        return d
    result: Dict[str, Any] = {}
    for key, value in d.items():
        best = key
        best_dist = threshold
        for expected in expected_keys:
            dist = _levenshtein(key, expected)
            if dist < best_dist:
                best_dist = dist
                best = expected
        result[best] = value
    return result


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def clean_json_response(response: str) -> str:
    """
    Clean an LLM response to extract valid JSON.

    Removes markdown code fences and finds JSON object boundaries.
    Detects truncation when braces are unbalanced.

    Args:
        response: Raw response text from LLM

    Returns:
        Cleaned JSON string (or original if no JSON boundaries found)
    """
    if not response or not response.strip():
        return response or ""

    # Remove markdown if exists
    response = response.replace("```json", "").replace("```", "").strip()

    # Find JSON object boundaries
    start = response.find("{")
    end = response.rfind("}")

    if start != -1 and end != -1 and end > start:
        extracted = response[start : end + 1]
        # Detect truncation — count braces
        open_braces = extracted.count("{")
        close_braces = extracted.count("}")
        if open_braces > close_braces:
            logging.getLogger(__name__).warning(
                f"Detected truncated JSON: {open_braces} opening vs "
                f"{close_braces} closing braces"
            )
        return extracted

    return response


def repair_json(json_str: str) -> str:
    """
    Attempt to repair common JSON errors from LLM output using a character-by-character
    state machine to properly handle string boundaries.

    Repairs:
    - Escapes newlines inside string values
    - Escapes control characters inside string values
    - Adds missing commas between fields
    - Removes trailing commas before closing braces/brackets

    Args:
        json_str: Malformed JSON string to repair

    Returns:
        Repaired JSON string (may still be invalid if damage is severe)
    """
    # First, extract JSON boundaries
    json_str = clean_json_response(json_str)

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


def parse_json_with_repair(
    raw_response: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from an LLM response with automatic repair and retry.

    Tries parsing directly first, then falls back to repair.
    Tracks parse metrics (direct/repair/fail) via the global
    :class:`ParseMetrics` singleton.

    Args:
        raw_response: Raw response text from LLM
        logger: Optional logger for debug output

    Returns:
        Parsed dictionary or None if all parsing attempts fail
    """
    log = logger or logging.getLogger(__name__)
    metrics = get_parse_metrics()
    cleaned = clean_json_response(raw_response)

    # Try parsing directly first
    try:
        result = json.loads(cleaned)
        metrics.record_success()
        return result
    except json.JSONDecodeError as e:
        log.debug(f"Initial JSON parse failed: {e}")

    # Try repaired JSON
    repaired: Optional[str] = None
    try:
        repaired = repair_json(cleaned)
        log.debug(f"Repaired JSON (first 500 chars): {repaired[:500]}")
        result = json.loads(repaired)
        log.info("Successfully parsed JSON after repair")
        metrics.record_repaired()
        return result
    except json.JSONDecodeError as e:
        log.warning(f"Failed to parse repaired JSON: {e}")
        log.debug(f"Raw response (first 1000 chars): {raw_response[:1000]}")
        if repaired is not None:
            log.debug(f"Repaired (first 1000): {repaired[:1000]}")
        metrics.record_failure()
        return None
