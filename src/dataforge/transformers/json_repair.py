"""
JSON repair utilities for fixing common LLM output formatting issues.

LLMs frequently produce malformed JSON with issues like:
- Raw newlines inside string values
- Missing commas between fields
- Trailing commas before closing braces
- Control characters in strings
- Truncated or incomplete JSON

This module provides robust repair functions to handle these cases.
"""

import json
import logging
import re
from typing import Any, Dict, Optional


def clean_json_response(response: str) -> str:
    """
    Clean an LLM response to extract valid JSON.

    Removes markdown code fences and finds JSON object boundaries.

    Args:
        response: Raw response text from LLM

    Returns:
        Cleaned JSON string (or original if no JSON boundaries found)
    """
    # Remove markdown if exists
    response = response.replace("```json", "").replace("```", "").strip()

    # Find JSON object boundaries
    start = response.find("{")
    end = response.rfind("}")

    if start != -1 and end != -1 and end > start:
        return response[start : end + 1]

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

    Args:
        raw_response: Raw response text from LLM
        logger: Optional logger for debug output

    Returns:
        Parsed dictionary or None if all parsing attempts fail
    """
    log = logger or logging.getLogger(__name__)
    cleaned = clean_json_response(raw_response)

    # Try parsing directly first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.debug(f"Initial JSON parse failed: {e}")

    # Try repaired JSON
    try:
        repaired = repair_json(cleaned)
        log.debug(f"Repaired JSON (first 500 chars): {repaired[:500]}")
        result = json.loads(repaired)
        log.info("Successfully parsed JSON after repair")
        return result
    except json.JSONDecodeError as e:
        log.warning(f"Failed to parse repaired JSON: {e}")
        log.debug(f"Raw response (first 1000 chars): {raw_response[:1000]}")
        log.debug(f"Repaired (first 1000): {repaired[:1000]}")
        return None
