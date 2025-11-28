"""
JSON response parser for LLM outputs.
Handles parsing and validation of JSON responses from the classifier.
"""

import json
import re
from typing import Optional

from .models import JeopardyClassification
from .logger import logger


def parse_json_response(text: str) -> JeopardyClassification:
    """
    Parse JSON response from LLM output and return JeopardyClassification.

    Parameters:
    -----------
    text: Raw text output from the LLM

    Returns:
    --------
    JeopardyClassification object

    Raises:
    -------
    ValueError: If JSON cannot be parsed or is invalid
    """
    json_str = extract_json_from_text(text)

    if not json_str:
        raise ValueError("No JSON object found in response")

    try:
        data = json.loads(json_str)

        return JeopardyClassification(
            has_numbers=bool(data.get("has_numbers", False)),
            has_non_english=bool(data.get("has_non_english", False)),
            has_unusual_proper_nouns=bool(data.get("has_unusual_proper_nouns", False)),
        )

    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}")
        logger.warning(f"Problematic JSON: {json_str}")
        raise ValueError(f"Invalid JSON format: {e}")

    except Exception as e:
        logger.warning(f"Error parsing JSON response: {e}")
        raise ValueError(f"Failed to create classification object: {e}")


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON object from text that may contain additional content.

    Parameters:
    -----------
    text: Raw text that contains a JSON object

    Returns:
    --------
    Extracted JSON string or None if not found
    """
    text = text.strip()

    json_pattern = r'\{[^{}]*"has_numbers"[^{}]*"has_non_english"[^{}]*"has_unusual_proper_nouns"[^{}]*\}'
    match = re.search(json_pattern, text, re.DOTALL)

    if match:
        return match.group(0)

    brace_start = text.find("{")
    brace_end = text.rfind("}")

    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        return text[brace_start : brace_end + 1]

    return None
