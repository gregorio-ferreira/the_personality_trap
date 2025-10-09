"""Utilities to clean, validate, and repair JSON responses from models."""

import json
import logging
import re
from typing import Any, Dict, Optional, cast

from json_repair import repair_json

logger = logging.getLogger(__name__)


def remove_unicode_chars(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove unicode characters from a dictionary.

    Args:
        d (Dict[str, Any]): Dictionary with potential unicode characters

    Returns:
        Dict[str, Any]: Dictionary with unicode characters removed
    """
    new_dict = {}
    for key, value in d.items():
        # Ensure the key is a string, then encode to ASCII ignoring errors, and decode back to a string
        ascii_key = str(key).encode("ascii", "ignore").decode("ascii")

        # Use Any type for ascii_value since it can be any type
        ascii_value: Any

        if isinstance(value, str):
            # For string values, remove non-ASCII characters the same way
            ascii_value = value.encode("ascii", "ignore").decode("ascii")
        elif isinstance(value, dict):
            # If the value is a dictionary, recursively clean it
            ascii_value = remove_unicode_chars(value)
        else:
            # For other types, keep the value as is
            ascii_value = value
        new_dict[ascii_key] = ascii_value
    return new_dict


def remove_markdown_code_blocks(input_string: str) -> str:
    """
    Remove markdown code block delimiters (```json, ```) from the input string.

    Args:
        input_string (str): Input string possibly containing markdown code blocks

    Returns:
        str: String with markdown code block markers removed
    """
    # Remove starting code block marker (```json or just ```)
    if input_string.strip().startswith("```"):
        # Find the first line break after the marker
        first_break = input_string.find("\n")
        if first_break != -1:
            input_string = input_string[first_break + 1 :]

    # Remove ending code block marker (```)
    if input_string.strip().endswith("```"):
        # Find the last occurrence of ```
        last_marker = input_string.rfind("```")
        if last_marker != -1:
            input_string = input_string[:last_marker]

    return input_string.strip()


def remove_before_first_brace(input_string: str) -> str:
    """
    Remove text before the first '{' character.

    Args:
        input_string (str): Input string

    Returns:
        str: String starting from the first '{'
    """
    # Find the index of the first '{'
    brace_index = input_string.find("{")

    # If a '{' is found, return the string from there; otherwise, return the original string
    return input_string[brace_index:] if brace_index != -1 else input_string


def remove_after_last_brace(input_string: str) -> str:
    """
    Remove text after the last '}' character.

    Args:
        input_string (str): Input string

    Returns:
        str: String ending at the last '}'
    """
    # Find the index of the last '}'
    brace_index = input_string.rfind("}")

    # If a '}' is found, return the string up to and including that character;
    # otherwise, return the original string
    return input_string[: brace_index + 1] if brace_index != -1 else input_string


def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON content from a response that may contain markdown code blocks.

    Args:
        response_text (str): The text response that may contain JSON in code blocks.

    Returns:
        dict or None: Parsed JSON object if found, None otherwise.
    """
    # Pattern to match JSON content inside triple backticks with optional 'json' language specifier
    pattern = r"```(?:json)?\s*([\s\S]*?)```"

    matches = re.search(pattern, response_text)

    if matches:
        json_string = matches.group(1).strip()
        try:
            return cast(Dict[str, Any], json.loads(json_string))
        except json.JSONDecodeError:
            logger.debug(
                "Found code block but could not parse as JSON: %s...",
                json_string[:100],
            )
            return None

    # If no code blocks found, try parsing directly
    try:
        return cast(Dict[str, Any], json.loads(response_text))
    except json.JSONDecodeError:
        return None


def clean_json_string(input_string: str, logger: Optional[logging.Logger] = None) -> str:
    """
    Apply all preprocessing steps to clean a string for JSON parsing.

    Args:
        input_string (str): Input string to clean
        logger (logging.Logger, optional): Logger for logging messages

    Returns:
        str: Cleaned string ready for JSON parsing
    """
    if not input_string:
        return ""

    # Create a local logger if none provided
    log = logger or logging.getLogger(__name__)

    # 1. Remove markdown code blocks
    input_string = remove_markdown_code_blocks(input_string)

    # 2. Remove anything before the first '{'
    input_string = remove_before_first_brace(input_string)

    # 3. Remove anything after the last '}'
    input_string = remove_after_last_brace(input_string)

    log.debug(f"Cleaned JSON string: {input_string[:100]}...")

    return input_string


def validate_and_fix_json(
    input_string: str,
    expected_schema: Optional[Dict[str, str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Validate and fix a JSON string, with schema validation.

    Args:
        input_string (str): Input JSON string
        expected_schema (Dict[str, str], optional): Expected schema for validation
        logger (logging.Logger, optional): Logger for logging messages

    Returns:
        Dict[str, Any]: Parsed and validated JSON
    """
    # Create a local logger if none provided
    log = logger or logging.getLogger(__name__)

    if not input_string:
        return {"error": "Empty response"}

    # Clean the input string
    input_string = clean_json_string(input_string, log)

    # Parse JSON
    try:
        parsed_json = json.loads(input_string)
    except json.JSONDecodeError as e:
        log.info(f"Initial JSON parsing failed: {e}, attempting repair")

        # Try json_repair
        try:
            repaired_json = repair_json(input_string)
            log.info("JSON repaired successfully")

            # Check the type of repaired_json before parsing
            if isinstance(repaired_json, (dict, list)):
                # If repair_json returned an already parsed object, use it directly
                parsed_json = repaired_json
            elif isinstance(repaired_json, (str, bytes, bytearray)):
                # If it's a string/bytes type, parse it
                parsed_json = json.loads(repaired_json)
            else:
                # If it's another type, convert to string first
                parsed_json = json.loads(str(repaired_json))
        except Exception as repair_error:
            log.error(f"JSON repair failed: {repair_error}")

            # Last resort - try common LLM formatting issues
            try:
                # Remove escaped quotes and fix newlines
                cleaned = input_string.replace('\\"', '"').replace("\\n", " ")
                # Make sure we have a string before parsing
                if not isinstance(cleaned, str):
                    cleaned = str(cleaned)
                # Try parsing again
                parsed_json = json.loads(cleaned)
            except Exception as final_error:
                log.error(f"All JSON parsing attempts failed: {final_error}")
                return {"error": f"Unable to parse JSON: {str(final_error)}"}

    # Schema validation if schema provided
    if expected_schema and parsed_json:
        missing_fields = []
        type_errors = []

        for field, expected_type in expected_schema.items():
            # Check for missing fields
            if field not in parsed_json:
                missing_fields.append(field)
                continue

            # Basic type validation
            if expected_type == "integer":
                try:
                    # Convert to int if it's a string number
                    if isinstance(parsed_json[field], str) and parsed_json[field].isdigit():
                        parsed_json[field] = int(parsed_json[field])
                    # Validate it's actually an int
                    if not isinstance(parsed_json[field], int):
                        type_errors.append(
                            f"{field} should be integer, got {type(parsed_json[field]).__name__}"
                        )
                except (ValueError, TypeError):
                    type_errors.append(f"{field} conversion to integer failed")

        # Log validation issues
        if missing_fields:
            log.warning(f"JSON missing required fields: {missing_fields}")
            # Set default values for missing fields
            for field in missing_fields:
                if expected_schema[field] == "string":
                    parsed_json[field] = ""
                elif expected_schema[field] == "integer":
                    parsed_json[field] = 0

        if type_errors:
            log.warning(f"JSON type validation errors: {type_errors}")

    # Ensure we're returning a dictionary
    if not isinstance(parsed_json, dict):
        log.warning(f"Parsed JSON is not a dictionary, wrapping it: {type(parsed_json).__name__}")
        return {"error": "Parsed JSON is not a dictionary"}

    return parsed_json


def process_model_response(
    response_text: str,
    expected_schema: Optional[Dict[str, str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Process an LLM response text to extract and validate JSON.
    Comprehensive function that combines all processing steps.

    Args:
        response_text (str): Raw response text from LLM
        expected_schema (Dict[str, str], optional): Schema for validation
        logger (logging.Logger, optional): Logger for logging

    Returns:
        Dict[str, Any]: Processed and validated JSON
    """
    # Create a local logger if none provided
    log = logger or logging.getLogger(__name__)

    if not response_text:
        return {"error": "Empty response"}

    # First try to extract JSON from markdown code blocks
    extracted_json = extract_json_from_response(response_text)
    if extracted_json:
        log.info("Successfully extracted JSON from markdown code blocks")
        # Still validate against schema if provided
        if expected_schema:
            return validate_schema(extracted_json, expected_schema, log)
        # Make sure we're returning a dict
        if isinstance(extracted_json, dict):
            return extracted_json
        else:
            log.warning(f"Expected dict but got {type(extracted_json).__name__}")
            return {"result": extracted_json}

    # If that fails, try the full validation pipeline
    return validate_and_fix_json(response_text, expected_schema, log)


def validate_schema(
    json_data: Dict[str, Any],
    schema: Dict[str, str],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Validate JSON data against a schema and fix type issues.

    Args:
        json_data (Dict[str, Any]): JSON data to validate
        schema (Dict[str, str]): Schema defining expected types
        logger (logging.Logger, optional): Logger for messages

    Returns:
        Dict[str, Any]: Validated and fixed JSON data
    """
    log = logger or logging.getLogger(__name__)

    # Ensure we're working with a dictionary
    if not isinstance(json_data, dict):
        log.warning(f"Expected dict but got {type(json_data).__name__}. Returning empty dict.")
        return {}

    if not schema:
        return json_data

    missing_fields = []
    type_errors = []

    for field, expected_type in schema.items():
        # Check for missing fields
        if field not in json_data:
            missing_fields.append(field)
            continue

        # Type validation and conversion
        if expected_type == "integer":
            try:
                # Convert to int if it's a string number
                if isinstance(json_data[field], str) and json_data[field].isdigit():
                    json_data[field] = int(json_data[field])
                # Validate it's actually an int
                if not isinstance(json_data[field], int):
                    type_errors.append(
                        f"{field} should be integer, got {type(json_data[field]).__name__}"
                    )
            except (ValueError, TypeError):
                type_errors.append(f"{field} conversion to integer failed")

    # Set default values for missing fields
    if missing_fields:
        log.warning(f"JSON missing required fields: {missing_fields}")
        for field in missing_fields:
            if schema[field] == "string":
                json_data[field] = ""
            elif schema[field] == "integer":
                json_data[field] = 0

    if type_errors:
        log.warning(f"JSON type validation errors: {type_errors}")

    return json_data
