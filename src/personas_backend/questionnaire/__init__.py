"""
Questionnaire module for personas_backend.

This module provides functionality for administering and analyzing
different personality questionnaires.
"""

from typing import Any, Optional

from personas_backend.questionnaire.bigfive import BigFive
from personas_backend.questionnaire.bigfive import (
    questionnaire_json as big_five_questionnaire_json,
)
from personas_backend.questionnaire.epqr_a import EPQRA
from personas_backend.questionnaire.epqr_a import (
    questionnaire_json as epqra_questionnaire_json,
)


def get_questionnaire_handler(
    questionnaire_type: str,
    model_wrapper: Any,
    exp_handler: Any,
    db_conn: Any,
    logger: Optional[Any] = None,
    schema: Optional[str] = None,
) -> Any:
    """
    Factory function to get the appropriate questionnaire handler.

    Args:
        questionnaire_type (str): Type of questionnaire ('epqra' or 'bigfive')
        model_wrapper: Model wrapper instance
        exp_handler: Experiment handler instance
        db_conn: Database connection
        logger: Logger instance (optional)
        schema: Database schema (optional)

    Returns:
        QuestionnaireBase: An instance of the appropriate questionnaire handler

    Raises:
        ValueError: If questionnaire_type is not supported
    """
    if questionnaire_type == "epqra":
        return EPQRA(model_wrapper, exp_handler, db_conn, logger, schema)
    elif questionnaire_type in ["bigfive", "big_five", "big5"]:
        return BigFive(model_wrapper, exp_handler, db_conn, logger, schema)
    else:
        raise ValueError(f"Unsupported questionnaire type: {questionnaire_type}")


def get_questionnaire_json(questionnaire_type: str) -> str:
    """
    Get the questionnaire JSON for a specified questionnaire type.

    Args:
        questionnaire_type (str): Type of questionnaire ('epqra' or 'bigfive')

    Returns:
        str: The questionnaire JSON

    Raises:
        ValueError: If questionnaire_type is not supported
    """
    if questionnaire_type == "epqra":
        return str(epqra_questionnaire_json)
    elif questionnaire_type in ["bigfive", "big_five", "big5"]:
        return str(big_five_questionnaire_json)
    else:
        raise ValueError(f"Unsupported questionnaire type: {questionnaire_type}")
