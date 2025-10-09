from typing import Any, Optional

import pandas as pd
from personas_backend.db.schema_config import get_target_schema
from personas_backend.questionnaire.base_questionnaire import QuestionnaireBase


class EPQRA(QuestionnaireBase):
    """Handler for the EPQR-A (Eysenck Personality Questionnaire
    Revised - Abbrev.).

    Provides functionality to:
    - Present the questionnaire
    - Process and validate responses
    - Store results in the database
    - Handle retries and error cases
    """

    # Default schema name used in database operations
    DEFAULT_SCHEMA = get_target_schema()

    # Number of questions in the questionnaire
    NUM_QUESTIONS = 24

    # Uses boolean responses (True/False) but stores as integers (1/0)
    NUMERIC_RESPONSES = False

    # Questionnaire content
    QUESTIONNAIRE_JSON = """
You are being asked to complete a questionnaire.

**Questionnaire:**

{
  "1": "Does your mood often go up and down?",
  "2": "Are you a talkative person?",
  "3": "Would being in debt worry you?",
  "4": "Are you rather lively?",
    "5": "Were you ever greedy by taking more than your share of anything?",
  "6": "Would you take drugs which may have strange or dangerous effects?",
    "7": "Have you ever blamed someone for what you knew was your fault?",
  "8": "Do you prefer to go your own way rather than act by the rules?",
  "9": "Do you often feel 'fed-up'?",
        "10": "Have you ever taken anything (even a pin) that belonged to "
        "someone else?",
  "11": "Would you call yourself a nervous person?",
  "12": "Do you think marriage is old-fashioned and should be done away with?",
  "13": "Can you easily get some life into a rather dull party?",
  "14": "Are you a worrier?",
  "15": "Do you tend to keep in the background on social occasions?",
  "16": "Does it worry you if you know there are mistakes in your work?",
  "17": "Have you ever cheated at a game?",
  "18": "Do you suffer from 'nerves'?",
  "19": "Have you ever taken advantage of someone?",
  "20": "Are you mostly quiet when you are with other people?",
  "21": "Do you often feel lonely?",
  "22": "Is it better to follow society's rules than go your own way?",
  "23": "Do other people think of you as being very lively?",
  "24": "Do you always practice what you preach?"
}

**Instructions:**

1. **Answer Format:** Provide answers in one JSON object. Use question numbers
    as keys (strings) and the responses as values.

2. **Responses:** For each question answer only with `"True"` or `"False"`.

3. **Order:** Keep the same order and numbering.

4. **Explanation:** Add key `"explanation"` with <100 word summary of your
    reasoning.

5. **Output Only JSON:** Respond only with the JSON object. No extra text,
   markdown, or commentary.

**Example Response Format:**

{
  "1": "True",
  "2": "False",
  "3": "True",
  "...": "...",
  "24": "False",
  "explanation": "explain your reasoning here"
}
"""

    # Expected schema for validation
    EXPECTED_SCHEMA = {
        "1": "string",
        "2": "string",
        "3": "string",
        "4": "string",
        "5": "string",
        "6": "string",
        "7": "string",
        "8": "string",
        "9": "string",
        "10": "string",
        "11": "string",
        "12": "string",
        "13": "string",
        "14": "string",
        "15": "string",
        "16": "string",
        "17": "string",
        "18": "string",
        "19": "string",
        "20": "string",
        "21": "string",
        "22": "string",
        "23": "string",
        "24": "string",
        "explanation": "string",
    }

    # Value mapping for standardizing responses - now maps to integers
    RESPONSE_MAPPING = {
        "False": 0,
        "Falso": 0,
        "Falša": 0,
        "Falšovanie": 0,
        "Falošné": 0,
        "No": 0,
        "false": 0,
        "Pravda": 1,
        "Nepravda": 1,
        "True": 1,
        "Verdadero": 1,
        "Yes": 1,
        "Cierto": 1,
        "true": 1,
        "Verdad": 1,
    }

    def _validate_answers(self, exp_df: pd.DataFrame) -> bool:
        """
        Validate answers for EPQR-A questionnaire.
        Answers should be boolean values (mapped to 0 or 1).

        Args:
            exp_df: DataFrame with question_number and answer columns

        Returns:
            bool: True if answers are valid, False otherwise
        """
        # Check if answers are all 0 or 1
        valid_values = exp_df["answer"].isin([0, 1])

        if not valid_values.all():
            self.logger.warning("EPQRA answers contain values other than 0 and 1")
            return False

        # Basic check: ensure not all answers are the same
        answer_sum = exp_df.answer.sum()
        if answer_sum == 0 or answer_sum == self.NUM_QUESTIONS:
            self.logger.warning(f"EPQRA answers are all the same: {answer_sum}")
            return False

        return True


# For backwards compatibility, expose key functions as module-level functions
def create_epqra_handler(
    model_wrapper: Any,
    exp_handler: Any,
    db_conn: Any,
    logger: Optional[Any] = None,
    schema: Optional[str] = None,
) -> EPQRA:
    """Factory function to create an EPQRA handler instance."""
    return EPQRA(model_wrapper, exp_handler, db_conn, logger, schema)


# Expose questionnaire JSON at module level for backwards compatibility
questionnaire_json = EPQRA.QUESTIONNAIRE_JSON
system_prompt_expected_schema = EPQRA.EXPECTED_SCHEMA
