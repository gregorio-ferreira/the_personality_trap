from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from personas_backend.db.schema_config import get_target_schema
from personas_backend.questionnaire.base_questionnaire import QuestionnaireBase


class BigFive(QuestionnaireBase):
    """
    Class for handling the Big Five personality assessment.
    """

    # Default schema name used in database operations
    DEFAULT_SCHEMA = get_target_schema()

    # Flag: questionnaire uses numeric responses (1-5) not boolean
    NUMERIC_RESPONSES = True

    # Number of questions in the questionnaire
    NUM_QUESTIONS = 44

    # Questionnaire content
    QUESTIONNAIRE_JSON = """
Here are a number of characteristics that may or may not apply to you.
For example, do you agree that you are someone who likes to spend
time with others?
Please write a number next to each statement to indicate the extent
to which you agree or disagree with that statement. For each
question, you will respond on a scale from 1 to 5, indicating how
much you agree or disagree that the statement applies to you.

Use the following scale for your answers:

1: Disagree strongly
2: Disagree a little
3: Neither agree nor disagree
4: Agree a little
5: Agree strongly

- Each question must be interpreted with the preamble: "I see myself as
someone who...".
{
    "1": "Is talkative",
    "2": "Tends to find fault with others",
    "3": "Does a thorough job",
    "4": "Is depressed, blue",
    "5": "Is original, comes up with new ideas",
    "6": "Is reserved",
    "7": "Is helpful and unselfish with others",
    "8": "Can be somewhat careless",
    "9": "Is relaxed, handles stress well",
    "10": "Is curious about many different things",
    "11": "Is full of energy",
    "12": "Starts quarrels with others",
    "13": "Is a reliable worker",
    "14": "Can be tense",
    "15": "Is ingenious, a deep thinker",
    "16": "Generates a lot of enthusiasm",
    "17": "Has a forgiving nature",
    "18": "Tends to be disorganized",
    "19": "Worries a lot",
    "20": "Has an active imagination",
    "21": "Tends to be quiet",
    "22": "Is generally trusting",
    "23": "Tends to be lazy",
    "24": "Is emotionally stable, not easily upset",
    "25": "Is inventive",
    "26": "Has an assertive personality",
    "27": "Can be cold and aloof",
    "28": "Perseveres until the task is finished",
    "29": "Can be moody",
    "30": "Values artistic, aesthetic experiences",
    "31": "Is sometimes shy, inhibited",
    "32": "Is considerate and kind to almost everyone",
    "33": "Does things efficiently",
    "34": "Remains calm in tense situations",
    "35": "Prefers work that is routine",
    "36": "Is outgoing, sociable",
    "37": "Is sometimes rude to others",
    "38": "Makes plans and follows through with them",
    "39": "Gets nervous easily",
    "40": "Likes to reflect, play with ideas",
    "41": "Has few artistic interests",
    "42": "Likes to cooperate with others",
    "43": "Is easily distracted",
    "44": "Is sophisticated in art, music, or literature"
}
- You will receive the questions as a JSON object with numbers as keys
and statements as values.
- You must reply exclusively with a JSON object. The JSON should:
    - Use the same question numbers (as string keys) to record your answers.
        - Include an additional key `"explanation"`, containing a brief
            explanation (under 100 words) summarizing the reasoning behind
            your responses.
- Do not include extra text, markdown, code blocks, or commentary.
"""

    # Expected schema for validation - all 44 questions plus explanation
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
        "25": "string",
        "26": "string",
        "27": "string",
        "28": "string",
        "29": "string",
        "30": "string",
        "31": "string",
        "32": "string",
        "33": "string",
        "34": "string",
        "35": "string",
        "36": "string",
        "37": "string",
        "38": "string",
        "39": "string",
        "40": "string",
        "41": "string",
        "42": "string",
        "43": "string",
        "44": "string",
        "explanation": "string",
    }

    # Scoring rules for calculating personality traits
    SCORING_RULES = {
        "extraversion": [1, -6, 11, 16, -21, 26, -31, 36],
        "agreeableness": [-2, 7, -12, 17, 22, -27, 32, -37, 42],
        "conscientiousness": [3, -8, 13, -18, -23, 28, 33, 38, -43],
        "neuroticism": [4, -9, 14, 19, -24, 29, -34, 39],
        "openness": [5, 10, 15, 20, 25, 30, -35, 40, -41, 44],
    }

    # Values must be in the range 1-5
    VALID_ANSWER_RANGE = range(1, 6)

    def parse_valid_response(
        self,
        experiment_id: int,
        json_eval: Dict[str, Any],
        *,
        request_json: Optional[Dict[str, Any]] = None,
        response_json: Optional[Dict[str, Any]] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
        answer_explanation: bool = True,
        auto_record: bool = True,
    ) -> Optional[str]:
        """
        Parse and validate the BigFive questionnaire response.

        Args:
            experiment_id: ID of the experiment
            json_eval: JSON response from the model
            request_json: Original request payload sent to the model
            response_json: Raw response payload from the model
            request_metadata: Optional metadata captured with the request
            answer_explanation: Whether to process explanation field

        Returns:
            Optional[str]: Explanation text when parsing succeeds
        """
        try:
            exp_df = pd.DataFrame(json_eval, index=[0]).T
            exp_df.reset_index(inplace=True, drop=False)
            exp_df.rename(columns={"index": "question_number", 0: "answer"}, inplace=True)

            explanation = None

            if answer_explanation:
                explanation = exp_df.loc[exp_df.question_number == "explanation", "answer"].iloc[0]
                # Remove the 'explanation' row
                exp_df = exp_df[exp_df.question_number != "explanation"]
                # Convert the question_number column to integers
                exp_df["question_number"] = exp_df["question_number"].astype(int)

            # Expect 44 questions for BigFive
            if exp_df.shape[0] == self.NUM_QUESTIONS:
                # Convert answers to integers
                try:
                    exp_df["answer"] = pd.to_numeric(exp_df["answer"], errors="raise")
                except ValueError:
                    self.logger.warning(
                        "Non-numeric answers detected for experiment_id: %s",
                        experiment_id,
                    )
                    return None

                # Check if all answers are within valid range
                valid_answers = exp_df["answer"].apply(lambda x: x in self.VALID_ANSWER_RANGE)
                if valid_answers.all():
                    # Store responses in database
                    exp_df = self._insert_db_eval_questionnaires(exp_df, experiment_id)

                    # Calculate trait scores from answers but don't store them
                    self.calculate_scores(experiment_id, exp_df)

                    if auto_record:
                        self.exp_handler.record_request_metadata_and_status(
                            self.db_conn,
                            self.logger,
                            experiment_id=experiment_id,
                            succeeded=True,
                            llm_explanation=explanation,
                            request_json=request_json,
                            response_json=response_json,
                            request_metadata=request_metadata,
                            schema=self.schema,
                        )
                    return explanation
                else:
                    self.logger.warning(
                        "Invalid answer values for experiment_id: %s, " "not all in range 1-5",
                        experiment_id,
                    )
            else:
                self.logger.warning(
                    "Invalid response format for experiment_id: %s, shape: %s",
                    experiment_id,
                    exp_df.shape,
                )
            return None

        except Exception as e:
            self.logger.exception(f"Error parsing BigFive response: {str(e)}")
            return None

    def _validate_answers(self, exp_df: pd.DataFrame) -> bool:  # type: ignore[override]
        """
        Validate answers for Big Five questionnaire.
        Answers should be numeric values from 1 to 5.

        Args:
            exp_df: DataFrame with question_number and answer columns

        Returns:
            bool: True if answers are valid, False otherwise
        """
        # Convert to numeric if needed
        try:
            if not pd.api.types.is_numeric_dtype(exp_df["answer"]):
                exp_df["answer"] = pd.to_numeric(exp_df["answer"], errors="raise")
        except ValueError:
            self.logger.warning("BigFive answers contain non-numeric values")
            return False

        # Check if answers are within the valid range (1-5)
        valid_range = exp_df["answer"].apply(lambda x: x in self.VALID_ANSWER_RANGE)

        if not valid_range.all():
            invalid_values = exp_df.loc[~valid_range, ["question_number", "answer"]]
            self.logger.warning(
                "BigFive answers outside valid range 1-5: %s",
                invalid_values.to_dict(),
            )
            return False

        # Check for reasonable variance in responses
        std_dev = exp_df["answer"].std()
        if std_dev < 0.5:  # This threshold can be adjusted
            self.logger.warning(f"BigFive answers show low variance: std_dev={std_dev:.2f}")
            return False

        return True

    def calculate_scores(  # type: ignore[override]
        self, experiment_id: int, exp_df: Optional[pd.DataFrame] = None
    ) -> Optional[Dict[str, Optional[float]]]:
        """
        Calculate Big Five trait scores for an experiment.
        If exp_df is not provided, will fetch data from the database.

        Args:
            experiment_id: ID of the experiment
            exp_df: DataFrame with responses (optional)

        Returns:
            dict: Dictionary with calculated trait scores
        """
        try:
            # If exp_df is not provided, retrieve answers from database
            if exp_df is None:
                # Query DB to get answers for the specified experiment
                query = f"""
                SELECT question_number, answer
                FROM {self.schema}.eval_questionnaires
                WHERE experiment_id = :experiment_id
                """
                exp_df = pd.read_sql_query(
                    query, self.db_conn, params={"experiment_id": experiment_id}
                )

                if exp_df.empty:
                    self.logger.warning(f"No answers found for experiment {experiment_id}")
                    return None

            # Create a dictionary to hold trait scores
            scores: Dict[str, Optional[float]] = {}

            # Calculate scores for each trait
            for trait, items in self.SCORING_RULES.items():
                total = 0
                count = 0
                for item in items:
                    if item < 0:
                        # Reverse scoring (6 minus the answer for 1-5 scale)
                        item_num = abs(item)
                        answer_row = exp_df.loc[exp_df["question_number"] == item_num]
                        if not answer_row.empty:
                            total += 6 - answer_row["answer"].iloc[0]
                            count += 1
                    else:
                        answer_row = exp_df.loc[exp_df["question_number"] == item]
                        if not answer_row.empty:
                            total += answer_row["answer"].iloc[0]
                            count += 1

                # Calculate the average score if we have answers
                if count > 0:
                    scores[trait] = round(total / count, 2)
                else:
                    scores[trait] = None

            self.logger.info(
                "Calculated trait scores for experiment %s: %s",
                experiment_id,
                scores,
            )
            return scores

        except Exception as e:
            self.logger.exception(f"Error calculating trait scores: {str(e)}")
            return None


# Backwards compatibility: expose questionnaire JSON variables
questionnaire_json = BigFive.QUESTIONNAIRE_JSON
system_prompt_expected_schema = BigFive.EXPECTED_SCHEMA
