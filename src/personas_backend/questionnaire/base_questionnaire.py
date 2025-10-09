"""Core questionnaire orchestration primitives.

Defines :class:`QuestionnaireBase`, encapsulating the generic lifecycle:

1. Build/derive system role from a persona description.
2. Send questionnaire JSON prompt to the model (wrapper provided externally).
3. Parse, validate, optionally repair JSON answers.
4. Persist answers (long + wide formats).
5. Optionally compute scores.

Required subclass inputs:
* ``NUM_QUESTIONS`` (int)
* ``QUESTIONNAIRE_JSON`` (str)
* ``_validate_answers`` implementation

Optional:
* ``RESPONSE_MAPPING`` for normalising raw answers
* ``calculate_scores`` override
* ``parse_valid_response`` override for complex schemas (e.g. BigFive)
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import botocore  # type: ignore
import pandas as pd  # type: ignore
import sqlalchemy  # type: ignore
from personas_backend.models_providers.models_utils import get_retry_config


@dataclass
class ExperimentRunResult:
    """Container summarising the outcome of a questionnaire execution."""

    experiment_id: int
    explanation: Optional[str]
    request_json: Optional[Dict[str, Any]] = None
    response_json: Optional[Dict[str, Any]] = None
    request_metadata: Optional[Dict[str, Any]] = None


class QuestionnaireBase:
    """Abstract base coordinating a questionnaire run.

    Extension points
    ----------------
    * ``_validate_answers`` (required)
    * ``calculate_scores`` (optional)
    * ``parse_valid_response`` (override for non-boolean / complex schemas)
    """

    DEFAULT_SCHEMA: Optional[str] = None
    QUESTIONNAIRE_JSON: Optional[str] = None
    EXPECTED_SCHEMA: Optional[Dict[str, Any]] = None
    RESPONSE_MAPPING: Optional[Dict[str, Any]] = None
    NUM_QUESTIONS: Optional[int] = None
    NUMERIC_RESPONSES: bool = False

    def __init__(
        self,
        model_wrapper: Any,
        exp_handler: Any,
        db_conn: Any,
        logger: Optional[logging.Logger] = None,
        schema: Optional[str] = None,
    ) -> None:
        self.model_wrapper = model_wrapper
        self.exp_handler = exp_handler
        self.db_conn = db_conn
        self.logger = logger or logging.getLogger(__name__)
        self.schema = schema or self.DEFAULT_SCHEMA
        self.retry_config = get_retry_config()
        self._repair_attempted: bool = False

    # ------------------------------------------------------------------
    # JSON parsing / repair helpers
    # ------------------------------------------------------------------
    def repair_malformed_json_response(self, json_eval: Dict[str, Any]) -> Dict[str, Any]:
        """Conservative repair keeping only integer-like keys + explanation."""
        repaired: Dict[str, Any] = {}
        if "explanation" in json_eval:
            repaired["explanation"] = json_eval["explanation"]
        for k, v in json_eval.items():
            if k == "explanation":
                continue
            try:
                int(k)
            except ValueError:
                continue
            repaired[k] = v
        return repaired

    # ------------------------------------------------------------------
    # DB safety
    # ------------------------------------------------------------------
    def _rollback_transaction(self) -> None:
        try:  # noqa: BLE001
            if hasattr(self.db_conn, "connection") and hasattr(
                self.db_conn.connection(), "invalidate"
            ):
                self.db_conn.connection().invalidate()
            elif hasattr(self.db_conn, "rollback"):
                self.db_conn.rollback()
        except Exception:  # noqa: BLE001
            self.logger.exception("Rollback failed")

    # ------------------------------------------------------------------
    # Generic parsing path (boolean / mapped answers)
    # ------------------------------------------------------------------
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
        """Parse & persist a questionnaire JSON, returning the explanation text.

        Designed for simple key/value answer sets (EPQR-A style). A single
        automatic repair attempt is performed if initial parsing fails. ``None``
        is returned when parsing or validation fails.
        """
        try:  # noqa: BLE001
            df = pd.DataFrame(json_eval, index=[0]).T.reset_index(drop=False)
            df.rename(
                columns={"index": "question_number", 0: "answer"},
                inplace=True,
            )
            explanation: Optional[str] = None
            if answer_explanation and "explanation" in df.question_number.values:
                explanation = df.loc[
                    df.question_number == "explanation",
                    "answer",
                ].iloc[0]
                df = df[df.question_number != "explanation"]
                try:
                    df["question_number"] = df["question_number"].astype(int)
                except ValueError:
                    repaired = self.repair_malformed_json_response(json_eval)
                    return self.parse_valid_response(
                        experiment_id,
                        repaired,
                        request_json=request_json,
                        response_json=response_json,
                        request_metadata=request_metadata,
                        answer_explanation=answer_explanation,
                        auto_record=auto_record,
                    )
            if df.shape[0] == self.NUM_QUESTIONS:
                if self.RESPONSE_MAPPING:
                    df["answer"] = df.answer.replace(self.RESPONSE_MAPPING).infer_objects()
                if self.NUMERIC_RESPONSES:
                    try:
                        df["answer"] = pd.to_numeric(df["answer"], errors="raise")
                    except ValueError:
                        return None
                if self._validate_answers(df):
                    df = self._insert_db_eval_questionnaires(df, experiment_id)
                    self.calculate_scores(experiment_id, df)
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
            return None
        except Exception:  # noqa: BLE001
            self.logger.exception("parse_valid_response failed")
            if not self._repair_attempted:
                self._repair_attempted = True
                try:  # noqa: BLE001
                    repaired2 = self.repair_malformed_json_response(json_eval)
                    return self.parse_valid_response(
                        experiment_id,
                        repaired2,
                        request_json=request_json,
                        response_json=response_json,
                        request_metadata=request_metadata,
                        answer_explanation=answer_explanation,
                        auto_record=auto_record,
                    )
                except Exception:  # noqa: BLE001
                    self.logger.exception("Repair attempt failed")
                finally:
                    self._repair_attempted = False
            return None

    # ------------------------------------------------------------------
    # Validation & persistence helpers
    # ------------------------------------------------------------------
    def _validate_answers(self, exp_df: pd.DataFrame) -> bool:  # type: ignore[override]
        if exp_df["answer"].isna().sum() > 0:
            return False
        raise NotImplementedError("Subclasses must implement _validate_answers")

    def _insert_db_eval_questionnaires(
        self, exp_df: pd.DataFrame, experiment_id: int
    ) -> pd.DataFrame:
        astype_map = {
            "experiment_id": sqlalchemy.types.Integer,
            "question_number": sqlalchemy.types.Integer,
            "answer": sqlalchemy.types.Integer,
        }
        exp_df["experiment_id"] = experiment_id
        exp_df[
            [
                "experiment_id",
                "question_number",
                "answer",
            ]
        ].to_sql(  # type: ignore[arg-type]
            "eval_questionnaires",
            self.db_conn,
            schema=self.schema,
            if_exists="append",
            index=False,
            dtype=astype_map,  # type: ignore[arg-type]
        )
        return exp_df

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def calculate_scores(
        self, experiment_id: int, exp_df: Optional[pd.DataFrame] = None
    ) -> Optional[Dict[str, Any]]:  # noqa: D401
        return None

    # ------------------------------------------------------------------
    # Execution driver
    # ------------------------------------------------------------------
    def process_experiment(
        self,
        my_experiment: Dict[str, Any],
        persona_description_json: str,
        system_role: Optional[str] = None,
        answer_explanation: bool = True,
        auto_record: bool = True,
    ) -> Optional[ExperimentRunResult]:
        experiment_id = int(my_experiment["experiment_id"])
        self._rollback_transaction()
        attempt = 0
        max_retries = int(self.retry_config.get("max_retries", 3))
        success = False
        if not system_role and persona_description_json:
            try:  # noqa: BLE001
                if isinstance(persona_description_json, dict):
                    items = persona_description_json.items()
                else:
                    items = json.loads(persona_description_json).items()
                formatted = "\n".join(
                    [f"{k}: {v}" for k, v in items if k != "ref_original_description"]
                )
                system_role = (
                    "You are required to adopt and impersonate the "
                    "personality of the human described as follow:\n" + formatted
                )
            except Exception:  # noqa: BLE001
                system_role = (
                    "You are required to adopt and impersonate the "
                    "personality of the human described as follow:\n\n"
                    + str(persona_description_json)
                )
        while attempt < max_retries and not success:
            time.sleep((2**attempt) + random.uniform(0, 1))
            try:  # noqa: BLE001
                request_json, response, valid_response = self.model_wrapper.generate_conversation(
                    user_text_input=self.QUESTIONNAIRE_JSON,
                    my_experiment=my_experiment,
                    system_role=system_role,
                    expected_schema=self.EXPECTED_SCHEMA,
                )
                if (
                    response
                    and isinstance(valid_response, dict)
                    and not valid_response.get("error")
                ):
                    try:  # noqa: BLE001
                        response_payload = (
                            response.model_dump() if hasattr(response, "model_dump") else response
                        )

                        # Extract request metadata from response
                        request_metadata = {}
                        if isinstance(response_payload, dict):
                            if "usage" in response_payload:
                                request_metadata["usage"] = response_payload["usage"]
                            if "model" in response_payload:
                                request_metadata["model"] = response_payload["model"]
                            if "created" in response_payload:
                                request_metadata["created"] = response_payload["created"]

                        explanation = self.parse_valid_response(
                            experiment_id,
                            valid_response,
                            request_json=request_json,
                            response_json=response_payload,
                            answer_explanation=answer_explanation,
                            auto_record=auto_record,
                        )
                        if explanation is not None:
                            success = True
                            if auto_record:
                                return ExperimentRunResult(
                                    experiment_id=experiment_id,
                                    explanation=explanation,
                                )
                            return ExperimentRunResult(
                                experiment_id=experiment_id,
                                explanation=explanation,
                                request_json=request_json,
                                response_json=response_payload,
                                request_metadata=request_metadata,
                            )
                        self.exp_handler.record_request_metadata_and_status(
                            self.db_conn,
                            self.logger,
                            experiment_id=experiment_id,
                            succeeded=False,
                            request_json=request_json,
                            response_json=response_payload,
                            schema=self.schema,
                        )
                    except Exception:  # noqa: BLE001
                        self._rollback_transaction()
                attempt += 1
            except botocore.exceptions.ClientError:  # type: ignore[attr-defined]
                attempt += 1
                if attempt >= max_retries:
                    raise
            except Exception:  # noqa: BLE001
                attempt += 1
                if attempt >= max_retries:
                    raise
        return None
