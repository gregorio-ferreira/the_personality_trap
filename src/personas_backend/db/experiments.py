"""Legacy ExperimentHandler (to be refactored) with repository-backed shim.

This file maintains the original interface expected across the codebase
while internally delegating to the new SQLModel-based repository layer.
Gradually callers will be moved to import and use repositories directly.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, TypeVar

import pandas as pd  # type: ignore
from personas_backend.db.db_handler import DatabaseHandler  # type: ignore
from personas_backend.db.models import ExperimentsList  # type: ignore
from personas_backend.db.repositories import (  # type: ignore
    request_metadata_repo as req_meta_repo,
)
from personas_backend.db.repositories.experiments_repo import (  # type: ignore
    ExperimentsRepository,
)
from personas_backend.db.schema_config import get_target_schema
from personas_backend.db.session import get_session  # type: ignore
from personas_backend.utils.post_processing_response import (  # type: ignore
    remove_unicode_chars,
)
from sqlalchemy.engine.base import Connection, Engine  # type: ignore
from sqlalchemy.exc import SQLAlchemyError  # type: ignore

# Type variable for database connection
DBConn = TypeVar("DBConn", Connection, Engine)


class ExperimentHandler:
    """
    Handler for experiment-related database operations.
    """

    def __init__(
        self,
        db_handler: Optional[DatabaseHandler] = None,
        logger: Optional[logging.Logger] = None,
        schema: str = get_target_schema(),
    ):
        """
        Initialize experiment handler.

        Args:
            db_handler (DatabaseHandler, optional): Database handler instance.
                If None, a new one will be created.
            logger (logging.Logger, optional): Logger instance.
            schema (str, optional): Database schema to use (defaults to target
                schema resolved from env/YAML).
        """
        self.logger = logger or logging.getLogger(__name__)
        self.db_handler = db_handler or DatabaseHandler(logger=self.logger)
        self.connection = self.db_handler.connection
        self.schema = schema

    def register_experiment(
        self, my_experiment: Dict[str, Any], schema: Optional[str] = None
    ) -> int:
        """
        Register a new experiment in the database.

        Args:
            my_experiment (Dict[str, Any]): Experiment data dictionary
            schema (str, optional): Database schema. Defaults to
                self.schema.

        Returns:
            int: New experiment ID
        """
        if schema is None:
            schema = self.schema
        try:
            with get_session() as session:  # type: ignore
                repo = ExperimentsRepository(session, schema=schema)
                next_id = repo.next_experiment_id()
                language_questionnaire = my_experiment.get("language_questionnaire")
                if not language_questionnaire:
                    language_questionnaire = my_experiment.get("language_questionnaire_regional")

                exp_kwargs = {
                    "experiment_id": next_id,
                    "experiments_group_id": my_experiment["experiments_group_id"],
                    "repeated": my_experiment.get("repeated"),
                    "questionnaire": my_experiment.get("questionnaire"),
                    "language_instructions": my_experiment.get("language_instructions"),
                    "language_questionnaire": language_questionnaire,
                    "model_provider": my_experiment.get("model_provider"),
                    "model": my_experiment.get("model"),
                    "population": my_experiment.get("population"),
                    "personality_id": my_experiment.get("personality_id"),
                    "repo_sha": my_experiment.get("repo_sha"),
                }
                exp = ExperimentsList(**{k: v for k, v in exp_kwargs.items() if v is not None})
                repo.create_experiment(exp)
                self.logger.info(
                    "Registered new experiment id=%s in schema=%s",
                    exp.experiment_id,
                    schema,
                )
                return int(exp.experiment_id)  # type: ignore[arg-type]

        except Exception as e:
            self.logger.error("Error registering new experiment: %s", e)
            raise

    def get_unprocessed_experiments(
        self, exp_group: int, schema: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get all registered experiments that haven't been fully processed yet.

        Args:
            exp_group (int): Experiment group ID
            schema (str, optional): Database schema. Defaults to
                self.schema.

        Returns:
            pd.DataFrame: DataFrame containing unprocessed experiments
        """
        if schema is None:
            schema = self.schema
        try:
            query = f"""
                SELECT
                    _EXP.QUESTIONNAIRE,
                    _EXP.MODEL_PROVIDER,
                    _EXP.MODEL,
                    _EXP.POPULATION,
                    _EXP.PERSONALITY_ID,
                    _EXP.EXPERIMENT_ID,
                    _EXP.SUCCEEDED,
                    _EXP.LLM_EXPLANATION
                FROM
                    {schema}.EXPERIMENTS_LIST AS _EXP
                WHERE
                    _EXP.EXPERIMENTS_GROUP_ID = {exp_group}
                    AND _EXP.SUCCEEDED IS NULL
                ORDER BY
                    _EXP.EXPERIMENT_ID DESC;
            """
            self.logger.info(
                "Querying unprocessed experiments for experiment_group %s",
                exp_group,
            )
            return pd.read_sql(query, self.connection)
        except Exception as e:
            self.logger.error("Error getting unprocessed experiments: %s", e)
            raise

    def get_ref_experiment_data(
        self, reference_questionnaires_table: str, schema: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve experiment data from database.

        Args:
            reference_questionnaires_table (str): Table name for questionnaires
            schema (str, optional): Database schema. Defaults to
                self.schema.

        Returns:
            pd.DataFrame: Experiment data
        """
        if schema is None:
            schema = self.schema
        try:
            query = f"""
                SELECT
                    PERSONALITY_ID,
                    EXPERIMENT_ID,
                    QUESTIONNAIRE,
                    QUESTION_NUMBER,
                    QUESTION,
                    ANSWER,
                    CATEGORY,
                    KEY,
                    EVAL
                FROM
                    {schema}.{reference_questionnaires_table}
            """
            self.logger.info(
                "Querying experiment data from %s.%s",
                schema,
                reference_questionnaires_table,
            )
            return pd.read_sql(query, self.connection)
        except Exception as e:
            self.logger.error("Error getting reference experiment data: %s", e)
            raise

    def record_request_metadata_and_status(
        self,
        _: DBConn,
        logger: Optional[logging.Logger],
        *,
        experiment_id: int,
        succeeded: bool,
        llm_explanation: Optional[str] = None,
        request_json: Optional[Dict[str, Any]] = None,
        response_json: Optional[Dict[str, Any]] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
        schema: Optional[str] = None,
    ) -> Optional[int]:
        """
        Persist request metadata and update experiment outcome in one step.
        """

        if schema is None:
            schema = self.schema

        use_logger = logger or self.logger
        use_logger.info(
            "%s - Sending to DB, experiment_id: %s",
            datetime.now(),
            experiment_id,
        )

        cleaned_response = remove_unicode_chars(response_json or {})
        metadata_id: Optional[int] = None

        try:
            with get_session() as session:  # type: ignore
                # CRITICAL: Pass schema to repository to ensure correct schema usage
                repo = ExperimentsRepository(session, schema=schema)
                metadata_repo = req_meta_repo.ExperimentRequestMetadataRepository(
                    session, schema=schema
                )

                # Always create metadata if any data is provided
                # (check for None explicitly - empty dicts are falsy!)
                if (
                    request_json is not None
                    or cleaned_response is not None
                    or request_metadata is not None
                ):
                    metadata = metadata_repo.create_entry(
                        experiment_id=experiment_id,
                        request_json=request_json,
                        response_json=cleaned_response,
                        request_metadata=request_metadata,
                    )
                    metadata_id = int(metadata.id) if metadata.id is not None else None

                if succeeded:
                    repo.mark_experiment_succeeded(experiment_id, llm_explanation=llm_explanation)
                    exp = repo.get_experiment(experiment_id)
                    if exp and exp.experiments_group_id:
                        repo.check_and_mark_group_processed(exp.experiments_group_id)
                    use_logger.info(
                        "Updated experiment %s status to succeeded",
                        experiment_id,
                    )
                elif llm_explanation is not None:
                    repo.update_llm_explanation(experiment_id, llm_explanation)

        except SQLAlchemyError as exc:  # pragma: no cover
            self.logger.error("SQLAlchemy Error recording experiment status: %s", exc)
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Error recording experiment status: %s", exc)
            raise

        return metadata_id

    def update_questionnarie_succeeded(
        self, db_conn: DBConn, experiment_id: int, schema: Optional[str] = None
    ) -> None:
        """
        Backward-compatibility wrapper to mark an experiment as succeeded.
        """

        self.record_request_metadata_and_status(
            db_conn,
            self.logger,
            experiment_id=experiment_id,
            succeeded=True,
            schema=schema,
        )

    def get_experiment_data(self, experiment_id: int, schema: Optional[str] = None) -> pd.DataFrame:
        """
        Get experiment data for a specific experiment ID.

        Args:
            experiment_id (int): Experiment ID
            schema (str, optional): Database schema. Defaults to
                self.schema.

        Returns:
            pd.DataFrame: Experiment data
        """
        if schema is None:
            schema = self.schema
        try:
            query = (
                "SELECT * FROM "
                f"{schema}.questionnaire_answers WHERE experiment_id = "
                f"{experiment_id} ORDER BY personality_id, question_number"
            )
            self.logger.info(
                "Querying experiment data for experiment_id %s",
                experiment_id,
            )
            assert self.connection is not None  # help type checker
            return pd.read_sql(query, self.connection)
        except Exception as e:
            self.logger.error("Error getting experiment data: %s", e)
            raise

    # NOTE: Deprecated legacy save_experiment_results removed during cleanup.
