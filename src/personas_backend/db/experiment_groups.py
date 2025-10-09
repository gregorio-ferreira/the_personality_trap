"""Interface for managing experiment groups within PostgreSQL schemas."""

import logging
from typing import Any, Dict, Optional

from personas_backend.db.db_handler import DatabaseHandler
from personas_backend.db.repositories.experiments_repo import (
    ExperimentsRepository,
)
from personas_backend.db.schema_config import get_target_schema
from personas_backend.db.session import get_session


class ExperimentGroupHandler:
    """Handler for experiment group-related database operations.

    An **experiment group** is a logical container that defines a batch of
    related experiments. It stores:
    - Description: What we're testing (e.g., "bigfive questionnaire for GPT-4o")
    - Configuration: LLM parameters (temperature, top_p)
    - Prompts: System role and questionnaire JSON
    - Metadata: Timestamps, processing status

    **Architecture:**
    ```
    experiments_groups (1 record) ─┬─> experiments_list (N records)
                                   ├─> One experiment per persona × repetition
                                   └─> Linked via experiments_group_id
    ```

    **Usage Pattern:**
    1. Create experiment group first (defines "what" and "how")
    2. Register individual experiments that reference the group
    3. Execute experiments (populate answers and metadata)
    4. Mark group as concluded when all experiments complete

    This handler is typically used by higher-level registration APIs
    (e.g., register_questionnaire_experiments) rather than called directly.
    """

    def __init__(
        self,
        db_handler: Optional[DatabaseHandler] = None,
        logger: Optional[logging.Logger] = None,
        schema: str = get_target_schema(),
    ) -> None:
        """Initialize experiment group handler.

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

    def register_new_experiments_group(
        self,
        description: str,
        system_role: str,
        base_prompt: str,
        translated: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        ii_repeated: int = 0,
        schema: Optional[str] = None,
    ) -> int:
        """Register a new experiment group (repository-backed)."""
        if schema is None:
            schema = self.schema
        try:
            with get_session() as session:  # type: ignore
                repo = ExperimentsRepository(session, schema=schema)
                group = repo.create_group(
                    description=description,
                    system_role=system_role,
                    base_prompt=base_prompt,
                    translated=translated,
                    temperature=temperature,
                    top_p=top_p,
                    ii_repeated=bool(ii_repeated),
                )
                self.logger.info(
                    "Registered new experiment group with ID: %s in schema: %s",
                    group.experiments_group_id,
                    schema,
                )
                return int(group.experiments_group_id)  # type: ignore[arg-type]
        except Exception as e:  # broad to preserve legacy behavior
            self.logger.error("Error registering experiment group: %s", e)
            raise

    def process_experiment_group(
        self,
        exp_group_id: Optional[int],
        description_prefix: str,
        exp_group_dict: Dict[str, Any],
    ) -> int:
        """
        Process experiment group, creating a new one if needed.

        Args:
            exp_group_id: Existing experiment group ID or None
            description_prefix (str): Prefix for the experiment group
                description
            exp_group_dict (Dict[str, Any]): Dictionary with experiment group
                data

        Returns:
            int: Experiment group ID
        """
        if not exp_group_id:
            exp_group_description = (
                f"{description_prefix} - Temp: "
                f"{exp_group_dict['model_config']['TEMPERATURE']}, "
                f"model: {exp_group_dict['model']}, lang_q: "
                f"{exp_group_dict['LANGUAGE_QUESTIONNAIRE']}, population: "
                f"{exp_group_dict['model_config']['POPULATION']}, "
                f"ii_repeated: {exp_group_dict['REPETITIONS'] > 1}"
            )

            exp_group_id = self.register_new_experiments_group(
                exp_group_description,
                exp_group_dict["system_role"],
                exp_group_dict["questionnaire_json"],
                exp_group_dict["TRANSLATED"],
                exp_group_dict["model_config"]["TEMPERATURE"],
                exp_group_dict["model_config"]["TOP_P"],
                True if exp_group_dict["REPETITIONS"] > 1 else False,
            )

            self.logger.info("Created experiment group ID: %s", exp_group_id)
        else:
            self.logger.info("Processing existing exp_group_id: %s", exp_group_id)
        return int(exp_group_id)

    def update_experiments_groups_conclude(
        self,
        experiments_group_id_start: int,
        experiments_group_id_end: int,
        schema: Optional[str] = None,
    ) -> None:
        """
        Update experiment groups to mark them as concluded.

        Args:
            experiments_group_id_start (int): Start ID of experiment groups
            experiments_group_id_end (int): End ID of experiment groups
            schema (str, optional): Database schema. Defaults to self.schema.
        """
        if schema is None:
            schema = self.schema
        try:
            with get_session() as session:  # type: ignore
                repo = ExperimentsRepository(session)
                updated = repo.mark_groups_concluded(
                    experiments_group_id_start,
                    experiments_group_id_end,
                )
                self.logger.info(
                    "Updated %d experiment groups (%d-%d) as concluded",
                    updated,
                    experiments_group_id_start,
                    experiments_group_id_end,
                )
        except Exception as e:  # broad to match legacy interface
            self.logger.error("Error updating experiment groups: %s", e)
            raise
