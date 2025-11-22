"""Helpers to retrieve and persist population data in PostgreSQL."""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd  # type: ignore
import sqlalchemy  # type: ignore
from personas_backend.db.db_handler import DatabaseHandler  # type: ignore
from personas_backend.db.schema_config import (  # type: ignore
    get_experimental_schema,
    get_target_schema,
)


class Population:
    """Data access helpers for persona reference and generated populations."""

    def __init__(
        self,
        db_handler: Optional[DatabaseHandler] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize Population handler.

        Args:
            db_handler (DatabaseHandler, optional): Database handler instance.
                If None, a new one will be created.
            logger (logging.Logger, optional): Logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.db_handler = db_handler or DatabaseHandler(logger=self.logger)
        self.connection = self.db_handler.connection

    @staticmethod
    def _validate_identifier(identifier: str) -> str:
        """Ensure schema/table/column names are simple SQL identifiers."""

        if (
            not identifier
            or not identifier.replace("_", "").isalnum()
            or not (identifier[0].isalpha() or identifier[0] == "_")
        ):
            raise ValueError(f"Invalid identifier: {identifier}")
        return identifier

    def get_ref_population_data(
        self, population: str, schema: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve population data from database.

        Args:
            population (str): Population identifier
            schema (str, optional): Database schema to query. If None, uses
                the configured target schema from schema_config.

        Returns:
            pd.DataFrame: Population data
        """
        use_schema = self._validate_identifier(schema or get_target_schema())
        # Source of reference personas is the unified 'personas' table.
        # Map ref_personality_id -> persona_id for downstream compatibility.
        query = sqlalchemy.text(
            f"""
            SELECT ref_personality_id AS persona_id,
                   gender,
                   age,
                   population,
                   description
            FROM {use_schema}.personas
            WHERE population = :population
            ORDER BY persona_id
            """
        )
        self.logger.info(
            "Querying population data for %s from schema %s (personas)",
            population,
            use_schema,
        )
        return pd.read_sql(query, self.connection, params={"population": population})

    def get_baseline_population_data(
        self,
        population: str,
        table_name: str = "exp20250312_generated_population",
    ) -> pd.DataFrame:
        """
        Retrieve population data from database.

        Args:
            population (str): Population identifier

        Returns:
            pd.DataFrame: Population data
        """
        safe_table = self._validate_identifier(table_name)
        query = sqlalchemy.text(
            f"""
            SELECT ref_personality_id AS persona_id,
                   gender,
                   age,
                   population,
                   description
            FROM population.{safe_table}
            WHERE population = :population
            ORDER BY population, persona_id
            """
        )
        self.logger.info(f"Querying population data for {population}")
        return pd.read_sql(query, self.connection, params={"population": population})

    def check_existing_personality(
        self,
        population: str,
        personality_id: str,
        repeated: int,
        new_population_table: str,
        schema: Optional[str] = None,
    ) -> bool:
        """
        Check if a personality already exists in the database.

        Args:
            population (str): Population name
            personality_id (str): Personality ID
            new_population_table (str): Table name where to check

        Returns:
            bool: True if personality exists, False otherwise
        """
        # Prefer experimental schema for generated personas;
        # allow explicit override.
        target_schema = self._validate_identifier(schema or get_experimental_schema())
        safe_table = self._validate_identifier(new_population_table)
        query = sqlalchemy.text(
            f"""
            SELECT ref_personality_id
            FROM {target_schema}.{safe_table}
            WHERE population = :population
              AND ref_personality_id = :personality_id
              AND repetitions = :repetitions
            """
        )
        params = {
            "population": population,
            "personality_id": personality_id,
            "repetitions": repeated,
        }
        existing_ids_df = pd.read_sql_query(query, self.connection, params=params)
        return not existing_ids_df.empty

    def save_generated_persona(
        self,
        persona_df: pd.DataFrame,
        population_table: str,
        schema: Optional[str] = None,
    ) -> None:
        """
        Save generated persona to database.

        Args:
            persona_df (pd.DataFrame): DataFrame containing persona data
            population_table (str): Table name to save the data
            schema (str, optional): Database schema. If None, uses
                configured target schema.
        """
        try:
            # Resolve schema: default to experimental schema for generated data
            schema = schema or get_experimental_schema()
            # Personas table schema (from migration/models). Only keep columns
            # that exist on the target table to avoid append errors.
            allowed_columns = {
                "ref_personality_id",
                "population",
                "model",
                "name",
                "age",
                # Original fields as produced by model
                "gender_original",
                "race_original",
                "sexual_orientation_original",
                "religious_belief_original",
                "occupation_original",
                "political_orientation_original",
                "location_original",
                # Mapped/standardized fields (may be empty initially)
                "gender",
                "race",
                "sexual_orientation",
                "ethnicity",
                "religious_belief",
                "occupation",
                "political_orientation",
                "location",
                "description",
                "word_count_description",
                "repetitions",
                "model_print",
                "population_print",
            }

            # Project the dataframe to allowed columns only
            # (drop extras like ref_population, ref_experiment_id,
            # ref_original_description, json_answer)
            keep_cols = [c for c in persona_df.columns if c in allowed_columns]
            persona_df = persona_df[keep_cols].copy()

            # Define data types for SQL of kept columns only
            dtype_dict = {
                "ref_personality_id": sqlalchemy.types.Integer,
                "population": sqlalchemy.types.String,
                "model": sqlalchemy.types.String,
                "name": sqlalchemy.types.String,
                "age": sqlalchemy.types.Integer,
                "gender_original": sqlalchemy.types.String,
                "race_original": sqlalchemy.types.String,
                "sexual_orientation_original": sqlalchemy.types.String,
                "religious_belief_original": sqlalchemy.types.String,
                "occupation_original": sqlalchemy.types.String,
                "political_orientation_original": sqlalchemy.types.String,
                "location_original": sqlalchemy.types.String,
                "gender": sqlalchemy.types.String,
                "race": sqlalchemy.types.String,
                "sexual_orientation": sqlalchemy.types.String,
                "ethnicity": sqlalchemy.types.String,
                "religious_belief": sqlalchemy.types.String,
                "occupation": sqlalchemy.types.String,
                "political_orientation": sqlalchemy.types.String,
                "location": sqlalchemy.types.String,
                "description": sqlalchemy.types.Text,
                "word_count_description": sqlalchemy.types.Integer,
                "repetitions": sqlalchemy.types.Integer,
                "model_print": sqlalchemy.types.String,
                "population_print": sqlalchemy.types.String,
            }

            full_table_name = f"{schema}.{population_table}"

            # Check if table exists, create if not
            if not self.db_handler.check_table_exists(schema, population_table):
                self.logger.info("Table %s does not exist. Creating it.", full_table_name)
                persona_df.to_sql(
                    population_table,
                    self.connection,
                    schema=schema,
                    if_exists="replace",
                    index=False,
                    dtype=dtype_dict,  # type: ignore[arg-type]
                )
            else:
                self.logger.info("Appending data to existing table %s", full_table_name)
                persona_df.to_sql(
                    population_table,
                    self.connection,
                    schema=schema,
                    if_exists="append",
                    index=False,
                    dtype=dtype_dict,  # type: ignore[arg-type]
                )

            self.logger.info("Successfully saved persona to %s", full_table_name)

        except Exception as e:
            self.logger.error("Error saving persona to database: %s", e)
            raise

    def get_personality_ids(self, population: str, limit: Optional[int] = None) -> List[int]:
        """
        Get list of personality IDs for a specific population.

        Args:
            population (str): Population identifier
            limit (int, optional): Maximum number of IDs to return

        Returns:
            List[int]: List of personality IDs
        """
        # Use unified personas table and configured schema.
        use_schema = self._validate_identifier(get_target_schema())
        query = sqlalchemy.text(
            f"""
            SELECT ref_personality_id AS persona_id
            FROM {use_schema}.personas
            WHERE population = :population
            ORDER BY persona_id
            """
        )

        params: Dict[str, Any] = {"population": population}
        if limit:
            query = sqlalchemy.text(str(query) + " LIMIT :limit")
            params["limit"] = limit

        df = pd.read_sql(query, self.connection, params=params)
        # Explicitly cast to List[int] to avoid the Any return type error
        return [int(pid) for pid in df["persona_id"].tolist()]
