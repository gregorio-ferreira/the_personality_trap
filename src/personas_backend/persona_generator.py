"""
Unified PersonaGenerator for AI persona generation with questionnaire data.

This module provides a comprehensive persona generation system that supports:
- Multiple LLM providers (OpenAI, Bedrock, local models)
- Parallel processing for batch generation
- Borderline personality conditions for research
- Flexible database schema/table configurations
- Both standalone and integrated usage patterns
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from personas_backend.core.enums import ModelID, PersonaGenerationCondition
from personas_backend.db.db_handler import DatabaseHandler
from personas_backend.models_providers.models_config import get_models_config
from personas_backend.models_providers.models_utils import get_model_wrapper
from personas_backend.population.population import Population
from personas_backend.utils.config import ConfigManager
from sqlalchemy import text

# Experimental models and conditions that require explicit opt-in
EXPERIMENTAL_MODELS = [ModelID.LLAMA3_170B]
EXPERIMENTAL_CONDITIONS = [
    PersonaGenerationCondition.MAX_ALL,
    PersonaGenerationCondition.MIN_E_MAX_REST,
]


class PersonaGenerator:
    """
    Unified persona generator supporting both simple and complex workflows.

    Features:
    - Multiple LLM providers and models
    - Parallel processing with configurable workers
    - Borderline personality condition generation
    - Flexible database operations
    - Comprehensive error handling and logging
    - Fixed OpenAI JSON mode requirements
    """

    def __init__(
        self,
        db_handler: Optional[DatabaseHandler] = None,
        population_handler: Optional[Population] = None,
        ref_population: str = "spain826",
        logger: Optional[logging.Logger] = None,
        config_manager: Optional[ConfigManager] = None,
        expected_schema: Optional[Dict[str, str]] = None,
        schema: Optional[str] = None,
    ):
        """
        Initialize persona generator.

        Args:
            db_handler: Database handler for persistence
            ref_population: Reference population identifier
            logger: Logger instance
            config_manager: Configuration manager for API keys
            expected_schema: JSON schema for persona validation
            schema: Optional explicit target schema for DB operations
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config_manager = config_manager or ConfigManager(logger=self.logger)
        self.db_handler = db_handler
        # Resolve schema once and reuse. Prefer explicit arg, else config manager.
        try:
            cfg_schema = (
                self.config_manager.schema_config.target_schema
                if self.config_manager and self.config_manager.schema_config
                else None
            )
        except Exception:
            cfg_schema = None
        self.schema = schema or cfg_schema
        self.population_handler = population_handler or (
            Population(db_handler=self.db_handler, logger=self.logger) if self.db_handler else None
        )
        self.ref_population = ref_population

        # Default JSON schema for persona generation
        default_schema = {
            "name": "string",
            "age": "integer",
            "gender": "string",
            "sexual_orientation": "string",
            "race": "string",
            "ethnicity": "string",
            "religious_belief": "string",
            "occupation": "string",
            "political_orientation": "string",
            "location": "string",
            "description": "string",
        }
        self.expected_schema = json.dumps(
            expected_schema if expected_schema else default_schema, indent=2
        )

    def _validate_personality_id(self, personality_id: Optional[Union[int, str]]) -> Optional[int]:
        """
        Validate and convert personality_id to proper type for database consistency.

        Args:
            personality_id: Input personality ID (int, str, or None)

        Returns:
            Validated integer personality_id or None

        Raises:
            TypeError: If personality_id cannot be converted to int
            ValueError: If string personality_id is not numeric
        """
        if personality_id is None:
            return None

        if isinstance(personality_id, (int, np.integer)):
            # Accept both Python int and numpy integer types
            result = int(personality_id)  # Convert numpy types to Python int
            self.logger.debug(f"personality_id is integer type: {personality_id} → {result}")
            return result

        if isinstance(personality_id, str):
            # Check if string is numeric
            try:
                converted_id = int(personality_id)
                self.logger.warning(
                    f"Converting string personality_id '{personality_id}' to int {converted_id}. "
                    f"Consider using int directly for database consistency."
                )
                return converted_id
            except ValueError as e:
                error_msg = (
                    f"personality_id must be numeric. Got string '{personality_id}' which cannot be "
                    f"converted to int. Database field 'personality_id' expects Integer type."
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg) from e

        # Handle unexpected types
        error_msg = (
            f"personality_id must be int, numeric string, or None. "
            f"Got {type(personality_id).__name__}: {personality_id}. "
            f"Database field 'personality_id' expects Integer type."
        )
        self.logger.error(error_msg)
        raise TypeError(error_msg)

    def _validate_generation_parameters(
        self,
        questionnaire_df: pd.DataFrame,
        model: ModelID,
        personality_id: Optional[Union[int, str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Dict:
        """
        Comprehensive validation of all parameters before API calls.

        Args:
            questionnaire_df: DataFrame with questionnaire responses
            model: Model to use for generation
            personality_id: Optional personality identifier
            temperature: LLM temperature (0-1)
            max_tokens: Maximum tokens for response

        Returns:
            Dict with validated parameters

        Raises:
            ValueError: If any parameter is invalid
            TypeError: If parameter types are incorrect
        """
        errors = []

        # Validate questionnaire DataFrame
        if not isinstance(questionnaire_df, pd.DataFrame):
            errors.append(
                f"questionnaire_df must be pandas DataFrame, "
                f"got {type(questionnaire_df).__name__}"
            )
        elif questionnaire_df.empty:
            errors.append("questionnaire_df cannot be empty")
        else:
            required_cols = ["question", "answer"]
            missing_cols = [col for col in required_cols if col not in questionnaire_df.columns]
            if missing_cols:
                errors.append(f"questionnaire_df missing required columns: {missing_cols}")

        # Validate model
        if not isinstance(model, ModelID):
            errors.append(f"model must be ModelID enum, got {type(model).__name__}")

        # Validate personality_id (this will also handle type conversion)
        try:
            validated_personality_id = self._validate_personality_id(personality_id)
        except (TypeError, ValueError) as e:
            errors.append(f"personality_id validation failed: {str(e)}")
            validated_personality_id = None

        # Validate temperature
        if not isinstance(temperature, (int, float)):
            errors.append(f"temperature must be numeric, got {type(temperature).__name__}")
        elif not 0 <= temperature <= 2:
            errors.append(f"temperature must be between 0-2, got {temperature}")

        # Validate max_tokens
        if not isinstance(max_tokens, int):
            errors.append(f"max_tokens must be int, got {type(max_tokens).__name__}")
        elif max_tokens <= 0:
            errors.append(f"max_tokens must be positive, got {max_tokens}")

        # If we have errors, log them and raise
        if errors:
            error_msg = "Parameter validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Return validated parameters
        return {
            "questionnaire_df": questionnaire_df,
            "model": model,
            "personality_id": validated_personality_id,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

    def generate_single_persona(
        self,
        questionnaire_df: pd.DataFrame,
        model: ModelID,
        personality_id: Optional[Union[int, str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        custom_schema: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate a single persona from questionnaire data.

        Args:
            questionnaire_df: DataFrame with questionnaire responses
            model: Model to use for generation
            personality_id: Optional personality identifier
            temperature: LLM temperature (0-1)
            max_tokens: Maximum tokens for response
            custom_schema: Optional custom JSON schema

        Returns:
            Dict containing generated persona data and metadata
        """
        try:
            # Comprehensive parameter validation BEFORE expensive API calls
            validated_params = self._validate_generation_parameters(
                questionnaire_df, model, personality_id, temperature, max_tokens
            )
            self.logger.info(
                f"Parameters validated for model {model.value}, "
                f"personality_id: {validated_params['personality_id']}"
            )

            # Use validated parameters
            questionnaire_df = validated_params["questionnaire_df"]
            model = validated_params["model"]
            validated_personality_id = validated_params["personality_id"]
            temperature = validated_params["temperature"]
            max_tokens = validated_params["max_tokens"]

            # Prepare questionnaire data for prompt
            persona_data = self._prepare_persona_data(questionnaire_df)

            # Get system role and schema
            schema = custom_schema or json.loads(self.expected_schema)
            system_role = self.get_generate_persona_system_role(model)

            # Get model wrapper
            model_wrapper = get_model_wrapper(
                model.value, self.logger, config_manager=self.config_manager
            )

            # Create experiment config
            experiment_config = {
                "model": get_models_config(model.value)["MODEL_ID"],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Generate persona
            self.logger.info(f"Generating persona with {model.value}")

            request, response, parsed_json = model_wrapper.generate_conversation(
                user_text_input=persona_data,
                my_experiment=experiment_config,
                system_role=system_role,
                expected_schema=schema,
            )

            # Check for errors
            if isinstance(parsed_json, dict) and "error" in parsed_json:
                raise ValueError(f"Model returned error: {parsed_json.get('error')}")

            # Return complete result
            result = {
                "personality_id": validated_personality_id,
                "model": model.value,
                "generated_persona": parsed_json,
                "raw_response": response,
                "request_data": request,
                "timestamp": datetime.now().isoformat(),
                "questionnaire_summary": {
                    "total_questions": len(questionnaire_df),
                    "categories": questionnaire_df.get("category", pd.Series()).nunique(),
                    "answered_questions": len(
                        questionnaire_df[questionnaire_df.get("answer", False) == True]
                    ),
                },
            }

            self.logger.info(
                f"Successfully generated persona: {parsed_json.get('name', 'Unknown')}"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error generating persona: {e}")
            raise

    def generate_personas(
        self,
        models: List[ModelID],
        population_df: pd.DataFrame,
        new_population_table: str,
        exp_df: pd.DataFrame,
        max_workers: int = 5,
        start_repetitions: int = 1,
        repetitions: int = 1,
        allow_experimental: bool = False,
        schema: Optional[str] = None,
    ) -> None:
        """
        Generate personas in parallel for multiple models.

        Args:
            models: List of models to use
            population_df: Population dataframe with reference data
            new_population_table: Name of the table to save results
            exp_df: Experiment dataframe with questionnaire data
            max_workers: Maximum number of parallel workers
            start_repetitions: Starting repetition number
            repetitions: Number of repetitions to generate
            allow_experimental: Permit experimental models
        """
        if not self.population_handler:
            raise ValueError("Population handler required for batch generation")

        if not allow_experimental:
            unsupported = [m.value for m in models if m in EXPERIMENTAL_MODELS]
            if unsupported:
                msg = "Experimental models require allow_experimental=True: " + ", ".join(
                    unsupported
                )
                raise ValueError(msg)

        # Resolve schema to use for DB I/O
        use_schema = schema or self.schema

        for model in models:
            self.logger.info("Processing with model: %s", model.value)
            population_name = f"generated_{model.value}_{self.ref_population}"

            # Collect personalities to process
            personalities_to_process = pd.DataFrame(columns=["personality_id", "repeated"])

            for personality_id in population_df["persona_id"].unique():
                for repeated in range(start_repetitions, start_repetitions + repetitions):
                    # Check if personality already exists
                    if self.population_handler.check_existing_personality(
                        population_name,
                        personality_id,
                        repeated,
                        new_population_table,
                        schema=use_schema,
                    ):
                        self.logger.info(
                            "Personality %s rep %s exists for %s (skip)",
                            personality_id,
                            repeated,
                            population_name,
                        )
                        continue

                    # Extract persona data
                    personality_id_df = exp_df[exp_df["personality_id"] == personality_id]
                    if personality_id_df.empty:
                        continue

                    personalities_to_process = pd.concat(
                        [
                            personalities_to_process,
                            pd.DataFrame(
                                {
                                    "personality_id": [personality_id],
                                    "repeated": [repeated],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

            # Log processing summary
            self.logger.info(
                "Found %d personalities for model %s", len(personalities_to_process), model.value
            )

            if personalities_to_process.empty:
                self.logger.info("No personalities for model %s (skip)", model.value)
                continue

            # Process personas in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for _, personality_row in personalities_to_process.iterrows():
                    # Get personality data
                    personality_id_df = exp_df[
                        exp_df["personality_id"] == personality_row.personality_id
                    ]

                    # Get original description
                    original_description = None
                    if not population_df.empty and "description" in population_df.columns:
                        descriptions = population_df.loc[
                            population_df["persona_id"] == personality_row.personality_id,
                            "description",
                        ].values
                        if len(descriptions) > 0:
                            original_description = descriptions[0]

                    futures.append(
                        executor.submit(
                            self._process_persona,
                            personality_row.personality_id,
                            personality_id_df,
                            model,
                            original_description,
                            new_population_table,
                            population_name,
                            personality_row.repeated,
                            use_schema,
                        )
                    )

                # Process results as they complete
                for future in futures:
                    try:
                        result = future.result()
                        if result is not None:
                            self.logger.info(
                                "Successfully processed persona %s",
                                result["ref_personality_id"].values[0],
                            )
                    except Exception:
                        self.logger.exception("Error processing persona future")

    def generate_borderline_personas(
        self,
        model: Union[ModelID, List[ModelID]],
        borderline_list: List[PersonaGenerationCondition],
        personas_list: List[str],
        population_df: pd.DataFrame,
        new_population_table: str,
        exp_df: pd.DataFrame,
        max_workers: int = 5,
        start_repetitions: int = 1,
        repetitions: int = 5,
        allow_experimental: bool = False,
        schema: Optional[str] = None,
    ) -> None:
        """
        Generate borderline personality condition personas.

        Args:
            model: Model(s) to use for generation
            borderline_list: List of borderline conditions to apply
            personas_list: List of personality IDs to process
            population_df: Population dataframe with reference data
            new_population_table: Name of the table to save results
            exp_df: Experiment dataframe with questionnaire data
            max_workers: Maximum number of parallel workers
            start_repetitions: Starting repetition number
            repetitions: Number of repetitions to generate
            allow_experimental: Permit experimental models/conditions
        """
        if not self.population_handler:
            raise ValueError("Population handler required for borderline generation")

        if not allow_experimental:
            bad_conditions = [c.value for c in borderline_list if c in EXPERIMENTAL_CONDITIONS]
            if bad_conditions:
                msg = "Experimental conditions require allow_experimental=True: " + ", ".join(
                    bad_conditions
                )
                raise ValueError(msg)

            model_list = [model] if isinstance(model, ModelID) else model
            bad_models = [m.value for m in model_list if m in EXPERIMENTAL_MODELS]
            if bad_models:
                msg = "Experimental models require allow_experimental=True: " + ", ".join(
                    bad_models
                )
                raise ValueError(msg)

        models = [model] if isinstance(model, ModelID) else model
        # Resolve schema to use for DB I/O
        use_schema = schema or self.schema

        # Ensure experiment_id exists (some pipelines rely on it)
        if "experiment_id" not in exp_df.columns:
            exp_df = exp_df.copy()
            exp_df["experiment_id"] = 0

        for current_model in models:
            for borderline_population in borderline_list:
                self.logger.info(
                    "Processing borderline population=%s model=%s",
                    borderline_population.value,
                    current_model.value,
                )

                # Create borderline dataframe based on configuration
                exp_df_borderline = None
                population_name = None

                if borderline_population == PersonaGenerationCondition.MIN_L:
                    exp_df_borderline = self.compute_borderline(exp_df, "L", True)
                    population_name = f"borderline_minL_{current_model.value}"
                elif borderline_population == PersonaGenerationCondition.MAX_N:
                    exp_df_borderline = self.compute_borderline(exp_df, "N", False)
                    population_name = f"borderline_maxN_{current_model.value}"
                elif borderline_population == PersonaGenerationCondition.MAX_P:
                    exp_df_borderline = self.compute_borderline(exp_df, "P", False)
                    population_name = f"borderline_maxP_{current_model.value}"
                elif borderline_population == PersonaGenerationCondition.MAX_ALL:
                    exp_df_borderline = exp_df.copy()
                    for category in ["E", "N", "P", "L"]:
                        exp_df_borderline = self.compute_borderline(
                            exp_df_borderline, category, False
                        )
                    population_name = f"borderline_maxAll_{current_model.value}"
                elif borderline_population == PersonaGenerationCondition.MIN_E_MAX_REST:
                    exp_df_borderline = self.compute_borderline(exp_df, "E", True)
                    for category in ["N", "P", "L"]:
                        exp_df_borderline = self.compute_borderline(
                            exp_df_borderline, category, False
                        )
                    population_name = f"borderline_minE_maxRest_{current_model.value}"
                else:
                    raise ValueError(f"Invalid borderline population: {borderline_population}")

                if exp_df_borderline is not None:
                    # Ensure experiment_id exists in borderline DF as well
                    if "experiment_id" not in exp_df_borderline.columns:
                        exp_df_borderline = exp_df_borderline.copy()
                        exp_df_borderline["experiment_id"] = 0
                    # Store personalities that will be processed
                    personalities_to_process = pd.DataFrame(columns=["personality_id", "repeated"])

                    # Collect personalities to process
                    for personality_id in personas_list:
                        for repeated in range(start_repetitions, start_repetitions + repetitions):
                            # Check if personality already exists
                            if self.population_handler.check_existing_personality(
                                population_name,
                                personality_id,
                                repeated,
                                new_population_table,
                                schema=use_schema,
                            ):
                                self.logger.info(
                                    "Personality %s rep %s exists in %s (skip)",
                                    personality_id,
                                    repeated,
                                    population_name,
                                )
                                continue

                            # Extract persona data (robust against type mismatches)
                            pid_int = None
                            try:
                                pid_int = int(personality_id)
                            except Exception:
                                pid_int = None

                            if pid_int is not None:
                                personality_id_df = exp_df_borderline[
                                    exp_df_borderline["personality_id"] == pid_int
                                ]
                            else:
                                pid_str = str(personality_id)
                                personality_id_df = exp_df_borderline[
                                    exp_df_borderline["personality_id"].astype(str) == pid_str
                                ]

                            if personality_id_df.empty:
                                self.logger.info(
                                    "No data found for personality %s (after coercion)",
                                    personality_id,
                                )
                                continue

                            # Store the CONVERTED id (integer if possible) for later filtering
                            normalized_id = pid_int if pid_int is not None else personality_id

                            personalities_to_process = pd.concat(
                                [
                                    personalities_to_process,
                                    pd.DataFrame(
                                        {
                                            "personality_id": [normalized_id],
                                            "repeated": [repeated],
                                        }
                                    ),
                                ],
                                ignore_index=True,
                            )

                    # Log processing summary
                    self.logger.info(
                        "Found %d personalities model=%s border=%s",
                        len(personalities_to_process),
                        current_model.value,
                        borderline_population.value,
                    )

                    if personalities_to_process.empty:
                        self.logger.info(
                            "No personalities model=%s border=%s (skip)",
                            current_model.value,
                            borderline_population.value,
                        )
                        continue

                    # Process personas in parallel
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = []
                        success_count = 0
                        total_count = len(personalities_to_process)
                        for _, personality_row in personalities_to_process.iterrows():
                            # Get personality data
                            personality_id_df = exp_df_borderline[
                                exp_df_borderline["personality_id"]
                                == personality_row.personality_id
                            ]

                            # Get original description
                            original_description = None
                            if not population_df.empty and "description" in population_df.columns:
                                descriptions = population_df.loc[
                                    population_df["persona_id"] == personality_row.personality_id,
                                    "description",
                                ].values
                                if len(descriptions) > 0:
                                    original_description = descriptions[0]

                            futures.append(
                                executor.submit(
                                    self._process_persona,
                                    personality_row.personality_id,
                                    personality_id_df,
                                    current_model,
                                    original_description,
                                    new_population_table,
                                    population_name,
                                    personality_row.repeated,
                                    use_schema,
                                )
                            )

                        # Process results as they complete
                        for future in futures:
                            try:
                                result = future.result()
                                if result is not None:
                                    self.logger.info(
                                        "Processed persona %s rep %s",
                                        result["ref_personality_id"].values[0],
                                        result["repetitions"].values[0],
                                    )
                                    success_count += 1
                            except Exception:
                                self.logger.exception("Error processing borderline persona future")

                        self.logger.info(
                            "Borderline generation summary: %d/%d saved (population=%s, model=%s)",
                            success_count,
                            total_count,
                            population_name,
                            current_model.value,
                        )

    def save_persona_to_db(
        self,
        persona_result: Dict,
        schema: str,
        table: str = "personas",
        additional_fields: Optional[Dict] = None,
    ) -> None:
        """
        Save generated persona to database with flexible schema/table.

        Args:
            persona_result: Result from generate_single_persona()
            schema: Database schema name
            table: Table name, default to personas
            additional_fields: Optional additional fields to save
        """
        if not self.db_handler:
            raise ValueError("Database handler required for saving")

        def _validate_identifier(identifier: str, kind: str) -> str:
            if (
                not identifier
                or not identifier.replace("_", "").isalnum()
                or not (identifier[0].isalpha() or identifier[0] == "_")
            ):
                raise ValueError(f"Invalid {kind} name: {identifier}")
            return identifier

        try:
            # Extract persona data
            persona_data = persona_result["generated_persona"]
            personality_id = persona_result.get("personality_id")

            # Log personality_id type for debugging
            self.logger.debug(
                f"Saving persona with personality_id: {personality_id} "
                f"(type: {type(personality_id).__name__})"
            )

            # Create record for database
            record = {
                "personality_id": personality_id,
                "model": persona_result["model"],
                "json_response": json.dumps(persona_data),
                "name": persona_data.get("name", ""),
                "age": persona_data.get("age", 0),
                "gender": persona_data.get("gender", ""),
                "sexual_orientation": persona_data.get("sexual_orientation", ""),
                "race": persona_data.get("race", ""),
                "ethnicity": persona_data.get("ethnicity", ""),
                "religious_belief": persona_data.get("religious_belief", ""),
                "occupation": persona_data.get("occupation", ""),
                "political_orientation": persona_data.get("political_orientation", ""),
                "location": persona_data.get("location", ""),
                "description": persona_data.get("description", ""),
                "created_at": datetime.now(),
            }

            # Add any additional fields
            if additional_fields:
                record.update(additional_fields)

            # Create insert SQL
            columns = []
            for col in record.keys():
                columns.append(_validate_identifier(col, "column"))
            placeholders = [f":{col}" for col in columns]

            safe_schema = _validate_identifier(schema, "schema")
            safe_table = _validate_identifier(table, "table")

            insert_sql = text(
                f"INSERT INTO {safe_schema}.{safe_table} "
                f"({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
            )

            # Execute insert
            with self.db_handler.session() as session:
                session.execute(insert_sql, record)
                session.commit()

            self.logger.info(f"Saved persona to {safe_schema}.{safe_table}")

        except Exception as e:
            self.logger.error(f"Error saving persona to database: {e}")
            raise

    @staticmethod
    def create_borderline_questionnaire(
        questionnaire_df: pd.DataFrame, condition: PersonaGenerationCondition
    ) -> pd.DataFrame:
        """
        Create borderline questionnaire by modifying specific categories.

        Args:
            questionnaire_df: Original questionnaire DataFrame
            condition: Borderline condition to apply

        Returns:
            Modified questionnaire DataFrame
        """
        df = questionnaire_df.copy()

        # Ensure answer column is boolean
        if "answer" in df.columns and df["answer"].dtype != bool:
            df["answer"] = df["answer"].astype(bool)

        # Apply borderline modifications
        if condition == PersonaGenerationCondition.MAX_N:
            # Maximum Neuroticism: set all N category answers to match key (direct)
            mask = df["category"] == "N"
            df.loc[mask, "answer"] = df.loc[mask, "key"]

        elif condition == PersonaGenerationCondition.MAX_P:
            # Maximum Psychoticism: set all P category answers to match key (direct)
            mask = df["category"] == "P"
            df.loc[mask, "answer"] = df.loc[mask, "key"]

        elif condition == PersonaGenerationCondition.MIN_L:
            # Minimum Lie Scale: set all L category answers to opposite of key (inverse)
            mask = df["category"] == "L"
            df.loc[mask, "answer"] = ~df.loc[mask, "key"]

        else:
            raise ValueError(f"Unsupported borderline condition: {condition}")

        return df

    @staticmethod
    def compute_borderline(
        baseline_questionnaire: pd.DataFrame,
        category: str,
        apply_inverse: bool,
    ) -> pd.DataFrame:
        """
        Compute borderline questionnaire by modifying specific category.

        Args:
            baseline_questionnaire: Original questionnaire DataFrame
            category: Category to modify (E, N, P, L)
            apply_inverse: Whether to apply inverse logic

        Returns:
            Modified questionnaire DataFrame
        """
        df = baseline_questionnaire.copy()

        # Find rows for the specified category
        category_mask = df["category"] == category

        if apply_inverse:
            # Set answers to opposite of key
            df.loc[category_mask, "answer"] = ~df.loc[category_mask, "key"]
        else:
            # Set answers to match key
            df.loc[category_mask, "answer"] = df.loc[category_mask, "key"]

        return df

    def _prepare_persona_data(self, questionnaire_df: pd.DataFrame) -> str:
        """
        Convert questionnaire DataFrame to prompt text.

        Args:
            questionnaire_df: DataFrame with questionnaire responses

        Returns:
            Formatted prompt string
        """
        # Ensure we have required columns
        required_cols = ["question", "answer"]
        for col in required_cols:
            if col not in questionnaire_df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Sort by question_number if available
        if "question_number" in questionnaire_df.columns:
            df_sorted = questionnaire_df.sort_values("question_number")
        else:
            df_sorted = questionnaire_df

        # Create CSV-style data for the prompt
        csv_data = df_sorted[["question", "answer"]].to_csv(index=False)

        prompt_parts = [
            "The table below provides personality questionnaire data. Each row represents a question and the answer given by the person:",
            "- question: The text of the question asked",
            "- answer: The specific response (True/False or text)",
            "",
            csv_data.strip(),
            "",
            "Use this data to create a realistic persona whose personality traits align with the answers provided.",
        ]

        return "\n".join(prompt_parts)

    def _process_persona(
        self,
        personality_id: str,
        personality_id_df: pd.DataFrame,
        model: ModelID,
        original_description: Optional[str],
        new_population_table: str,
        population_name: str,
        repeated: int = 1,
        schema: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Process a single persona with full error handling.

        Args:
            personality_id: Personality ID
            personality_id_df: DataFrame with questionnaire data
            model: Model to use
            original_description: Optional original description
            new_population_table: Table to save results
            population_name: Population name for grouping
            repeated: Repetition number

        Returns:
            Generated persona DataFrame or None if failed
        """
        try:
            self.logger.info(
                "Processing persona %s for population %s", personality_id, population_name
            )

            # Prepare data and generate persona
            persona_data = self._prepare_persona_data(personality_id_df)

            # Get model wrapper and generate response
            model_wrapper = get_model_wrapper(
                model.value, self.logger, config_manager=self.config_manager
            )

            # Create experiment config
            my_experiment = {"model": get_models_config(model.value)["MODEL_ID"]}

            # Generate conversation and get response
            request, response, parsed_json = model_wrapper.generate_conversation(
                user_text_input=persona_data,
                my_experiment=my_experiment,
                system_role=self.get_generate_persona_system_role(model),
                expected_schema=json.loads(self.expected_schema),
            )

            # Check for errors
            if isinstance(parsed_json, dict) and "error" in parsed_json:
                self.logger.error(
                    "Error in model response for personality %s: %s",
                    personality_id,
                    parsed_json.get("error"),
                )
                return None

            # Handle different ethnicity field names for backwards compatibility
            ethnicity = parsed_json.get(
                "ethnicnicity",
                parsed_json.get("ethnicity", parsed_json.get("ethnic", "")),
            )

            # Extract experiment_id if present; use 0 as fallback (persona generation context)
            if not personality_id_df.empty and "experiment_id" in personality_id_df.columns:
                reference_exp_id = personality_id_df["experiment_id"].unique()[0]
            else:
                reference_exp_id = 0

            # Extract model-generated fields
            name_val = parsed_json.get("name", "")
            age_val = parsed_json.get("age", 0)
            gender_val = parsed_json.get("gender", "")
            race_val = parsed_json.get("race", "")
            sexual_val = parsed_json.get("sexual_orientation", "")
            rel_val = parsed_json.get("religious_belief", "")
            occup_val = parsed_json.get("occupation", "")
            pol_val = parsed_json.get("political_orientation", "")
            loc_val = parsed_json.get("location", "")
            desc_val = parsed_json.get("description", "")

            # Compute deriveds/prints
            try:
                word_count = int(len(str(desc_val).split())) if desc_val else 0
            except Exception:
                word_count = 0
            model_print = model.value
            population_print = population_name

            # Create dataframe with results
            # Convention: model outputs stored in *_original fields; mapped fields
            # (without _original) may be filled later by mapping pipelines.
            generated_persona = pd.DataFrame(
                {
                    "ref_population": [self.ref_population],
                    "ref_personality_id": [personality_id],
                    "ref_experiment_id": [reference_exp_id],
                    "ref_original_description": [original_description],
                    "population": [population_name],
                    "model": [model.value],
                    "json_answer": [json.dumps(parsed_json)],
                    "name": [name_val],
                    "age": [age_val],
                    # Original (as produced by the model)
                    "gender_original": [gender_val],
                    "race_original": [race_val],
                    "sexual_orientation_original": [sexual_val],
                    "religious_belief_original": [rel_val],
                    "occupation_original": [occup_val],
                    "political_orientation_original": [pol_val],
                    "location_original": [loc_val],
                    # Mapped fields (to be standardized later); leave blank for now
                    "gender": [""],
                    "race": [""],
                    "sexual_orientation": [""],
                    "religious_belief": [""],
                    "occupation": [""],
                    "political_orientation": [""],
                    "location": [""],
                    # Keep raw ethnicity directly (no original/mapped pair specified in schema)
                    "ethnicity": [ethnicity],
                    "description": [desc_val],
                    "word_count_description": [word_count],
                    "repetitions": [repeated],
                    "model_print": [model_print],
                    "population_print": [population_print],
                }
            )

            # Save to database if population handler available
            if self.population_handler:
                self.population_handler.save_generated_persona(
                    generated_persona,
                    new_population_table,
                    schema=schema or self.schema,
                )

            return generated_persona

        except Exception:
            self.logger.exception("Unhandled error processing personality_id=%s", personality_id)
            return None

    def get_generate_persona_system_role(self, model: ModelID) -> str:
        """
        Get system role based on the model with FIXED JSON mode support.

        Args:
            model: Model identifier

        Returns:
            System role string with proper JSON instructions
        """
        compact_instr = (
            "Write a coherent multi-paragraph bio from answers. "
            "No question text. Reflect behaviors and preferences. "
            "Clear concise language. Output JSON only with schema."
        )

        # CRITICAL: JSON instruction required for OpenAI JSON mode
        json_rule = "Return only JSON matching schema; no extra text or formatting."

        # ALL models now include JSON instruction to fix OpenAI JSON mode bug
        role_system_dict = {
            ModelID.GPT35: f"{compact_instr}\n{json_rule}\nSchema:\n{self.expected_schema}",
            ModelID.GPT4O: f"{compact_instr}\n{json_rule}\nSchema:\n{self.expected_schema}",
            ModelID.LLAMA3_23B: f"{compact_instr}\n{json_rule}\nSchema:\n{self.expected_schema}",
            ModelID.CLAUDE35_SONNET: f"{compact_instr}\n{json_rule}\nSchema:\n{self.expected_schema}",
            ModelID.LLAMA3_170B: f"{compact_instr}\n{json_rule}\nSchema:\n{self.expected_schema}",
        }

        try:
            return role_system_dict[model]
        except KeyError as exc:
            raise ValueError(f"Model {model.value} not found in system role dictionary") from exc


# Convenience functions for common usage patterns
def create_persona_generator(
    config_manager: Optional[ConfigManager] = None,
    db_handler: Optional[DatabaseHandler] = None,
    logger: Optional[logging.Logger] = None,
) -> PersonaGenerator:
    """
    Convenience function to create a PersonaGenerator instance.

    Args:
        config_manager: Configuration manager for API keys
        db_handler: Database handler for persistence
        logger: Logger instance

    Returns:
        Configured PersonaGenerator instance
    """
    return PersonaGenerator(
        config_manager=config_manager,
        db_handler=db_handler,
        logger=logger or logging.getLogger("persona_generator"),
    )
