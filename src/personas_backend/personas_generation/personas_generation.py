import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import pandas as pd  # type: ignore
from personas_backend.core.enums import ModelID, PersonaGenerationCondition
from personas_backend.db.db_handler import DatabaseHandler  # type: ignore
from personas_backend.models_providers.models_config import (
    get_models_config,  # type: ignore
)
from personas_backend.models_providers.models_utils import (
    get_model_wrapper,  # type: ignore
)
from personas_backend.population.population import Population  # type: ignore
from personas_backend.utils.config import ConfigManager  # type: ignore

EXPERIMENTAL_MODELS = {
    ModelID.GPT4O_OLD,
    ModelID.GPT4_TURBO,
    ModelID.LLAMA3_170B,
}

EXPERIMENTAL_CONDITIONS = {
    PersonaGenerationCondition.MIN_L,
    PersonaGenerationCondition.MAX_ALL,
    PersonaGenerationCondition.MIN_E_MAX_REST,
}


class PersonaGenerator:
    def __init__(
        self,
        db_handler: DatabaseHandler,
        population_handler: Optional["Population"] = None,
        ref_population: str = "spain826",
        logger: Optional[logging.Logger] = None,
        expected_schema: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize persona generator.

        Args:
            db_handler: Database handler
            population_handler: Optional population handler instance
            ref_population: Reference population name
            logger: Optional logger instance
            expected_schema: Optional JSON schema string
        """
        # # create a self class to manually test
        # class self:
        #     pass
        # self._prepare_persona_data = _prepare_persona_data
        # self.get_generate_persona_system_role = get_generate_persona_system_role

        self.logger = logger or logging.getLogger(__name__)
        self.db_handler = db_handler
        self.population_handler = population_handler or Population(
            db_handler=self.db_handler, logger=self.logger
        )
        self.config_manager = ConfigManager(logger=self.logger)
        self.ref_population = ref_population
        self.expected_schema = expected_schema

    def _prepare_persona_data(self, personality_id_df: pd.DataFrame) -> str:
        """
        Prepare persona data from questionnaire answers.

        Args:
            personality_df (pd.DataFrame): Dataframe with questionnaire answers

        Returns:
            str: Formatted persona data string
        """
        ordered_df = personality_id_df.sort_values(by="question_number")
        csv_string = ordered_df[["question", "answer"]].to_csv(index=False)

        parts = [
            "The table below provides the input data. Each row represents a question and the answer given by the person under consideration:",
            "- question: The text of the question formulated to the person.",
            "- answer: The specific response expected from the persona.",
            csv_string.strip(),
            "Use this data to ensure the persona's personality traits align with the answers provided.",
        ]
        return "\n".join(parts)

    def _create_messages(self, persona_data: str, model: ModelID) -> List[Dict[str, str]]:
        """
        Create message structure for API call.

        Args:
            persona_data (str): Formatted persona data
            model (ModelID): Model identifier

        Returns:
            List[Dict[str, str]]: Message structure for API
        """
        return [
            {
                "role": "system",
                "content": self.get_generate_persona_system_role(model),
            },
            {"role": "user", "content": persona_data},
        ]

    def _process_persona(
        self,
        personality_id: str,
        personality_id_df: pd.DataFrame,
        model: ModelID,
        original_description: Optional[str],
        new_population_table: str,
        population_name: str,
        repeated: int = 1,
    ) -> Optional[pd.DataFrame]:
        """
        Process a single persona.

        Args:
            personality_id (str): Personality ID
            exp_df (pd.DataFrame): Experiment dataframe
            model (str): Model name
            population_df (pd.DataFrame): Population dataframe
            new_population_table (str): Name of the table to save results
            population_name (str): Population name

        Returns:
            Optional[pd.DataFrame]: Persona dataframe or None if skipped/error
        """
        # personality_id=800
        # formerly: exp_df filter by personality_id
        # original_description = None

        try:
            self.logger.info(
                "Processing persona %s for population %s",
                personality_id,
                population_name,
            )

            # Prepare data and generate persona
            persona_data = self._prepare_persona_data(personality_id_df)
            # messages = self._create_messages(persona_data, model)

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
                expected_schema=self.expected_schema,
            )

            # check that parsed_json is not an error: {"error": str(e)}
            if isinstance(parsed_json, dict) and "error" in parsed_json:
                self.logger.error(
                    "Error in model response for personality %s: %s",
                    personality_id,
                    parsed_json.get("error"),
                )
                return None

            # Handle different ethnicity field names
            ethnicity = parsed_json.get(
                "ethnicnicity",
                parsed_json.get("ethnicity", parsed_json.get("ethnic", "")),
            )

            reference_exp_id = personality_id_df["experiment_id"].unique()[0]

            # Create dataframe with results
            generated_persona = pd.DataFrame(
                {
                    "ref_population": [self.ref_population],
                    "ref_personality_id": [personality_id],
                    "ref_experiment_id": [reference_exp_id],
                    "ref_original_description": [original_description],
                    "population": [population_name],
                    "model": [model.value],
                    "json_answer": [json.dumps(parsed_json)],
                    "name": [parsed_json.get("name", "")],
                    "age": [parsed_json.get("age", 0)],
                    "gender": [parsed_json.get("gender", "")],
                    "sexual_orientation": [parsed_json.get("sexual_orientation", "")],
                    "race": [parsed_json.get("race", "")],
                    "ethnicity": [ethnicity],
                    "religious_belief": [parsed_json.get("religious_belief", "")],
                    "occupation": [parsed_json.get("occupation", "")],
                    "political_orientation": [parsed_json.get("political_orientation", "")],
                    "location": [parsed_json.get("location", "")],
                    "description": [parsed_json.get("description", "")],
                    "repetitions": [repeated],
                }
            )

            self.population_handler.save_generated_persona(generated_persona, new_population_table)
            return generated_persona

        except Exception:  # pragma: no cover - defensive logging
            self.logger.exception("Unhandled error processing personality_id=%s", personality_id)
            return None

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
    ) -> None:
        """
        Generate personas in parallel.

        Args:
            models (List[str]): List of models to use
            personas_list (List[str]): List of personality IDs
            population_df (pd.DataFrame): Population dataframe
            new_population_table (str): Name of the table to save results
            exp_df (pd.DataFrame): Experiment dataframe
            max_workers (int): Maximum number of parallel workers
            allow_experimental (bool): Permit experimental models
        """
        if not allow_experimental:
            unsupported = [m.value for m in models if m in EXPERIMENTAL_MODELS]
            if unsupported:
                msg = "Experimental models require allow_experimental=True: " + ", ".join(
                    unsupported
                )
                raise ValueError(msg)

        for model in models:
            self.logger.info("Processing with model: %s", model.value)

            population_name = f"generated_{model.value}_{self.ref_population}"

            # Store personalities that will be processed
            personalities_to_process = pd.DataFrame(columns=["personality_id", "repeated"])

            # First check which personalities need processing
            for personality_id in population_df["persona_id"].unique():
                for repeated in range(start_repetitions, start_repetitions + repetitions):
                    # repetition loop (filtered for existing personalities)

                    # Check if personality already exists
                    if self.population_handler.check_existing_personality(
                        population_name,
                        personality_id,
                        repeated,
                        new_population_table,
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
                        self.logger.info(
                            "No data found for personality %s",
                            personality_id,
                        )
                        continue

                    # concat the personality_id and repeated to the dataframe
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

            # Log how many personalities will be processed
            self.logger.info(
                "Found %d personalities for model %s",
                len(personalities_to_process),
                model.value,
            )

            if personalities_to_process.empty:
                self.logger.info("No personalities for model %s (skip)", model.value)
                continue

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for _, personality_row in personalities_to_process.iterrows():
                    # Get personality data
                    personality_id_df = exp_df[
                        exp_df["personality_id"] == personality_row.personality_id
                    ]

                    # Get original description
                    if not population_df.empty:
                        original_description = population_df[
                            population_df["persona_id"] == personality_row.personality_id
                        ]["description"].values[0]
                    else:
                        original_description = None

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
                            # population_name = None
                    except Exception:  # pragma: no cover - defensive logging
                        self.logger.exception("Error processing persona future")

    @staticmethod
    def compute_borderline(
        baseline_questionnaire: pd.DataFrame,
        category: str,
        apply_inverse: bool,
    ) -> pd.DataFrame:
        """
        Compute borderline questionnaire based on category and inversion.

        Args:
            baseline_questionnaire (pd.DataFrame): Baseline questionnaire
            category (str): Category to modify
            apply_inverse (bool): Whether to apply inverse logic

        Returns:
            pd.DataFrame: Modified questionnaire dataframe
        """
        # Copy the synthetic questionnaires dataframe
        df = baseline_questionnaire.copy()

        # Ensure answer column is boolean early to avoid dtype assignment warnings when source
        # dataframe stores answers as ints (0/1) or objects.
        if "answer" in df.columns and df["answer"].dtype != bool:
            try:
                df["answer"] = df["answer"].astype(bool)
            except Exception:  # pragma: no cover - defensive
                # Fallback: coerce truthiness
                df["answer"] = df["answer"].apply(lambda v: bool(v))

        # Apply inverse/direct mapping of 'key' to 'answer' (pure Python bools)
        mask = df["category"] == category
        if apply_inverse:
            df.loc[mask, "answer"] = df.loc[mask, "key"].apply(lambda v: not bool(v))
        else:
            df.loc[mask, "answer"] = df.loc[mask, "key"].apply(lambda v: bool(v))

        # Normalize column strictly to bool dtype (no object coercion needed)
        df["answer"] = df["answer"].astype(bool)

        return df

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
    ) -> None:
        """
        Generate borderline personas with specific trait configurations.

        Args:
            model (Union[str, List[str]]): Model or list of models to use
            borderline_list (List[str]): List of borderline configurations
            personas_list (List[str]): List of personality IDs
            population_df (pd.DataFrame): Population dataframe
            new_population_table (str): Name of the table to save results
            exp_df (pd.DataFrame): Experiment dataframe
            max_workers (int): Maximum number of parallel workers
            start_repetitions (int): Starting repetition number
            repetitions (int): Number of repetitions to generate
            allow_experimental (bool): Permit experimental models/conditions
        """
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
                    msg = f"Invalid borderline population: {borderline_population}"
                    raise ValueError(msg)

                if exp_df_borderline is not None:
                    # Store personalities that will be processed
                    personalities_to_process = pd.DataFrame(columns=["personality_id", "repeated"])

                    # First check which personalities need processing
                    for personality_id in personas_list:
                        for repeated in range(start_repetitions, start_repetitions + repetitions):
                            # Check if personality already exists
                            if self.population_handler.check_existing_personality(
                                population_name,
                                personality_id,
                                repeated,
                                new_population_table,
                            ):
                                self.logger.info(
                                    ("Personality %s rep %s exists in %s " "(skip)"),
                                    personality_id,
                                    repeated,
                                    population_name,
                                )
                                continue

                            # Extract persona data
                            personality_id_df = exp_df_borderline[
                                exp_df_borderline["personality_id"] == personality_id
                            ]
                            if personality_id_df.empty:
                                self.logger.info(
                                    "No data found for personality %s",
                                    personality_id,
                                )
                                continue

                            # append personality_id + repetition
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

                    # Log how many personalities will be processed
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

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = []
                        for (
                            index,
                            personality_row,
                        ) in personalities_to_process.iterrows():
                            # Get personality data
                            personality_id_df = exp_df_borderline[
                                exp_df_borderline["personality_id"]
                                == personality_row.personality_id
                            ]

                            # Get original description
                            original_description = None
                            if not population_df.empty:
                                descriptions = population_df[
                                    population_df["persona_id"] == personality_row.personality_id
                                ]["description"].values
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
                            except Exception:  # pragma: no cover - defensive
                                self.logger.exception(
                                    "Error processing borderline persona " "future"
                                )

    def get_generate_persona_system_role(
        self,
        model: ModelID,
    ) -> str:
        """
        Get system role based on the model.

        Args:
            model (str): Model name

        Returns:
            str: System role
        """
        compact_instr = (
            "Write a coherent multi-paragraph bio from answers. "
            "No question text. Reflect behaviors and preferences. "
            "Clear concise language. Output JSON only with schema."
        )
        json_rule = "Return only JSON matching schema; no extra text or formatting."
        role_system_dict = {
            ModelID.GPT35: f"{compact_instr}\n{json_rule}\nSchema:\n{self.expected_schema}",
            ModelID.GPT4O: f"{compact_instr}\n{json_rule}\nSchema:\n{self.expected_schema}",
            ModelID.LLAMA3_23B: f"{compact_instr}\n{json_rule}\nSchema:\n{self.expected_schema}",
            ModelID.CLAUDE35_SONNET: f"{compact_instr}\n{json_rule}\nSchema:\n{self.expected_schema}",
            ModelID.LLAMA3_170B: f"{compact_instr}\n{json_rule}\nSchema:\n{self.expected_schema}",
        }
        try:
            return role_system_dict[model]
        except KeyError as exc:  # pragma: no cover - validation
            raise ValueError(f"Model {model.value} not found in system role dictionary") from exc
