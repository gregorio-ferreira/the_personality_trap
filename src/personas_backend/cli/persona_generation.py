"""Helper functions to bridge Typer CLI commands with the persona backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import pandas as pd
from personas_backend.core.enums import ModelID, PersonaGenerationCondition
from personas_backend.db.db_handler import DatabaseHandler
from personas_backend.db.experiments import ExperimentHandler
from personas_backend.persona_generator import (
    EXPERIMENTAL_CONDITIONS,
    EXPERIMENTAL_MODELS,
    PersonaGenerator,
)
from personas_backend.population.population import Population


@dataclass
class ArtifactPopulationCollector:
    """In-memory stand-in for :class:`Population` used when ``--no-db`` is active."""

    generated_personas: List[pd.DataFrame] = field(default_factory=list)

    def check_existing_personality(
        self,
        population: str,
        personality_id: str | int,
        repeated: int,
        new_population_table: str,
    ) -> bool:
        """Pretend no personas exist so every request is processed."""

        return False

    def save_generated_persona(self, persona_df: pd.DataFrame, population_table: str) -> None:
        """Capture generated personas instead of persisting them to PostgreSQL."""

        # ``PersonaGenerator`` hands us a one-row dataframe – keep the schema intact.
        self.generated_personas.append(persona_df.copy(deep=True))

    def as_dataframe(self) -> pd.DataFrame:
        """Return a concatenated dataframe of all generated personas."""

        if not self.generated_personas:
            return pd.DataFrame()
        return pd.concat(self.generated_personas, ignore_index=True)


def _normalise_token(value: str) -> str:
    return value.replace("-", "_").lower()


_MODEL_BY_TOKEN = {_normalise_token(member.value): member for member in ModelID}
_CONDITION_BY_TOKEN = {
    _normalise_token(member.value): member for member in PersonaGenerationCondition
}


def resolve_model_ids(
    models: Sequence[str],
    *,
    allow_experimental: bool,
) -> List[ModelID]:
    """Convert CLI-provided model slugs into :class:`ModelID` values."""

    resolved: list[ModelID] = []
    for raw in models:
        token = _normalise_token(raw)
        model = _MODEL_BY_TOKEN.get(token)
        if model is None:  # pragma: no cover - defensive, surfaced via Typer
            raise ValueError(f"Unsupported model '{raw}'")
        if not allow_experimental and model in EXPERIMENTAL_MODELS:
            raise ValueError("Experimental models require --experimental flag: " f"{model.value}")
        resolved.append(model)
    return resolved


def resolve_borderline_conditions(
    conditions: Sequence[str],
    *,
    allow_experimental: bool,
) -> List[PersonaGenerationCondition]:
    """Convert CLI inputs into :class:`PersonaGenerationCondition` values."""

    resolved: list[PersonaGenerationCondition] = []
    for raw in conditions:
        token = _normalise_token(raw)
        condition = _CONDITION_BY_TOKEN.get(token)
        if condition is None:  # pragma: no cover - defensive, surfaced via Typer
            raise ValueError(f"Unsupported borderline condition '{raw}'")
        if not allow_experimental and condition in EXPERIMENTAL_CONDITIONS:
            raise ValueError(
                "Experimental conditions require --experimental flag: " f"{condition.value}"
            )
        resolved.append(condition)
    return resolved


def _load_reference_frames(
    *,
    db_handler: DatabaseHandler,
    ref_population: str,
    population_schema: str,
    reference_table: str,
    reference_schema: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load reference population and questionnaire dataframes from PostgreSQL."""

    population_handler = Population(db_handler=db_handler)
    population_df = population_handler.get_ref_population_data(
        ref_population, schema=population_schema
    )

    experiments_handler = ExperimentHandler(db_handler=db_handler)
    questionnaires_df = experiments_handler.get_ref_experiment_data(
        reference_table, schema=reference_schema
    )
    return population_df, questionnaires_df


def generate_personas_to_database(
    *,
    run_id: str,
    models: Sequence[ModelID],
    ref_population: str,
    population_schema: str,
    reference_table: str,
    reference_schema: str,
    db_handler: DatabaseHandler,
    start_repetitions: int,
    repetitions: int,
    allow_experimental: bool,
) -> str:
    """Execute persona generation and persist results to PostgreSQL."""

    population_df, questionnaires_df = _load_reference_frames(
        db_handler=db_handler,
        ref_population=ref_population,
        population_schema=population_schema,
        reference_table=reference_table,
        reference_schema=reference_schema,
    )

    population_handler = Population(db_handler=db_handler)
    generator = PersonaGenerator(
        db_handler=db_handler,
        population_handler=population_handler,
        ref_population=ref_population,
    )

    table_name = f"{run_id}_generated_population"
    generator.generate_personas(
        models=list(models),
        population_df=population_df,
        new_population_table=table_name,
        exp_df=questionnaires_df,
        start_repetitions=start_repetitions,
        repetitions=repetitions,
        allow_experimental=allow_experimental,
    )
    return table_name


def generate_personas_to_artifacts(
    *,
    run_id: str,
    models: Sequence[ModelID],
    ref_population: str,
    population_df: pd.DataFrame,
    questionnaires_df: pd.DataFrame,
    start_repetitions: int,
    repetitions: int,
    allow_experimental: bool,
) -> Tuple[str, pd.DataFrame]:
    """Run persona generation offline and return results as a dataframe."""

    collector = ArtifactPopulationCollector()
    generator = PersonaGenerator(
        population_handler=collector,
        ref_population=ref_population,
    )

    table_name = f"{run_id}_generated_population"
    generator.generate_personas(
        models=list(models),
        population_df=population_df,
        new_population_table=table_name,
        exp_df=questionnaires_df,
        start_repetitions=start_repetitions,
        repetitions=repetitions,
        allow_experimental=allow_experimental,
    )
    return table_name, collector.as_dataframe()


def _borderline_persona_ids(population_df: pd.DataFrame) -> List[str]:
    persona_column = "persona_id" if "persona_id" in population_df.columns else "personality_id"
    ids = population_df.get(persona_column, pd.Series(dtype="object"))
    return [str(value) for value in ids.dropna().unique()]


def generate_borderline_to_database(
    *,
    run_id: str,
    models: Sequence[ModelID],
    conditions: Sequence[PersonaGenerationCondition],
    ref_population: str,
    population_schema: str,
    reference_table: str,
    reference_schema: str,
    db_handler: DatabaseHandler,
    start_repetitions: int,
    repetitions: int,
    allow_experimental: bool,
) -> str:
    """Execute borderline persona generation and persist to PostgreSQL."""

    population_df, questionnaires_df = _load_reference_frames(
        db_handler=db_handler,
        ref_population=ref_population,
        population_schema=population_schema,
        reference_table=reference_table,
        reference_schema=reference_schema,
    )

    personas_list = _borderline_persona_ids(population_df)
    population_handler = Population(db_handler=db_handler)
    generator = PersonaGenerator(
        db_handler=db_handler,
        population_handler=population_handler,
        ref_population=ref_population,
    )

    table_name = f"{run_id}_borderline_population"
    generator.generate_borderline_personas(
        model=list(models) if len(models) > 1 else models[0],
        borderline_list=list(conditions),
        personas_list=personas_list,
        population_df=population_df,
        new_population_table=table_name,
        exp_df=questionnaires_df,
        start_repetitions=start_repetitions,
        repetitions=repetitions,
        allow_experimental=allow_experimental,
    )
    return table_name


def generate_borderline_to_artifacts(
    *,
    run_id: str,
    models: Sequence[ModelID],
    conditions: Sequence[PersonaGenerationCondition],
    ref_population: str,
    population_df: pd.DataFrame,
    questionnaires_df: pd.DataFrame,
    start_repetitions: int,
    repetitions: int,
    allow_experimental: bool,
) -> Tuple[str, pd.DataFrame]:
    """Run borderline persona generation offline and return results."""

    collector = ArtifactPopulationCollector()
    generator = PersonaGenerator(
        population_handler=collector,
        ref_population=ref_population,
    )

    table_name = f"{run_id}_borderline_population"
    generator.generate_borderline_personas(
        model=list(models) if len(models) > 1 else models[0],
        borderline_list=list(conditions),
        personas_list=_borderline_persona_ids(population_df),
        population_df=population_df,
        new_population_table=table_name,
        exp_df=questionnaires_df,
        start_repetitions=start_repetitions,
        repetitions=repetitions,
        allow_experimental=allow_experimental,
    )
    return table_name, collector.as_dataframe()
