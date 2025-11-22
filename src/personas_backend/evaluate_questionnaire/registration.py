"""Services for registering questionnaire experiments for stored personas.

This module provides the high-level API for creating experiment groups and
registering individual experiments in the research database.

**Architecture Overview:**

The experiment system uses a two-level hierarchy:

1. **Experiment Group** (experiments_groups table):
   - Logical container defining a batch of related experiments
   - Stores: description, system_role, questionnaire JSON, LLM parameters
   - Created first, receives a unique experiments_group_id

2. **Individual Experiments** (experiments_list table):
   - One record per persona × repetition combination
   - Links to parent group via experiments_group_id (foreign key)
   - Each experiment runs independently during execution
   - Stores: questionnaire type, model, population, personality_id, results

**Workflow:**

```python
# Step 1: Register (creates group + experiments)
group_id, exp_ids = register_questionnaire_experiments(
    questionnaire="bigfive",
    model="gpt4o",
    populations=["generated_gpt4o_spain826"]
)

# Step 2: Execute (runs LLM calls, stores answers)
run_pending_experiments(experiments_group_ids=[group_id])
```

**Database Flow:**

1. Fetch personas from `personas` table (filtered by model + populations)
2. Create experiment group in `experiments_groups` (1 record)
3. For each persona:
   - Read persona.repetitions (default: 1)
   - Create N experiments in `experiments_list` (N = repetitions)
4. Return (group_id, list_of_experiment_ids)

**Key Features:**

* Fetches personas from database using SQLModel sessions
* Creates experiment groups via :class:`ExperimentGroupHandler`
* Registers experiments via :class:`ExperimentHandler`
* Automatically handles persona repetitions for statistical reliability
* Returns identifiers for downstream orchestration

**Entry Point:**

:func:`register_questionnaire_experiments` - Main API for experiment creation
"""

from __future__ import annotations

import logging
import subprocess
from typing import Callable, ContextManager, Iterable, List, Sequence, Tuple

from personas_backend.db.experiment_groups import ExperimentGroupHandler
from personas_backend.db.experiments import ExperimentHandler
from personas_backend.db.models import Persona
from personas_backend.db.schema_config import get_experimental_schema
from personas_backend.db.session import get_session
from personas_backend.models_providers.models_config import get_models_config
from personas_backend.questionnaire import get_questionnaire_json
from sqlmodel import Session

SYSTEM_ROLE_TEMPLATE = (
    "You are required to adopt and impersonate the personality of the human "
    "described as follow:\n<row.description>"
)


def _normalize_populations(populations: Sequence[str] | str | None) -> List[str]:
    """Return a deduplicated list of populations preserving input order."""

    if populations is None:
        raise ValueError("At least one population must be provided")
    if isinstance(populations, str):
        candidates: Iterable[str] = [populations]
    else:
        candidates = populations
    normalised: List[str] = []
    seen: set[str] = set()
    for population in candidates:
        if not population:
            continue
        if population in seen:
            continue
        seen.add(population)
        normalised.append(population)
    if not normalised:
        raise ValueError("At least one valid population must be provided")
    return normalised


def _get_repo_sha() -> str:
    """Return the current repository SHA, falling back to ``unknown``."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:  # pragma: no cover - defensive fallback
        return "unknown"


def _resolve_personality_id(persona: Persona) -> int:
    """Choose the best identifier for the experiment payload."""

    if persona.ref_personality_id is not None:
        return int(persona.ref_personality_id)
    if persona.id is not None:
        return int(persona.id)
    raise ValueError("Persona is missing both ref_personality_id and id")


def _validate_identifier(identifier: str, *, kind: str = "identifier") -> str:
    """Ensure schema/table identifiers are simple and safe."""

    if (
        not identifier
        or not identifier.replace("_", "").isalnum()
        or not (identifier[0].isalpha() or identifier[0] == "_")
    ):
        raise ValueError(f"Invalid {kind}: {identifier}")
    return identifier


SessionFactory = Callable[[], ContextManager[Session]]


def fetch_personas(
    model: str,
    populations: Sequence[str] | str,
    *,
    session_factory: SessionFactory | None = None,
    logger: logging.Logger | None = None,
    schema: str | None = None,
) -> List[Persona]:
    """Load personas filtered by model and population using SQLModel sessions.

    Args:
        model: LLM model identifier (e.g., 'gpt4o')
        populations: Population name(s) to filter by
        session_factory: Optional session factory override
        logger: Optional logger instance
        schema: Database schema to query. If None, uses experimental schema
                from ConfigManager (schema.target_schema).

    Returns:
        List of Persona objects matching the filters
    """

    if not model:
        raise ValueError("Model must be provided")
    normalised_populations = _normalize_populations(populations)
    session_factory = session_factory or get_session
    logger = logger or logging.getLogger(__name__)

    # Determine which schema to query
    from personas_backend.db.schema_config import get_experimental_schema

    query_schema = _validate_identifier(schema or get_experimental_schema(), kind="schema")

    with session_factory() as session:  # type: ignore[misc]
        assert isinstance(session, Session)  # help type checkers
        # Detect if using SQLite (no schema support)
        is_sqlite = session.bind is not None and session.bind.dialect.name == "sqlite"
        table_ref = "personas" if is_sqlite else f"{query_schema}.personas"

        from sqlalchemy import text as sql_text

        # Adjust population filtering for SQLite
        if is_sqlite:
            # Build dynamic placeholders based on actual number of populations
            placeholders = ", ".join(f":pop{i+1}" for i in range(len(normalised_populations)))
            raw_query = f"""
                SELECT * FROM {table_ref}
                WHERE model = :model
                AND population IN ({placeholders})
            """
            pop_args = {f"pop{i+1}": p for i, p in enumerate(normalised_populations)}
            result = session.execute(sql_text(raw_query), {"model": model, **pop_args})
        else:
            raw_query = f"""
                SELECT * FROM {table_ref}
                WHERE model = :model
                AND population = ANY(:populations)
            """
            result = session.execute(
                sql_text(raw_query), {"model": model, "populations": normalised_populations}
            )

        personas = []
        for row in result:
            persona_dict = dict(row._mapping)
            personas.append(Persona(**persona_dict))

        logger.info(
            "Loaded %d personas from schema=%s for model=%s populations=%s",
            len(personas),
            query_schema,
            model,
            normalised_populations,
        )
        return personas


def register_questionnaire_experiments(
    questionnaire: str,
    model: str,
    populations: Sequence[str] | str,
    *,
    experiment_description: str | None = None,
    experiment_group_handler: ExperimentGroupHandler | None = None,
    experiment_handler: ExperimentHandler | None = None,
    session_factory: SessionFactory | None = None,
    logger: logging.Logger | None = None,
    max_personas: int | None = None,
    max_repetitions: int | None = None,
    schema: str | None = None,
) -> Tuple[int, List[int]]:
    """Register questionnaire experiments for personas matching a model/population.

    Parameters ``max_personas`` and ``max_repetitions`` are primarily intended for
    smoke-testing flows where it is useful to cap the number of registered
    experiments without altering database state.

    The ``schema`` parameter allows querying personas from a specific database
    schema (e.g., experimental vs production). If None, uses the experimental
    schema from ConfigManager (schema.target_schema).
    """

    logger = logger or logging.getLogger(__name__)
    personas = fetch_personas(
        model,
        populations,
        session_factory=session_factory,
        logger=logger,
        schema=schema,
    )
    if not personas:
        raise ValueError("No personas found for the provided filters")

    if max_personas is not None:
        if max_personas <= 0:
            raise ValueError("max_personas must be greater than zero")
        personas = personas[:max_personas]

    model_config = get_models_config(model)
    questionnaire_json = get_questionnaire_json(questionnaire)

    populations_list = _normalize_populations(populations)
    description = experiment_description or (
        f"{questionnaire} questionnaire for {model} ({', '.join(populations_list)})"
    )
    ii_repeated = any((persona.repetitions or 1) > 1 for persona in personas)

    group_handler = experiment_group_handler or ExperimentGroupHandler(
        logger=logger,
        schema=schema or get_experimental_schema(),
    )
    group_id = group_handler.register_new_experiments_group(
        description=description,
        system_role=SYSTEM_ROLE_TEMPLATE,
        base_prompt=questionnaire_json,
        translated=False,
        temperature=model_config.get("TEMPERATURE", 1.0),
        top_p=model_config.get("TOP_P", 1.0),
        ii_repeated=int(ii_repeated),
        schema=schema,  # Pass schema through
    )

    repo_sha = _get_repo_sha()
    experiments_handler = experiment_handler or ExperimentHandler(
        logger=logger,
        schema=schema or get_experimental_schema(),
    )
    experiment_ids: List[int] = []

    for persona in personas:
        repetitions = persona.repetitions or 1
        if max_repetitions is not None:
            if max_repetitions <= 0:
                raise ValueError("max_repetitions must be greater than zero")
            repetitions = min(repetitions, max_repetitions)
        for repeat_index in range(1, repetitions + 1):
            payload = {
                "experiments_group_id": group_id,
                "repeated": repeat_index,
                "questionnaire": questionnaire,
                "language_instructions": "english",
                "language_questionnaire": "english",
                "model_provider": model_config.get("MODEL_PROVIDER"),
                "model": model_config.get("MODEL_ID"),
                "population": persona.population,
                "personality_id": _resolve_personality_id(persona),
                "repo_sha": repo_sha,
            }
            experiment_id = experiments_handler.register_experiment(
                payload,
                schema=schema,  # Pass schema through
            )
            experiment_ids.append(experiment_id)
            logger.debug(
                "Registered experiment_id=%s for persona_id=%s repetition=%s",
                experiment_id,
                persona.id,
                repeat_index,
            )

    return group_id, experiment_ids


__all__ = [
    "fetch_personas",
    "register_questionnaire_experiments",
    "SYSTEM_ROLE_TEMPLATE",
]
