"""Orchestrate questionnaire evaluation for registered experiments.

This module mirrors the behaviour of the historical ``experiments_original``
script while leveraging the refactored repositories.  It loads pending
experiments, prepares persona/system-role payloads and executes the
questionnaire handlers concurrently in small batches.
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from personas_backend.db.db_handler import DatabaseHandler
from personas_backend.db.experiment_groups import ExperimentGroupHandler
from personas_backend.db.experiments import ExperimentHandler
from personas_backend.db.models import (
    ExperimentsGroup,
    ExperimentsList,
    Persona,
)
from personas_backend.db.schema_config import get_target_schema
from personas_backend.models_providers.models_utils import (
    get_model_name_by_id,
    get_model_wrapper,
)
from personas_backend.questionnaire import get_questionnaire_handler
from personas_backend.questionnaire.base_questionnaire import (
    ExperimentRunResult,
    QuestionnaireBase,
)
from personas_backend.utils.logger import setup_logger
from sqlalchemy import and_, or_, text
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

DEFAULT_SCHEMA = get_target_schema()


@dataclass(frozen=True)
class PendingExperiment:
    """Serializable payload for a queued experiment."""

    experiment_id: int
    experiments_group_id: int
    questionnaire: str
    model_provider: Optional[str]
    model_identifier: str
    model_name: str
    population: Optional[str]
    personality_id: Optional[int]
    repeated: Optional[int]
    system_role_template: Optional[str]
    base_prompt: Optional[str]
    group_description: Optional[str]
    persona_payload: Dict[str, Any]


def _validate_schema_name(schema: str) -> str:
    """Ensure schema identifier is simple and SQL-safe."""

    if (
        not schema
        or not schema.replace("_", "").isalnum()
        or not (schema[0].isalpha() or schema[0] == "_")
    ):
        raise ValueError(f"Invalid schema: {schema}")
    return schema


def _persona_to_payload(persona: Persona) -> Dict[str, Any]:
    """Convert a ``Persona`` row into a JSON-serialisable dictionary."""

    payload: Dict[str, Any] = persona.model_dump(exclude_none=True)
    # Remove SQLModel internal identity fields that are not serialisable.
    payload.pop("id", None)
    payload.pop("ref_personality_id", None)
    return payload


def _render_system_role(
    template: Optional[str],
    persona_payload: Dict[str, Any],
) -> Optional[str]:
    """Render system_role placeholders with persona data."""

    if not template:
        return template
    rendered = template
    for key, value in persona_payload.items():
        placeholder = f"<row.{key}>"
        if placeholder in rendered:
            rendered = rendered.replace(placeholder, str(value))
    return rendered


def _clear_existing_answers(
    engine: Engine, experiment_id: int, schema: str, logger: logging.Logger
) -> None:
    """Remove previous questionnaire answers before re-processing."""

    safe_schema = _validate_schema_name(schema)
    with engine.begin() as connection:
        connection.execute(
            text(
                f"DELETE FROM {safe_schema}.eval_questionnaires "
                "WHERE experiment_id = :experiment_id"
            ),
            {"experiment_id": experiment_id},
        )
    logger.debug(
        "Cleared eval_questionnaires rows for experiment_id=%s",
        experiment_id,
    )


def _resolve_persona(
    session: Session, experiment: ExperimentsList, model_name: str
) -> Optional[Persona]:
    """Locate the persona record backing an experiment."""

    stmt = (
        select(Persona)
        .where(Persona.population == experiment.population)
        .where(
            or_(
                Persona.ref_personality_id == experiment.personality_id,
                and_(
                    Persona.ref_personality_id.is_(None),
                    Persona.id == experiment.personality_id,
                ),
            )
        )
        .where(Persona.model == model_name)
    )
    persona = session.exec(stmt).first()
    if persona is not None:
        return persona

    # Fallback: drop the model filter in case legacy data used raw identifiers.
    fallback_stmt = (
        select(Persona)
        .where(Persona.population == experiment.population)
        .where(
            or_(
                Persona.ref_personality_id == experiment.personality_id,
                Persona.id == experiment.personality_id,
            )
        )
    )
    return session.exec(fallback_stmt).first()


def _load_pending_experiments_raw_sql(
    engine: Engine,
    *,
    experiments_group_ids: Optional[Sequence[int]] = None,
    batch_size: Optional[int] = None,
    schema: str = DEFAULT_SCHEMA,
    logger: logging.Logger,
) -> List[PendingExperiment]:
    """Fetch experiments with NULL success using raw SQL for schema support."""

    safe_schema = _validate_schema_name(schema)

    # Build the query with schema-aware table references
    query = f"""
    SELECT
        el.experiment_id,
        el.experiments_group_id,
        el.questionnaire,
        el.model_provider,
        el.model,
        el.population,
        el.personality_id,
        el.repeated,
        eg.system_role,
        eg.base_prompt,
        eg.description as group_description,
        p.name, p.age, p.gender, p.race, p.sexual_orientation,
        p.ethnicity, p.religious_belief, p.occupation,
        p.political_orientation, p.location, p.description,
        p.word_count_description, p.repetitions
    FROM {safe_schema}.experiments_list el
    JOIN {safe_schema}.experiments_groups eg
        ON el.experiments_group_id = eg.experiments_group_id
    LEFT JOIN {safe_schema}.personas p ON (
        p.ref_personality_id = el.personality_id
        AND p.population = el.population
    )
    WHERE el.succeeded IS NULL
    """

    params: Dict[str, object] = {}
    use_group_filter = experiments_group_ids is not None and len(experiments_group_ids) > 0
    if experiments_group_ids is not None and experiments_group_ids:
        # Use IN with expanding bindparam for safe list expansion
        query += " AND el.experiments_group_id IN :group_ids"
        params["group_ids"] = list(experiments_group_ids)

    query += " ORDER BY el.experiment_id DESC"

    if batch_size:
        query += " LIMIT :batch_size"
        params["batch_size"] = int(batch_size)

    # Execute raw SQL query
    import pandas as pd
    from sqlalchemy import bindparam, text

    stmt = text(query)
    if use_group_filter:
        # Enable expanding for IN (:group_ids)
        stmt = stmt.bindparams(bindparam("group_ids", expanding=True))

    df = pd.read_sql_query(stmt, engine, params=params)

    pending: List[PendingExperiment] = []
    for _, row in df.iterrows():
        if not row["model"]:
            logger.warning(
                "Skipping experiment_id=%s: model identifier missing",
                row["experiment_id"],
            )
            continue

        try:
            model_name = get_model_name_by_id(row["model"])
        except KeyError:
            logger.warning(
                "Skipping experiment_id=%s due to unknown model id '%s'",
                row["experiment_id"],
                row["model"],
            )
            # Skip experiments with unsupported/unknown model ids
            continue

        if pd.isna(row["name"]):  # No persona found
            logger.warning(
                "No persona for experiment_id=%s (personality_id=%s, pop=%s)",
                row["experiment_id"],
                row["personality_id"],
                row["population"],
            )
            continue

        # Build persona payload from the joined data
        persona_payload = {
            "name": row["name"],
            "age": row["age"],
            "gender": row["gender"],
            "race": row["race"],
            "sexual_orientation": row["sexual_orientation"],
            "ethnicity": row["ethnicity"],
            "religious_belief": row["religious_belief"],
            "occupation": row["occupation"],
            "political_orientation": row["political_orientation"],
            "location": row["location"],
            "description": row["description"],
            "word_count_description": row["word_count_description"],
            "repetitions": row["repetitions"],
        }
        # Remove None values
        persona_payload = {k: v for k, v in persona_payload.items() if pd.notna(v)}

        # Normalise questionnaire key to supported identifiers
        q_raw = (row["questionnaire"] or "epqra").strip().lower()
        if q_raw in {"bigfive", "big_five", "big5"}:
            questionnaire = "big5"
        elif q_raw in {"epqra"}:
            questionnaire = "epqra"
        else:
            logger.warning(
                "Skipping experiment_id=%s due to unsupported questionnaire '%s'",
                row["experiment_id"],
                row["questionnaire"],
            )
            continue

        pending.append(
            PendingExperiment(
                experiment_id=int(row["experiment_id"]),
                experiments_group_id=int(row["experiments_group_id"] or 0),
                questionnaire=questionnaire,
                model_provider=row["model_provider"],
                model_identifier=row["model"],
                model_name=model_name,
                population=row["population"],
                personality_id=row["personality_id"],
                repeated=row["repeated"],
                system_role_template=row["system_role"],
                base_prompt=row["base_prompt"],
                group_description=row["group_description"],
                persona_payload=persona_payload,
            )
        )

    return pending


def _load_pending_experiments(
    session: Session,
    *,
    experiments_group_ids: Optional[Sequence[int]] = None,
    batch_size: Optional[int] = None,
    logger: logging.Logger,
) -> List[PendingExperiment]:
    """Fetch ``succeeded IS NULL`` experiments joined to personas and groups.

    NOTE: This function only works with the default schema due to SQLModel
    limitations.
    Use _load_pending_experiments_raw_sql for schema-aware operations.
    """

    stmt = (
        select(ExperimentsList, ExperimentsGroup)
        .join(
            ExperimentsGroup,
            ExperimentsGroup.experiments_group_id == ExperimentsList.experiments_group_id,
        )
        .where(ExperimentsList.succeeded.is_(None))
        .order_by(ExperimentsList.experiment_id.desc())
    )
    if experiments_group_ids:
        stmt = stmt.where(ExperimentsList.experiments_group_id.in_(set(experiments_group_ids)))
    if batch_size:
        stmt = stmt.limit(int(batch_size))

    rows = session.exec(stmt).all()
    pending: List[PendingExperiment] = []

    for experiment, group in rows:
        if not experiment.model:
            logger.warning(
                "Skipping experiment_id=%s because model identifier is missing",
                experiment.experiment_id,
            )
            continue
        try:
            model_name = get_model_name_by_id(experiment.model)
        except KeyError:
            logger.warning(
                "Unknown model identifier '%s' for experiment_id=%s; " "using raw value",
                experiment.model,
                experiment.experiment_id,
            )
            model_name = experiment.model

        persona = _resolve_persona(session, experiment, model_name)
        if persona is None:
            logger.warning(
                "No persona found for experiment_id=%s " "(personality_id=%s, population=%s)",
                experiment.experiment_id,
                experiment.personality_id,
                experiment.population,
            )
            continue

        persona_payload = _persona_to_payload(persona)
        questionnaire = experiment.questionnaire or "epqra"

        pending.append(
            PendingExperiment(
                experiment_id=int(experiment.experiment_id),
                experiments_group_id=int(experiment.experiments_group_id or 0),
                questionnaire=questionnaire,
                model_provider=experiment.model_provider,
                model_identifier=experiment.model,
                model_name=model_name,
                population=experiment.population,
                personality_id=experiment.personality_id,
                repeated=experiment.repeated,
                system_role_template=group.system_role,
                base_prompt=group.base_prompt,
                group_description=group.description,
                persona_payload=persona_payload,
            )
        )

    return pending


def _group_by_experiment_group(
    experiments: Iterable[PendingExperiment],
) -> Dict[int, List[PendingExperiment]]:
    grouped: Dict[int, List[PendingExperiment]] = {}
    for experiment in experiments:
        grouped.setdefault(experiment.experiments_group_id, []).append(experiment)
    return grouped


def _process_single_experiment(
    pending: PendingExperiment,
    *,
    questionnaire_handler: QuestionnaireBase,
    exp_handler: ExperimentHandler,
    engine: Engine,
    logger: logging.Logger,
    schema: str,
) -> ExperimentRunResult:
    """Execute one experiment via the questionnaire handler."""

    _clear_existing_answers(engine, pending.experiment_id, schema, logger)
    persona_payload = dict(pending.persona_payload)
    system_role = _render_system_role(
        pending.system_role_template,
        persona_payload,
    )

    my_experiment = {
        "experiment_id": pending.experiment_id,
        "experiments_group_id": pending.experiments_group_id,
        "model_provider": pending.model_provider,
        "model": pending.model_identifier,
        "questionnaire": pending.questionnaire,
        "population": pending.population,
        "personality_id": pending.personality_id,
        "repeated": pending.repeated,
    }

    logger.info(
        "Submitting experiment %s for personality %s",
        pending.experiment_id,
        pending.personality_id,
    )

    result = questionnaire_handler.process_experiment(
        my_experiment,
        persona_payload,
        system_role=system_role,
        auto_record=False,
    )
    if result is None:
        raise RuntimeError("Experiment %s failed to produce valid answers" % pending.experiment_id)

    exp_handler.record_request_metadata_and_status(
        engine,
        logger,
        experiment_id=pending.experiment_id,
        succeeded=True,
        llm_explanation=result.explanation,
        request_json=result.request_json,
        response_json=result.response_json,
        request_metadata=result.request_metadata,
        schema=schema,
    )
    return result


def _process_experiment_group(
    group_id: int,
    experiments: List[PendingExperiment],
    *,
    logger: logging.Logger,
    engine: Engine,
    exp_handler: ExperimentHandler,
    exp_groups_handler: ExperimentGroupHandler,
    max_workers: int,
    schema: str,
) -> None:
    """Run experiments for a single group using a thread pool."""

    if not experiments:
        logger.info(
            "No experiments to process for experiments_group_id %s",
            group_id,
        )
        return

    logger.info("Processing experiments for experiments_group_id %s", group_id)
    logger.info("Found %d experiments to process", len(experiments))

    first = experiments[0]
    logger.info(
        "Model: %s, Population: %s, Questionnaire: %s",
        first.model_name,
        first.population,
        first.questionnaire,
    )

    model_wrapper = get_model_wrapper(first.model_name, logger)
    questionnaire_handler = get_questionnaire_handler(
        questionnaire_type=first.questionnaire,
        model_wrapper=model_wrapper,
        exp_handler=exp_handler,
        db_conn=engine,
        logger=logger,
        schema=schema,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_single_experiment,
                experiment,
                questionnaire_handler=questionnaire_handler,
                exp_handler=exp_handler,
                engine=engine,
                logger=logger,
                schema=schema,
            )
            for experiment in experiments
        ]

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
                completed += 1
                logger.info(
                    "Completed experiment %s/%s",
                    completed,
                    len(futures),
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error processing experiment: %s", exc)

    exp_groups_handler.update_experiments_groups_conclude(
        experiments_group_id_start=group_id,
        experiments_group_id_end=group_id,
    )
    logger.info("Finished processing experiment group %s", group_id)


def run_pending_experiments(
    experiments_group_ids: Optional[Sequence[int]] = None,
    *,
    batch_size: Optional[int] = None,
    max_workers: int = 3,
    schema: str = DEFAULT_SCHEMA,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Process queued questionnaire experiments with schema awareness."""

    logger = logger or setup_logger("evaluate_questionnaire_runner")
    db_handler = DatabaseHandler(logger=logger)
    engine = db_handler.connection
    exp_handler = ExperimentHandler(
        db_handler=db_handler,
        logger=logger,
        schema=schema,
    )
    exp_groups_handler = ExperimentGroupHandler(
        db_handler=db_handler,
        logger=logger,
        schema=schema,
    )

    try:
        # Use schema-aware raw SQL instead of SQLModel for flexibility
        pending = _load_pending_experiments_raw_sql(
            engine,
            experiments_group_ids=experiments_group_ids,
            batch_size=batch_size,
            schema=schema,
            logger=logger,
        )

        if not pending:
            logger.info("No unprocessed experiments found")
            return

        grouped = _group_by_experiment_group(pending)
        for group_id, experiments in grouped.items():
            _process_experiment_group(
                group_id,
                experiments,
                logger=logger,
                engine=engine,
                exp_handler=exp_handler,
                exp_groups_handler=exp_groups_handler,
                max_workers=max_workers,
                schema=schema,
            )
    finally:
        db_handler.close_connection()


__all__ = ["run_pending_experiments"]
