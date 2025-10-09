"""Typer-based CLI that orchestrates persona generation and evaluation."""

import runpy
from pathlib import Path
from typing import Annotated, Optional, Sequence

import pandas as pd
import typer
from personas_backend.cli import (
    generate_borderline_to_artifacts,
    generate_borderline_to_database,
    generate_personas_to_artifacts,
    generate_personas_to_database,
    resolve_borderline_conditions,
    resolve_model_ids,
)
from personas_backend.core.enums import PersonaGenerationCondition
from personas_backend.db.db_handler import DatabaseHandler
from personas_backend.evaluate_questionnaire.registration import (
    fetch_personas,
    register_questionnaire_experiments,
)
from personas_backend.evaluate_questionnaire.runner import run_pending_experiments
from personas_backend.population.random_population import (
    generate_random_questionnaires_from_distribution,
)

DEFAULT_REFERENCE_QUESTIONNAIRES_TABLE = "exp20250226_reference_questionnaires"
DEFAULT_POPULATION = "spain826"
DEFAULT_MODEL = "gpt4o"
DEFAULT_BORDERLINE_CONDITION = PersonaGenerationCondition.MAX_N.value
DEFAULT_POPULATION_SCHEMA = "epqra"


def _coerce_sequence(values: Sequence[str] | None, default: str) -> list[str]:
    """Normalise Typer ``multiple=True`` options into a populated list."""

    if values:
        normalised = [value for value in values if value]
        if normalised:
            return normalised
    return [default]


def _format_list(values: Sequence[str]) -> str:
    return ", ".join(values)


app = typer.Typer(help="End-to-end pipeline utilities")


@app.command("init-db")
def init_db_cmd() -> None:
    """Initialize database schema using SQLModel metadata."""
    from personas_backend.db.session import init_db

    init_db()
    typer.echo("Database initialized")


@app.command("seed-ref-pop")
def seed_ref_pop(
    table_name: Annotated[
        str,
        typer.Option(
            help=(f"Destination table name (default: {DEFAULT_REFERENCE_QUESTIONNAIRES_TABLE})")
        ),
    ] = DEFAULT_REFERENCE_QUESTIONNAIRES_TABLE,
    num_questionnaires: Annotated[
        int, typer.Option(help="Number of synthetic questionnaires to generate")
    ] = 826,
    source_personality_id: Annotated[
        int, typer.Option(help="Source personality ID to use as reference (default: 1)")
    ] = 1,
    no_db: Annotated[
        bool, typer.Option("--no-db", help="Write to artifacts/ instead of PostgreSQL")
    ] = False,
    artifacts_dir: Annotated[
        Path,
        typer.Option(
            help="Root artifact directory (inputs stored under <dir>/inputs when --no-db)"
        ),
    ] = Path("artifacts"),
) -> None:
    """Seed reference population using synthetic questionnaires generated from database data."""

    # Get reference data from the database
    db = DatabaseHandler()
    ref_eval = pd.read_sql(
        f"""
        SELECT category, key, question_number, answer, question
        FROM exp.reference_questionnaires
        WHERE personality_id = {source_personality_id}
        ORDER BY question_number
        """,
        db.connection,
    )

    if ref_eval.empty:
        typer.echo(f"❌ No reference data found for personality_id {source_personality_id}")
        return

    typer.echo(
        f"✅ Using reference data from personality_id {source_personality_id} ({len(ref_eval)} questions)"
    )

    questionnaires = generate_random_questionnaires_from_distribution(
        ref_eval, num_questionnaires=num_questionnaires
    )
    row_count = len(questionnaires)

    if no_db:
        input_dir = artifacts_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        dest = input_dir / f"{table_name}.csv"
        questionnaires.to_csv(dest, index=False)
        typer.echo(f"Generated {row_count} questionnaires to artifacts at {dest.resolve()}")
        return

    db = DatabaseHandler()
    questionnaires.to_sql(
        table_name,
        db.connection,
        schema="exp",
        if_exists="replace",
        index=False,
    )
    typer.echo(f"Seeded {row_count} questionnaires into exp.{table_name}")


@app.command("create-experiment-group")
def create_experiment_group(
    description: Annotated[str, typer.Option(help="Description for the experiment group")],
    system_role: Annotated[str, typer.Option(help="System role prompt for LLM")] = (
        "You are required to adopt and impersonate the personality of the human described as follow:\\n<row.description>"
    ),
    base_prompt: Annotated[str, typer.Option(help="Base prompt template")] = "{}",
    temperature: Annotated[float, typer.Option(help="Temperature for LLM")] = 1.0,
    top_p: Annotated[float, typer.Option(help="Top-p for LLM")] = 1.0,
    translated: Annotated[bool, typer.Option(help="Whether using translated content")] = False,
    ii_repeated: Annotated[bool, typer.Option(help="Whether using repeated iterations")] = False,
) -> None:
    """Create a new experiment group and return its ID."""
    from personas_backend.db.experiment_groups import ExperimentGroupHandler

    exp_group_handler = ExperimentGroupHandler()

    group_id = exp_group_handler.register_new_experiments_group(
        description=description,
        system_role=system_role,
        base_prompt=base_prompt,
        translated=translated,
        temperature=temperature,
        top_p=top_p,
        ii_repeated=ii_repeated,
    )

    typer.echo(f"Created experiment group with ID: {group_id}")
    typer.echo(f"Use --experiment-group-id {group_id} in other commands")


@app.command("generate-personas")
def generate_personas(
    run_id: str,
    models: Annotated[list[str] | None, typer.Option(help="Models to use")] = None,
    experimental: Annotated[bool, typer.Option(help="Enable experimental models")] = False,
    ref_population: str = "spain826",
    reference_questionnaires: str = DEFAULT_REFERENCE_QUESTIONNAIRES_TABLE,
    reference_schema: Annotated[
        str, typer.Option(help="Schema for reference questionnaires")
    ] = "exp",
    population_schema: Annotated[str, typer.Option(help="Schema for population table")] = "epqra",
    start_repetitions: int = 6,
    repetitions: int = 1,
    no_db: Annotated[
        bool,
        typer.Option(
            "--no-db",
            help=(
                "Write CSV artifacts instead of PostgreSQL outputs."
                " Requires population/questionnaire inputs."
            ),
        ),
    ] = False,
    artifacts_dir: Annotated[
        Path,
        typer.Option(help="Base directory for input and output artifacts when --no-db is set"),
    ] = Path("artifacts"),
    population_artifact: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Optional explicit path to the population CSV when running --no-db. "
                "Defaults to <artifacts-dir>/inputs/<ref_population>.csv"
            )
        ),
    ] = None,
    questionnaires_artifact: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Optional explicit path to questionnaire CSV when running --no-db. "
                "Defaults to <artifacts-dir>/inputs/<reference_questionnaires>.csv"
            )
        ),
    ] = None,
) -> None:
    """Generate personas using configured models (CSV export with --no-db)."""
    model_slugs = _coerce_sequence(models, DEFAULT_MODEL)
    try:
        model_ids = resolve_model_ids(model_slugs, allow_experimental=experimental)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))

    table_name: str
    if no_db:
        input_dir = artifacts_dir / "inputs"
        output_dir = artifacts_dir / "outputs" / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        population_path = population_artifact or input_dir / f"{ref_population}.csv"
        questionnaires_path = (
            questionnaires_artifact or input_dir / f"{reference_questionnaires}.csv"
        )

        if not population_path.exists():
            raise typer.BadParameter(
                f"Population artifact not found: {population_path}",
            )
        if not questionnaires_path.exists():
            raise typer.BadParameter(
                f"Questionnaire artifact not found: {questionnaires_path}",
            )

        population_df = pd.read_csv(population_path)
        questionnaires_df = pd.read_csv(questionnaires_path)
        try:
            table_name, df = generate_personas_to_artifacts(
                run_id=run_id,
                models=model_ids,
                ref_population=ref_population,
                population_df=population_df,
                questionnaires_df=questionnaires_df,
                start_repetitions=start_repetitions,
                repetitions=repetitions,
                allow_experimental=experimental,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc))
        dest = output_dir / f"{table_name}.csv"
        df.to_csv(dest, index=False)
        typer.echo(f"Generated {len(df)} personas to artifacts at {dest.resolve()}")
        return
    db_handler = DatabaseHandler()
    try:
        table_name = generate_personas_to_database(
            run_id=run_id,
            models=model_ids,
            ref_population=ref_population,
            population_schema=population_schema,
            reference_table=reference_questionnaires,
            reference_schema=reference_schema,
            db_handler=db_handler,
            start_repetitions=start_repetitions,
            repetitions=repetitions,
            allow_experimental=experimental,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(
        "Triggered persona generation for "
        f"{_format_list(model_slugs)} into population.{table_name}"
    )


@app.command("generate-borderline")
def generate_borderline(
    run_id: str,
    models: Annotated[list[str] | None, typer.Option(help="Models to use")] = None,
    conditions: Annotated[list[str] | None, typer.Option(help="Borderline conditions")] = None,
    experimental: Annotated[bool, typer.Option(help="Enable experimental options")] = False,
    ref_population: str = "spain826",
    reference_questionnaires: str = DEFAULT_REFERENCE_QUESTIONNAIRES_TABLE,
    reference_schema: Annotated[
        str, typer.Option(help="Schema for reference questionnaires")
    ] = "exp",
    start_repetitions: int = 1,
    repetitions: int = 5,
    no_db: Annotated[
        bool,
        typer.Option(
            "--no-db",
            help=(
                "Write CSV artifacts instead of PostgreSQL outputs."
                " Requires population/questionnaire inputs."
            ),
        ),
    ] = False,
    artifacts_dir: Annotated[
        Path,
        typer.Option(help="Base directory for input and output artifacts when --no-db is set"),
    ] = Path("artifacts"),
    population_artifact: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Optional explicit path to the population CSV when running --no-db. "
                "Defaults to <artifacts-dir>/inputs/<ref_population>.csv"
            )
        ),
    ] = None,
    questionnaires_artifact: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Optional explicit path to questionnaire CSV when running --no-db. "
                "Defaults to <artifacts-dir>/inputs/<reference_questionnaires>.csv"
            )
        ),
    ] = None,
) -> None:
    """Generate personas under borderline trait conditions (CSV export with --no-db)."""
    model_slugs = _coerce_sequence(models, DEFAULT_MODEL)
    try:
        model_ids = resolve_model_ids(model_slugs, allow_experimental=experimental)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))

    condition_slugs = _coerce_sequence(conditions, DEFAULT_BORDERLINE_CONDITION)
    try:
        condition_ids = resolve_borderline_conditions(
            condition_slugs, allow_experimental=experimental
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))

    table_name: str
    if no_db:
        input_dir = artifacts_dir / "inputs"
        output_dir = artifacts_dir / "outputs" / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        population_path = population_artifact or input_dir / f"{ref_population}.csv"
        questionnaires_path = (
            questionnaires_artifact or input_dir / f"{reference_questionnaires}.csv"
        )

        if not population_path.exists():
            raise typer.BadParameter(
                f"Population artifact not found: {population_path}",
            )
        if not questionnaires_path.exists():
            raise typer.BadParameter(
                f"Questionnaire artifact not found: {questionnaires_path}",
            )

        population_df = pd.read_csv(population_path)
        questionnaires_df = pd.read_csv(questionnaires_path)
        try:
            table_name, df = generate_borderline_to_artifacts(
                run_id=run_id,
                models=model_ids,
                conditions=condition_ids,
                ref_population=ref_population,
                population_df=population_df,
                questionnaires_df=questionnaires_df,
                start_repetitions=start_repetitions,
                repetitions=repetitions,
                allow_experimental=experimental,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc))
        dest = output_dir / f"{table_name}.csv"
        df.to_csv(dest, index=False)
        typer.echo(f"Generated {len(df)} borderline personas to artifacts at {dest.resolve()}")
        return
    db_handler = DatabaseHandler()
    try:
        table_name = generate_borderline_to_database(
            run_id=run_id,
            models=model_ids,
            conditions=condition_ids,
            ref_population=ref_population,
            population_schema=DEFAULT_POPULATION_SCHEMA,
            reference_table=reference_questionnaires,
            reference_schema=reference_schema,
            db_handler=db_handler,
            start_repetitions=start_repetitions,
            repetitions=repetitions,
            allow_experimental=experimental,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(
        "Triggered borderline persona generation for "
        f"{_format_list(condition_slugs)} into population.{table_name}"
    )


def _validate_positive(name: str, value: int | None) -> None:
    if value is None:
        return
    if value <= 0:
        raise typer.BadParameter(f"{name} must be greater than zero")


@app.command("register-experiments")
def register_experiments_cmd(
    questionnaire: Annotated[
        str,
        typer.Option(help="Questionnaire slug to register (e.g. epqra, bigfive)"),
    ] = "epqra",
    populations: Annotated[
        Sequence[str] | None,
        typer.Option(
            "--population",
            "-p",
            help="Population identifiers to include (repeat for multiples)",
            show_default=False,
        ),
    ] = None,
    models: Annotated[
        Sequence[str] | None,
        typer.Option(
            "--model",
            "-m",
            help="Model identifiers to register (repeat for multiples)",
            show_default=False,
        ),
    ] = None,
    max_personas: Annotated[
        int | None,
        typer.Option(help="Limit personas per model during registration"),
    ] = None,
    max_repetitions: Annotated[
        int | None,
        typer.Option(help="Cap repetitions per persona during registration"),
    ] = None,
    experiment_description: Annotated[
        Optional[str],
        typer.Option(help="Optional override for the experiment group description"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(help="Preview registration counts without writing to the DB"),
    ] = False,
) -> None:
    """Register questionnaire experiments for stored personas."""

    _validate_positive("max-personas", max_personas)
    _validate_positive("max-repetitions", max_repetitions)

    population_list = _coerce_sequence(populations, DEFAULT_POPULATION)
    model_list = _coerce_sequence(models, DEFAULT_MODEL)

    for model in model_list:
        if dry_run:
            personas = fetch_personas(model, population_list)
            if max_personas is not None:
                personas = personas[:max_personas]
            total_repetitions = 0
            for persona in personas:
                repetitions = persona.repetitions or 1
                if max_repetitions is not None:
                    repetitions = min(repetitions, max_repetitions)
                total_repetitions += repetitions
            typer.echo(
                f"[DRY RUN] {model}: {len(personas)} personas -> {total_repetitions} experiments"
            )
            continue

        try:
            group_id, experiment_ids = register_questionnaire_experiments(
                questionnaire=questionnaire,
                model=model,
                populations=population_list,
                experiment_description=experiment_description,
                max_personas=max_personas,
                max_repetitions=max_repetitions,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

        typer.echo(
            f"Registered {len(experiment_ids)} experiments for model {model} under group {group_id}"
        )


@app.command("run-experiments")
def run_experiments_cmd(
    experiments_group_id: Annotated[
        Sequence[int] | None,
        typer.Option(
            "--group-id",
            "-g",
            help="Existing experiment group IDs to process",
            show_default=False,
        ),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(help="Limit the number of experiments executed in this invocation"),
    ] = None,
    max_workers: Annotated[
        int,
        typer.Option(help="Maximum concurrent questionnaire executions"),
    ] = 3,
    schema: Annotated[
        str,
        typer.Option(help="Database schema containing questionnaire evaluation tables"),
    ] = "personality_trap",
    dry_run: Annotated[
        bool,
        typer.Option(help="Preview execution without contacting model providers"),
    ] = False,
) -> None:
    """Run pending questionnaire experiments with optional batching."""

    group_ids = list(experiments_group_id or [])
    if dry_run:
        scope = (
            f"experiment groups {', '.join(map(str, group_ids))}"
            if group_ids
            else "all pending experiment groups"
        )
        typer.echo(
            f"[DRY RUN] Would process {scope} (batch_size={batch_size or 'all'}, max_workers={max_workers})"
        )
        return

    typer.echo(
        "Processing questionnaire experiments for "
        + (f"groups {', '.join(map(str, group_ids))}" if group_ids else "all pending groups")
    )
    run_pending_experiments(
        experiments_group_ids=group_ids or None,
        batch_size=batch_size,
        max_workers=max_workers,
        schema=schema,
    )


@app.command("run-evals")
def run_evals(
    questionnaire: Annotated[
        str,
        typer.Option(help="Questionnaire slug to evaluate (e.g. epqra, bigfive)"),
    ] = "epqra",
    populations: Annotated[
        Sequence[str] | None,
        typer.Option(
            "--population",
            "-p",
            help="Populations to include when registering new experiments",
            show_default=False,
        ),
    ] = None,
    models: Annotated[
        Sequence[str] | None,
        typer.Option(
            "--model",
            "-m",
            help="Models to evaluate (repeat for multiple)",
            show_default=False,
        ),
    ] = None,
    max_personas: Annotated[
        int | None,
        typer.Option(help="Limit personas per model during registration"),
    ] = None,
    max_repetitions: Annotated[
        int | None,
        typer.Option(help="Cap repetitions per persona during registration"),
    ] = None,
    experiment_group_id: Annotated[
        Sequence[int] | None,
        typer.Option(
            "--experiment-group-id",
            "-g",
            help="Reuse existing experiment group IDs instead of creating new ones",
            show_default=False,
        ),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(help="Limit the number of experiments executed in this invocation"),
    ] = None,
    max_workers: Annotated[
        int,
        typer.Option(help="Maximum concurrent questionnaire executions"),
    ] = 3,
    schema: Annotated[
        str,
        typer.Option(help="Database schema containing questionnaire evaluation tables"),
    ] = "personality_trap",
    skip_registration: Annotated[
        bool,
        typer.Option(help="Skip registration and only run previously queued experiments"),
    ] = False,
    run_id: Annotated[
        Optional[str],
        typer.Option(help="Legacy alias retained for compatibility; no longer used"),
    ] = None,
) -> None:
    """Register questionnaire experiments and immediately process them."""

    _validate_positive("max-personas", max_personas)
    _validate_positive("max-repetitions", max_repetitions)

    if run_id:
        typer.echo(
            "ℹ️  --run-id is retained for backwards compatibility but no longer affects execution.",
        )

    group_ids = list(experiment_group_id or [])

    if not skip_registration:
        population_list = _coerce_sequence(populations, DEFAULT_POPULATION)
        model_list = _coerce_sequence(models, DEFAULT_MODEL)

        for model in model_list:
            typer.echo(
                f"Registering questionnaire '{questionnaire}' for model {model}"
                f" across populations {_format_list(population_list)}"
            )
            try:
                group_id, experiment_ids = register_questionnaire_experiments(
                    questionnaire=questionnaire,
                    model=model,
                    populations=population_list,
                    max_personas=max_personas,
                    max_repetitions=max_repetitions,
                )
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc

            typer.echo(f"Registered {len(experiment_ids)} experiments under group {group_id}")
            group_ids.append(group_id)
    elif not group_ids:
        raise typer.BadParameter("--skip-registration requires at least one --experiment-group-id")

    unique_group_ids = list(dict.fromkeys(group_ids))

    if not unique_group_ids:
        typer.echo("No experiment groups to execute; exiting.")
        return

    typer.echo(
        "Running questionnaire experiments for groups: " + ", ".join(map(str, unique_group_ids))
    )
    run_pending_experiments(
        experiments_group_ids=unique_group_ids,
        batch_size=batch_size,
        max_workers=max_workers,
        schema=schema,
    )


@app.command("analyze")
def analyze() -> None:
    """Run population analysis scripts."""
    runpy.run_module("evaluations.population_analysis", run_name="__main__")


if __name__ == "__main__":
    app()
