from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pandas as pd
import typer
from typer.testing import CliRunner

from tools import pipeline
from personas_backend.core.enums import ModelID, PersonaGenerationCondition

_cli_app = typer.Typer()


@_cli_app.command("register-experiments")
def register_experiments_cli(
    model: Optional[List[str]] = typer.Option(None, "--model", "-m"),
    population: Optional[List[str]] = typer.Option(None, "--population", "-p"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    pipeline.register_experiments_cmd(
        questionnaire="epqra",
        populations=population,
        models=model,
        max_personas=None,
        max_repetitions=None,
        experiment_description=None,
        dry_run=dry_run,
    )


@_cli_app.command("run-experiments")
def run_experiments_cli(
    group_id: Optional[List[int]] = typer.Option(None, "--group-id", "-g"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"),
    max_workers: int = typer.Option(3, "--max-workers"),
    schema: str = typer.Option("personality_trap", "--schema"),
):
    pipeline.run_experiments_cmd(
        experiments_group_id=group_id,
        batch_size=batch_size,
        max_workers=max_workers,
        schema=schema,
        dry_run=False,
    )


@_cli_app.command("generate-personas")
def generate_personas_cli(
    run_id: str,
    no_db: bool = typer.Option(False, "--no-db"),
    artifacts_dir: Path = typer.Option(Path("artifacts"), "--artifacts-dir"),
):
    pipeline.generate_personas(
        run_id=run_id,
        no_db=no_db,
        artifacts_dir=artifacts_dir,
    )


@_cli_app.command("generate-borderline")
def generate_borderline_cli(
    run_id: str,
    no_db: bool = typer.Option(False, "--no-db"),
    artifacts_dir: Path = typer.Option(Path("artifacts"), "--artifacts-dir"),
):
    pipeline.generate_borderline(
        run_id=run_id,
        no_db=no_db,
        artifacts_dir=artifacts_dir,
    )


def test_register_experiments_invokes_registration(monkeypatch):
    runner = CliRunner()
    recorded_calls: List[dict] = []

    def fake_register(**kwargs):
        recorded_calls.append(kwargs)
        return 42, [1, 2, 3]

    monkeypatch.setattr("tools.pipeline.register_questionnaire_experiments", fake_register)
    monkeypatch.setattr("tools.pipeline.fetch_personas", lambda *args, **kwargs: [])

    result = runner.invoke(
        _cli_app,
        [
            "register-experiments",
            "--model",
            "gpt4o",
            "--population",
            "pop_a",
            "--population",
            "pop_b",
        ],
    )

    assert result.exit_code == 0
    assert recorded_calls == [
        {
            "questionnaire": "epqra",
            "model": "gpt4o",
            "populations": ["pop_a", "pop_b"],
            "experiment_description": None,
            "max_personas": None,
            "max_repetitions": None,
        }
    ]


def test_register_experiments_dry_run_counts_personas(monkeypatch):
    runner = CliRunner()
    persona_calls: List[tuple] = []

    def fake_fetch(model, populations):
        persona_calls.append((model, tuple(populations)))
        return [SimpleNamespace(repetitions=2), SimpleNamespace(repetitions=None)]

    monkeypatch.setattr("tools.pipeline.fetch_personas", fake_fetch)
    monkeypatch.setattr(
        "tools.pipeline.register_questionnaire_experiments",
        lambda **_: (_ for _ in ()).throw(AssertionError("should not call")),
    )

    result = runner.invoke(
        _cli_app,
        [
            "register-experiments",
            "--model",
            "gpt4o",
            "--population",
            "pop_a",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert persona_calls == [("gpt4o", ("pop_a",))]
    assert "[DRY RUN] gpt4o: 2 personas -> 3 experiments" in result.stdout


def test_run_experiments_invokes_runner(monkeypatch):
    runner = CliRunner()
    recorded_calls: List[dict] = []

    def fake_run(**kwargs):
        recorded_calls.append(kwargs)

    monkeypatch.setattr("tools.pipeline.run_pending_experiments", fake_run)

    result = runner.invoke(
        _cli_app,
        [
            "run-experiments",
            "--group-id",
            "1",
            "--group-id",
            "2",
            "--batch-size",
            "5",
            "--max-workers",
            "4",
            "--schema",
            "custom",
        ],
    )

    assert result.exit_code == 0
    assert recorded_calls == [
        {
            "experiments_group_ids": [1, 2],
            "batch_size": 5,
            "max_workers": 4,
            "schema": "custom",
        }
    ]


def test_generate_personas_no_db_invokes_helper(monkeypatch, tmp_path):
    runner = CliRunner()
    artifacts_dir = tmp_path / "artifacts"
    input_dir = artifacts_dir / "inputs"
    input_dir.mkdir(parents=True)

    population_df = pd.DataFrame(
        {"persona_id": [1], "population": ["spain826"], "description": ["desc"]}
    )
    questionnaires_df = pd.DataFrame(
        {
            "personality_id": [1],
            "experiment_id": [101],
            "question": ["Q"],
            "answer": [True],
        }
    )
    population_path = input_dir / f"{pipeline.DEFAULT_POPULATION}.csv"
    questionnaires_path = (
        input_dir / f"{pipeline.DEFAULT_REFERENCE_QUESTIONNAIRES_TABLE}.csv"
    )
    population_df.to_csv(population_path, index=False)
    questionnaires_df.to_csv(questionnaires_path, index=False)

    recorded: dict = {}

    def fake_helper(**kwargs):
        recorded["kwargs"] = kwargs
        return "demo_generated_population", pd.DataFrame(
            {
                "ref_personality_id": [1],
                "population": ["generated"],
            }
        )

    monkeypatch.setattr("tools.pipeline.generate_personas_to_artifacts", fake_helper)

    result = runner.invoke(
        _cli_app,
        [
            "generate-personas",
            "demo",
            "--no-db",
            "--artifacts-dir",
            str(artifacts_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert recorded["kwargs"]["run_id"] == "demo"
    assert recorded["kwargs"]["models"] == [ModelID(pipeline.DEFAULT_MODEL)]
    assert recorded["kwargs"]["ref_population"] == pipeline.DEFAULT_POPULATION
    assert recorded["kwargs"]["population_df"].equals(population_df)
    assert recorded["kwargs"]["questionnaires_df"].equals(questionnaires_df)

    output_path = artifacts_dir / "outputs" / "demo" / "demo_generated_population.csv"
    assert output_path.exists()


def test_generate_borderline_no_db_invokes_helper(monkeypatch, tmp_path):
    runner = CliRunner()
    artifacts_dir = tmp_path / "artifacts"
    input_dir = artifacts_dir / "inputs"
    input_dir.mkdir(parents=True)

    population_df = pd.DataFrame(
        {"persona_id": [1], "population": ["spain826"], "description": ["desc"]}
    )
    questionnaires_df = pd.DataFrame(
        {
            "personality_id": [1],
            "experiment_id": [101],
            "question": ["Q"],
            "answer": [True],
        }
    )
    population_df.to_csv(input_dir / f"{pipeline.DEFAULT_POPULATION}.csv", index=False)
    questionnaires_df.to_csv(
        input_dir / f"{pipeline.DEFAULT_REFERENCE_QUESTIONNAIRES_TABLE}.csv", index=False
    )

    recorded: dict = {}

    def fake_helper(**kwargs):
        recorded["kwargs"] = kwargs
        return "demo_borderline_population", pd.DataFrame(
            {
                "ref_personality_id": [1],
                "population": ["borderline"],
            }
        )

    monkeypatch.setattr(
        "tools.pipeline.generate_borderline_to_artifacts", fake_helper
    )

    result = runner.invoke(
        _cli_app,
        [
            "generate-borderline",
            "demo",
            "--no-db",
            "--artifacts-dir",
            str(artifacts_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert recorded["kwargs"]["conditions"] == [PersonaGenerationCondition.MAX_N]
    assert recorded["kwargs"]["models"] == [ModelID(pipeline.DEFAULT_MODEL)]
    assert recorded["kwargs"]["population_df"].equals(population_df)
    assert recorded["kwargs"]["questionnaires_df"].equals(questionnaires_df)

    output_path = artifacts_dir / "outputs" / "demo" / "demo_borderline_population.csv"
    assert output_path.exists()
