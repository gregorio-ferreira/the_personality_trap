# Usage & Replication Guide

This document walks through the full workflow required to reproduce the experiments described in **The Personality Trap**.
Follow the steps below to provision the environment, populate the database, and execute the Typer-based research pipeline.

## 1. Install System Dependencies

The quickest path is to use the `make setup` target, which installs uv, synchronizes Python dependencies, and validates
PostgreSQL tooling.

```bash
make setup
```

If you prefer a manual setup, install uv and synchronize the environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --all-extras
```

## 2. Configure Credentials & Services

1. Copy the example configuration and fill in provider credentials:
   ```bash
   cp examples/example_config.yaml config.yaml
   ```
2. Export the required environment variables for LLM access and database connectivity (see [`README.md`](../README.md) for the
   complete list).
3. Ensure that PostgreSQL is accessible. You can start a local instance with Docker using `make db-up`.

## 3. Initialize or Restore the Database

Choose one of the following approaches:

- **Restore the published dataset (recommended for replication):**
  ```bash
  make db-upgrade    # Applies consolidated migration
  make db-restore    # Restores dataset/20251008/sql/personas_database_backup.sql.gz
  ```
  Detailed options and validation queries live in [`DATABASE_BACKUP.md`](DATABASE_BACKUP.md).

- **Run migrations on an empty database (for fresh experiments):**
  ```bash
  make db-upgrade
  ```

## 4. Explore the Pipeline CLI

The research workflow is orchestrated via `tools/pipeline.py`, built on top of [Typer](https://typer.tiangolo.com/). The CLI is
organized into subcommands that mirror the pipeline stages:

| Command | Purpose |
| --- | --- |
| `uv run python tools/pipeline.py init-db` | Create administrative tables and default schema entries. |
| `uv run python tools/pipeline.py seed-ref-pop` | Seed the baseline reference population (826 questionnaires). |
| `uv run python tools/pipeline.py generate-personas` | Generate AI personas using configured LLM providers. |
| `uv run python tools/pipeline.py register-experiments` | Register experimental configurations for questionnaire runs. |
| `uv run python tools/pipeline.py run-experiments` | Execute questionnaires against generated personas. |
| `uv run python tools/pipeline.py run-evals` | Produce evaluation outputs for downstream analysis. |
| `uv run python tools/pipeline.py analyze` | Generate summary statistics reproduced in the paper. |

List the available options for any subcommand with `--help`:

```bash
uv run python tools/pipeline.py generate-personas --help
```

## 5. End-to-End Replication Recipe

The following script recreates the results submitted with the paper once the database is restored:

```bash
# 1. Verify schema
uv run python scripts/manage_schemas.py list

# 2. Re-run evaluation pipeline to recompute metrics
uv run python tools/pipeline.py run-evals --population spain826 --model gpt4o

# 3. Produce the public report artefacts
uv run python tools/pipeline.py analyze
```

Repeat the `run-evals` invocation for each population/model combination you restored. Set `PERSONAS_TARGET_SCHEMA` before running these commands when working outside the default schema.

## 6. Running Ad-Hoc Experiments

1. **Create an isolated schema** for your experiment:
   ```bash
   export PERSONAS_TARGET_SCHEMA=researcher_xyz
   make db-upgrade
   ```
2. **Generate personas** with the desired model mix:
   ```bash
   uv run python tools/pipeline.py generate-personas --run-id researcher_xyz --models gpt4o claude35sonnet
   ```
3. **Register questionnaire experiments** and note the emitted group ID(s):
   ```bash
   uv run python tools/pipeline.py register-experiments --population spain826 --model gpt4o
   ```
4. **Execute questionnaires** for the registered groups:
   ```bash
   uv run python tools/pipeline.py run-experiments --group-id <GROUP_ID_FROM_STEP_3>
   ```
5. **Evaluate and analyze** the results:
   ```bash
   uv run python tools/pipeline.py run-evals --experiment-group-id <GROUP_ID_FROM_STEP_3> --skip-registration
   uv run python tools/pipeline.py analyze
   ```

Set `PERSONAS_TARGET_SCHEMA` before each command if you are working outside the default schema. Use `--no-db` with persona generation when you want CSV artifacts instead of database writes.

## 7. Data Exports & Reuse

- Add `--no-db` to persona or questionnaire commands to emit CSV artifacts under `artifacts/outputs/`.
- Execute `uv run python evaluations/population_analysis_comparissons.py` to generate CSV summaries referenced in the paper.
- For notebook-driven exploration, install the `notebooks` extra (`uv sync --extra notebooks`) and open `evaluations/*.ipynb`.

## 8. Troubleshooting Checklist

| Issue | Suggested Fix |
| --- | --- |
| CLI reports missing credentials | Confirm `config.yaml` values or re-export the `PERSONAS_*` environment variables. |
| Persona generation fails with API quota errors | Reduce `--count`, rotate providers, or schedule during off-peak hours. |
| `psycopg2` import errors | Install PostgreSQL client headers (`sudo apt-get install libpq-dev`). |
| Analysis output is empty | Ensure `run-evals` completed successfully and that you selected the correct schema/run identifiers. |

For complete dataset documentation, refer to [`../dataset_description.md`](../dataset_description.md).
