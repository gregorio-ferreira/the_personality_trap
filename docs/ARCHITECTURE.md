# Experiments Runner & Evaluation Migration Report

This document summarises the backend migration that aligned the questionnaire
experiment flow with the new `personality_trap` schema and describes how to
operate, troubleshoot, and extend the implementation.  It supersedes the legacy
scripts under `experiments_original/` and should be used as the canonical
reference when running research experiments.

## 1. High-level architecture

The updated flow has three cooperating layers:

1. **Registration** – selects personas for a given model/population, builds the
   experiment group metadata, and inserts one row per persona repetition into
   `personality_trap.experiments_list` (`src/personas_backend/evaluate_questionnaire/registration.py`).
2. **Execution Runner** – reads pending experiments (`succeeded IS NULL`),
   impersonates personas via the configured model wrapper, persists the answers
   to `personality_trap.eval_questionnaires`, and marks the experiment as
   succeeded (`src/personas_backend/evaluate_questionnaire/runner.py`).
3. **CLI Orchestration** – exposes Typer commands that combine registration and
   execution (`tools/pipeline.py`).  The CLI now mirrors the original automation
   scripts while supporting dry-run and batching capabilities.

All components rely on the SQLModel models defined in
`src/personas_backend/db/models.py`.  The previous "parsed_answers" columns,
`answered_questionnaires` table, and CSV shims have been removed or replaced by
helpers that match the current schema.

## 2. Database integration updates

### 2.1 Experiment handler & repositories

* `ExperimentHandler.register_experiment` now builds `ExperimentsList` rows using
  only columns present in the new schema (e.g. `language_instructions`,
  `language_questionnaire`, `model_provider`, `model`, `population`,
  `personality_id`, `repo_sha`).  Deprecated payload keys are ignored.
* `ExperimentHandler.record_request_metadata_and_status` centralises the write
  path for request/response payloads and status updates.  It uses
  `ExperimentRequestMetadataRepository` to persist raw JSON blobs into
  `personality_trap.experiment_request_metadata` and then marks the experiment as
  succeeded while storing the optional `llm_explanation`.
* `ExperimentsRepository` gained helpers such as `mark_experiment_succeeded`,
  `update_llm_explanation`, and `check_and_mark_group_processed` to keep group
  state in sync.

### 2.2 Questionnaire persistence

`QuestionnaireBase.parse_valid_response` now:

* Normalises answers into a long-format DataFrame, applies any mapping/typing,
  and writes directly to `personality_trap.eval_questionnaires` via
  `_insert_db_eval_questionnaires`.
* Returns the explanation string so callers (e.g. the runner) can persist it on
  the experiment row.
* Delegates to `ExperimentHandler.record_request_metadata_and_status` to store
  request/response JSON and mark success when `auto_record=True` (the default).

### 2.3 Request metadata repository

`src/personas_backend/db/repositories/request_metadata_repo.py` now manages the
`experiment_request_metadata` table (and retains the file-backed version for
`--no-db` workflows).  Every runner invocation stores the raw request/response
payloads, aiding reproducibility and debugging.

## 3. Registration workflow

`register_questionnaire_experiments` orchestrates group/experiment creation:

1. `fetch_personas` loads personas filtered by `model` and `population` using the
   SQLModel session factory.  Input populations are normalised (deduplicated,
   empty strings removed) via `_normalize_populations`.
2. The questionnaire JSON template is obtained with `get_questionnaire_json` and
   the system role template defaults to
   `"You are required to adopt and impersonate the personality of the human described as follow:\n<row.description>"`.
3. Model configuration (temperature, top_p, provider, concrete model ID) is
   derived from `models_providers.models_config.get_models_config`.
4. For each persona repetition a new `ExperimentsList` record is created with the
   computed payload and the repository SHA (from `git rev-parse HEAD`).

The function returns the created experiment group ID and the list of experiment
IDs, enabling higher-level orchestration to track work.

Key optional arguments:

* `max_personas` / `max_repetitions` limit registrations for smoke tests.
* `experiment_description` overrides the default group description.
* `session_factory`, `experiment_group_handler`, and `experiment_handler` allow
  dependency injection (used heavily in the unit tests).

## 4. Execution runner

`run_pending_experiments` replaces the legacy parallel runner:

1. Loads all `succeeded IS NULL` rows (optionally filtering by group IDs or
   limiting via `batch_size`).
2. Resolves the persona backing each experiment.  The helper
   `_resolve_persona` matches on `population`, `personality_id`, and model name,
   with a fallback query to support legacy identifiers.
3. Builds a `PendingExperiment` dataclass containing the persona payload,
   questionnaire slug, model metadata, and experiment identifiers.
4. For each group, the runner creates a model wrapper (`get_model_wrapper`),
   obtains the questionnaire handler (`get_questionnaire_handler`), and executes
   experiments concurrently through a `ThreadPoolExecutor` (`max_workers`
   configurable).
5. Before each run `_clear_existing_answers` removes previous rows from
   `eval_questionnaires` to support re-processing.
6. After `QuestionnaireBase.process_experiment` returns, the runner stores
   request/response JSON, marks the experiment succeeded, and writes the
   explanation back to the database.
7. When all experiments in a group finish the group is marked concluded via
   `ExperimentGroupHandler.update_experiments_groups_conclude`.

Errors during execution are logged and do not halt other experiments; failed
runs remain with `succeeded=NULL` for later re-processing.

## 5. Typer CLI commands

`tools/pipeline.py` exposes a cohesive CLI for end-to-end operations:

* `register-experiments` – registers questionnaire runs for selected models and
  populations.  Supports `--dry-run`, `--max-personas`, `--max-repetitions`, and
  custom descriptions.
* `run-experiments` – processes pending experiments, optionally constrained to
  specific groups or batch sizes.  `--dry-run` prints the intended scope without
  contacting model providers.
* `run-evals` – convenience command that registers experiments (unless
  `--skip-registration` is set) and immediately executes them.  This command is a
  drop-in replacement for the historic pipeline entry point.

The existing commands (`init-db`, `seed-ref-pop`, `generate-personas`, `analyze`)
remain available.  See `docs/USAGE.md` for quick examples.

## 6. Notebook validation

`examples/questionnaires_evaluation.ipynb` demonstrates the new workflow end to
end:

1. Loads configuration and utility helpers.
2. Selects two personas per model/population combination.
3. Registers a fresh experiment group for each questionnaire.
4. Runs the pending experiments through the runner.
5. Displays summaries of `experiments_list` and `eval_questionnaires` to verify
   stored answers.

This notebook mirrors the persona-generation example and is intended for manual
QA or demos.

## 7. Automated tests

The following pytest suites cover the new functionality:

* `tests/test_evaluate_registration.py` – validates persona selection,
  experiment group creation, and repetition handling using in-memory SQLModel
  sessions and dependency injection.
* `tests/test_evaluate_runner.py` – exercises the runner, mocking the model
  wrapper to produce deterministic responses and asserting that experiments are
  marked succeeded, request metadata is stored, and answers land in
  `eval_questionnaires`.
* `tests/test_pipeline_cli.py` – uses Typer's CLI runner to ensure command-line
  argument wiring, dry-run paths, and error handling behave as expected.
* Supporting fixtures in `tests/conftest.py` spin up an in-memory database that
  mirrors the relevant `personality_trap` tables.

When developing locally run:

```bash
uv run pytest tests/test_evaluate_registration.py tests/test_evaluate_runner.py tests/test_pipeline_cli.py
```

to focus on the critical paths.

## 8. Troubleshooting tips

* **No pending experiments found** – ensure that registration succeeded and the
  relevant rows in `experiments_list` still have `succeeded=NULL`.  Use
  `tools/pipeline.py register-experiments --dry-run` to confirm persona counts.
* **Persona lookup warnings** – `_resolve_persona` logs a warning when it cannot
  find a matching persona.  Verify the persona exists with the specified model
  and population and that `personality_id` matches either `id` or
  `ref_personality_id`.
* **Schema mismatches** – the repositories expect the tables declared in
  `src/personas_backend/db/models.py`.  Run the Alembic migrations or consult
  `DATABASE_RELEASE_NOTES.md` if the database is missing columns.
* **Re-running experiments** – the runner clears existing rows from
  `eval_questionnaires` before invoking the questionnaire handler, so repeated
  runs will overwrite previous answers while preserving request metadata
  history.

## 9. Next steps

Potential follow-up enhancements:

* Add progress reporting/metrics for long-running experiment batches.
* Implement configurable backoff/ retry policies per model provider in the
  runner (currently inherited from `QuestionnaireBase`).
* Extend the notebook to include exploratory analysis of the collected answers
  (e.g. scoring, visualisations).
* Integrate experiment registration with persona-generation outputs to support
  fully automated end-to-end pipelines.

---

For additional context on configuration and usage, see `README.md`,
`docs/USAGE.md`, and the migration notes embedded in the source files mentioned
above.
