## Copilot agent guide for this repository

This repository contains a Python 3.12+ backend for generating AI personas and evaluating psychological profiles. It uses uv for dependency management, SQLModel + Alembic for PostgreSQL migrations, and pytest + Ruff + Black + MyPy for quality gates. Keep changes small, typed, and aligned with repository patterns.

### What to read first
- Repository-wide standards: `.github/copilot-instructions.md` (authoritative overview and patterns)
- Tooling: `pyproject.toml` (formatters, linters, type-checker settings) and `Makefile` (single-source of truth for commands)

## Build, test, and validate
Use the Makefile targets below. Prefer these over ad-hoc commands. All commands assume Linux bash and uv.

- Bootstrap local dev (venv, deps, DB, Jupyter):
  - make dev
- Install only project deps:
  - make setup
- Install dev tooling (pytest, ruff, black, mypy):
  - make dev-setup
- Format and lint (fixes applied where safe):
  - make format-all
- Lint only:
  - make lint
- Type check:
  - make typecheck
- Run tests:
  - make test           # full suite
  - make test-fast      # quick subset for iteration
  - make smoke          # minimal smoke
- All CI checks (format + lint/typecheck + tests):
  - make ci             # full
  - make ci-fast        # faster subset

Database (local via Docker):
- Start Postgres: make db-up
- Apply migrations: make db-upgrade (or make db-setup for start+upgrade)
- Status/logs: make db-status, make db-logs

Notes
- uv is the package manager. Avoid pip/poetry directly; prefer uv and Make targets.
- Tests should pass without network access. Stub or skip external calls. Do not embed secrets.
- If a DB is needed, use make db-setup and Alembic migration flow; do not write raw SQL for schema changes.

## Repository layout and patterns
- src/personas_backend/: core backend modules
  - db/: SQLModel models, handlers, repositories (use repository pattern)
  - connectors/: LLM providers (OpenAI, Bedrock, local) with tenacity retries
  - population/: population generation, demographics
  - questionnaires/: Big Five logic and scoring
  - utils/: configuration, logging, formatting
- src/evaluations/: analysis utilities and tables
- alembic/: migration scripts and env
- scripts/: CLI and maintenance utilities
- examples/ and docs/: user documentation and notebooks (no LaTeX export in repo)

Key conventions
- Configuration: YAML with env overrides using PERSONAS_SECTION__KEY (e.g., PERSONAS_OPENAI__API_KEY). Never commit secrets.
- Database: SQLModel with explicit schema in __table_args__; use Alembic for migrations; repository pattern over raw SQL.
- LLM calls: use connectors with retry (tenacity) and json-repair; avoid network in tests.
- Analysis: produce DataFrames/CSV; LaTeX generation is out-of-scope.

## Coding standards (from pyproject.toml)
- Python: 3.12+
- Black: line-length 100
- Ruff: target-version py312; rules E,F,I,B with repository-specific ignores
- MyPy: disallow untyped defs; strict optional; ignore_missing_imports=True
- Style: type hints everywhere; f-strings; keep public APIs stable; small focused changes

Acceptance criteria for changes
- Format and lint clean: make format-all and make check pass
- Tests pass locally: make test (or at minimum make smoke and make test-fast during iteration)
- No secrets or live-network dependencies in tests
- DB migrations (if schema changes): Alembic revision via make db-revision msg="..." and upgrade tested
- Adhere to repository patterns (repositories, SQLModel models with schema, typed functions)

## Typical workflows
Small code change
1) make dev-setup (first time) or ensure .venv exists
2) Implement change with types and unit tests (prefer tests under tests/ mirroring module paths)
3) Run make format-all, then make test-fast; iterate until green
4) Run make check and make test; ensure all pass

Adding a DB model or schema change
1) Define SQLModel with __table_args__ schema (e.g., {"schema": "personality_trap"})
2) Generate migration: make db-revision msg="describe change"
3) Apply migrations: make db-upgrade; verify models work
4) Add repository methods; write unit/integration tests without external network

Working with LLM connectors
- Use existing connector abstractions; keep retries via tenacity
- Parse responses robustly (json-repair where relevant); do not add live calls to test suite

## Guardrails and pitfalls
- Do not reformat unrelated files; keep diffs minimal and focused
- Do not introduce LaTeX exports; keep analysis to DataFrames/CSV
- Do not hardcode credentials or endpoints; use ConfigManager and env overrides
- Prefer repository pattern over raw SQL; migrations via Alembic only
- Respect line length 100 (pyproject is authoritative; earlier docs may mention 88)

## Useful Make targets at a glance
- Setup: make dev, make setup, make dev-setup
- Quality: make format-all, make lint, make typecheck, make check
- Tests: make test, make test-fast, make smoke
- DB: make db-up, make db-upgrade, make db-status, make db-logs, make db-revision msg="..."
- Utilities: make status, make clean, make notebook

## PR readiness checklist
- [ ] Code formatted and linted (Black/Ruff) and typed (MyPy)
- [ ] Tests updated/added and passing; no network/secrets in tests
- [ ] Migrations present and applied if schema changed
- [ ] Follows repository patterns (config, repositories, models, utils)
- [ ] Documentation touched where helpful (docs/ or examples/)

## Alembic and schema isolation

This repo supports per-schema migrations so the same `models.py` can be used across isolated schemas (e.g., production vs test). Key points:

- env.py wiring
  - `alembic/env.py` imports `SQLModel.metadata` and configures Alembic with a schema-aware version table:
    - `version_table_schema = schema_name` and `version_table = "alembic_version"`.
  - The target schema is read from `config.attributes["target_schema"]` when migrations run; if not provided, it falls back to `get_default_schema()` from `schema_config.py`.
  - In online mode, the script ensures the schema exists: `CREATE SCHEMA IF NOT EXISTS <schema_name>`.

- Choosing the schema
  - `src/personas_backend/db/schema_config.py` exposes:
    - `get_default_schema()` → production schema from YAML (fallback `personality_trap`).
    - `get_experimental_schema()` → preferred testing/experimental schema from YAML `schema.target_schema`, falling back to default.
    - `get_table_args(schema)` used by all SQLModel tables to set `__table_args__ = {"schema": <target>}`.

- Running migrations for a specific schema
  - Preferred: use the repository helper which passes `target_schema` into Alembic:
    - `DatabaseHandler.run_migrations(schema="my_schema", revision="head")`
      - This sets `alembic.config.attributes["target_schema"] = "my_schema"`, then runs `alembic upgrade`.
  - If no `schema` is given, `run_migrations()` uses YAML: `schema.target_schema` → `schema.default_schema`.

- Creating/listing/dropping schemas
  - Utility scripts are provided under `scripts/`:
    - `scripts/manage_schemas.py`:
      - `create <schema>` → creates schema and tables by temporarily setting `PERSONAS_TARGET_SCHEMA` and calling `SQLModel.metadata.create_all()` (isolation from production).
      - `drop <schema> [--cascade]` → drops a non-protected schema.
      - `list` → prints available schemas.
    - `scripts/manage_test_schemas.py`:
      - `create <schema> --personality-ids="1,2,3"` → clones structure from `personality_trap` and seeds selected rows for validation.
      - `drop <schema>` and `list_schemas` helpers.

- Model reuse across schemas
  - All table classes call `get_table_args()` so the same Python models are reused across schemas without duplication.
  - When switching schemas, ensure your YAML config (or explicit `schema=` arg) points to the intended target before running migrations or tests that touch the DB.


