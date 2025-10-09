# Colors for pretty output
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
BLUE   := $(shell tput -Txterm setaf 4)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

TARGET_MAX_CHAR_NUM=20

.DEFAULT_GOAL := help

## Show help
help:
	@echo ''
	@echo '${BLUE}Personas Backend - Development Commands${RESET}'
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo 'Targets:'
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
	helpMessage = match(lastLine, /^## (.*)/); \
	if (helpMessage) { \
	helpCommand = substr($$1, 0, index($$1, ":")-1); \
	helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
	printf "  ${YELLOW}%-$(TARGET_MAX_CHAR_NUM)s${RESET} ${GREEN}%s${RESET}\n", helpCommand, helpMessage; \
	} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)
	@echo ''

# =============================================================================
# Environment Setup
# =============================================================================

## Create virtual environment and install dependencies
setup:
	@echo "${GREEN}Creating virtual environment and installing dependencies...${RESET}"
	uv venv .venv
	uv pip install -e "."
	@echo "${GREEN}Virtual environment created at .venv${RESET}"

## Install development dependencies
dev-setup: setup
	@echo "${GREEN}Installing development dependencies...${RESET}"
	uv pip install -e ".[dev]"
	uv pip install jupyter ipykernel notebook
	@echo "${GREEN}Development dependencies installed${RESET}"

## Setup Jupyter notebook kernel
jupyter-setup:
	@echo "${GREEN}Setting up Jupyter kernel...${RESET}"
	source .venv/bin/activate && \
	python -m ipykernel install --user --name "personas_backend" --display-name "Python (Personas Backend)"
	@echo "${GREEN}Jupyter kernel 'Python (Personas Backend)' installed successfully!${RESET}"

# =============================================================================
# Code Quality & Formatting
# =============================================================================

## Format code with black and ruff
format:
	@echo "${GREEN}Formatting code with black and ruff...${RESET}"
	uv run black src/
	@if [ -d experiments ]; then uv run black experiments/; fi
	@if [ -d evaluations ]; then uv run black evaluations/; fi
	uv run ruff check src/ --fix
	@if [ -d experiments ]; then uv run ruff check experiments/ --fix; fi
	@if [ -d evaluations ]; then uv run ruff check evaluations/ --fix || true; fi

## Run comprehensive formatting and linting (recommended)
format-all:
	@echo "${GREEN}Running comprehensive formatting and linting...${RESET}"
	uv run black src/
	@if [ -d experiments ]; then uv run black experiments/; fi
	@if [ -d evaluations ]; then uv run black evaluations/; fi
	uv run ruff check src/ --fix
	@if [ -d experiments ]; then uv run ruff check experiments/ --fix; fi
	@if [ -d evaluations ]; then uv run ruff check evaluations/ || true; fi
	@echo "${GREEN}All formatting and linting complete!${RESET}"

## Run ruff linting only
lint:
	@echo "${GREEN}Running ruff linting...${RESET}"
	uv run ruff check src/
	@if [ -d experiments ]; then uv run ruff check experiments/; fi
	@if [ -d evaluations ]; then uv run ruff check evaluations/ || true; fi

## Run mypy type checking
typecheck:
	@echo "${GREEN}Running mypy static type checking...${RESET}"
	uv run mypy src/personas_backend/ --ignore-missing-imports --disallow-untyped-defs

## Run all code quality checks
check: lint typecheck
	@echo "${GREEN}All code quality checks completed!${RESET}"

# =============================================================================
# Testing
# =============================================================================

## Run all tests
test:
	@echo "${GREEN}Running tests...${RESET}"
	uv run pytest -v tests

## Run only configuration related tests
test-config:
	@echo "${GREEN}Running configuration tests...${RESET}"
	uv run pytest -v tests/test_config_manager.py

## Run model wrapper tests (no network)
test-models:
	@echo "${GREEN}Running model wrapper tests...${RESET}"
	uv run pytest -v -k models_wrapper

## Run fast unit tests (subset)
test-fast:
	@echo "${GREEN}Running fast test subset...${RESET}"
	uv run pytest -q -k 'config_manager or models_wrapper or db_handler'

## Run smoke test (config + db handler url + persona utils)
smoke:
	@echo "${GREEN}Running smoke tests...${RESET}"
	uv run pytest -q -k 'config_manager and compute_borderline'

# =============================================================================
# Database Management
# =============================================================================

## Start local Postgres via Docker Compose
db-up:
	@echo "${GREEN}Starting Postgres (docker compose)...${RESET}"
	docker compose up -d postgres
	docker compose ps

## Stop and remove Postgres container
db-down:
	@echo "${GREEN}Stopping Postgres (docker compose)...${RESET}"
	docker compose down

## Tail Postgres logs
db-logs:
	@echo "${GREEN}Tailing Postgres logs (Ctrl+C to exit)...${RESET}"
	docker compose logs -f postgres

## Run Alembic upgrade to latest
db-upgrade:
	@echo "${GREEN}Applying Alembic migrations...${RESET}"
	DATABASE_URL=$$(echo $${DATABASE_URL:-postgresql+psycopg2://personas:personas@localhost:5432/personas}) uv run alembic upgrade head

## Generate a new Alembic revision (usage: make db-revision msg="add new table")
db-revision:
	@echo "${GREEN}Creating Alembic revision...${RESET}"
	@if [ -z "$(msg)" ]; then echo "${YELLOW}Usage: make db-revision msg=\"description\"${RESET}" && exit 1; fi
	DATABASE_URL=$$(echo $${DATABASE_URL:-postgresql+psycopg2://personas:personas@localhost:5432/personas}) uv run alembic revision --autogenerate -m "$(msg)"

## Downgrade one revision
db-downgrade:
	@echo "${GREEN}Downgrading one Alembic revision...${RESET}"
	DATABASE_URL=$$(echo $${DATABASE_URL:-postgresql+psycopg2://personas:personas@localhost:5432/personas}) uv run alembic downgrade -1

## Check current database migration status
db-status:
	@echo "${GREEN}Checking database migration status...${RESET}"
	DATABASE_URL=$$(echo $${DATABASE_URL:-postgresql+psycopg2://personas:personas@localhost:5432/personas}) uv run alembic current

## Full database setup (start + migrate)
db-setup: db-up
	@echo "${GREEN}Waiting for database to be ready...${RESET}"
	sleep 5
	$(MAKE) db-upgrade
	@echo "${GREEN}Database setup complete!${RESET}"

## Restore complete dataset from backup (698,632 records)
db-restore:
	@echo "${GREEN}Restoring complete research dataset...${RESET}"
	@if [ ! -f backup/20251008/sql/personas_database_backup.sql.gz ]; then \
		echo "${YELLOW}Error: Backup file not found at backup/20251008/sql/personas_database_backup.sql.gz${RESET}" && exit 1; \
	fi
	@echo "${YELLOW}Restoring from SQL backup...${RESET}"
	gunzip -c backup/20251008/sql/personas_database_backup.sql.gz | \
		docker exec -i personas_postgres psql -U personas -d personas
	@echo "${YELLOW}Cleaning up old alembic version entries...${RESET}"
	docker exec personas_postgres psql -U personas -d personas -c \
		"DELETE FROM personality_trap.alembic_version WHERE version_num != '001_initial_schema';"
	@echo "${YELLOW}Creating materialized view...${RESET}"
	docker exec -i personas_postgres psql -U personas -d personas < backup/20251008/sql/create_experiments_evals_view.sql
	@echo "${GREEN}Dataset restoration complete!${RESET}"
	@echo "${YELLOW}Verifying restoration...${RESET}"
	@docker exec personas_postgres psql -U personas -d personas -c "SELECT COUNT(*) as total_personas FROM personality_trap.personas;"

## Restore dataset from CSV files (slower, more control)
db-restore-csv:
	@echo "${GREEN}Restoring dataset from CSV files...${RESET}"
	cd backup/20251008/scripts && uv run python restore_from_csv.py
	@echo "${YELLOW}Creating materialized view...${RESET}"
	docker exec -i personas_postgres psql -U personas -d personas < backup/20251008/sql/create_experiments_evals_view.sql
	@echo "${GREEN}CSV restoration complete!${RESET}"

# =============================================================================
# Development & Research
# =============================================================================

## Start Jupyter notebook server
notebook:
	@echo "${GREEN}Starting Jupyter notebook server...${RESET}"
	uv run jupyter notebook

## Generate personas with default models
generate-personas:
	@echo "${GREEN}Generating personas with default models...${RESET}"
	uv run python experiments/generate_personas.py

## Create full production run and shutdown
prod-run:
	@echo "${GREEN}Starting production persona generation run...${RESET}"
	bash scripts/run_personas_generation.sh

# =============================================================================
# Utility
# =============================================================================

## Clean up cache and build files
clean:
	@echo "${GREEN}Cleaning up cache and build files...${RESET}"
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name "*.egg" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name ".mypy_cache" -delete 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -r {} + 2>/dev/null || true
	@echo "${GREEN}Cleanup complete!${RESET}"

## Show project status (git, database, environment)
status:
	@echo "${BLUE}=== Project Status ===${RESET}"
	@echo "${YELLOW}Git Status:${RESET}"
	@git status --short || echo "Not a git repository"
	@echo ""
	@echo "${YELLOW}Database Status:${RESET}"
	@docker compose ps postgres 2>/dev/null || echo "Database not running"
	@echo ""
	@echo "${YELLOW}Virtual Environment:${RESET}"
	@if [ -d ".venv" ]; then echo "✓ Virtual environment exists"; else echo "✗ Virtual environment missing (run 'make setup')"; fi

# =============================================================================
# Configuration Utilities
# =============================================================================

## Validate configuration (OpenAI + DB + Bedrock)
config-validate:
	@echo "${GREEN}Validating configuration...${RESET}"
	uv run python scripts/validate_config.py --all

## Show resolved database port from configuration
config-show-port:
	@python - <<'PY'
	from personas_backend.utils.config import ConfigManager
	cfg = ConfigManager()
	print(f"Resolved DB host: {cfg.pg_config.host}")
	print(f"Resolved DB port: {cfg.pg_config.port}")
	PY

# =============================================================================
# Composite Commands
# =============================================================================

## Run full development setup
dev: dev-setup db-setup jupyter-setup
	@echo "${GREEN}Development environment ready!${RESET}"

## Run all quality checks and tests
ci: format-all check test
	@echo "${GREEN}All CI checks passed!${RESET}"

## Faster CI (lint + fast tests)
ci-fast: format-all test-fast
	@echo "${GREEN}Fast CI checks passed!${RESET}"

.PHONY: help setup dev-setup jupyter-setup format format-all lint typecheck check test \
	test-config test-models test-fast smoke ci-fast \
	notebook generate-personas prod-run db-up db-down db-logs db-upgrade \
	db-revision db-downgrade db-status db-setup db-restore db-restore-csv clean status dev ci
