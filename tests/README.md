# Test Suite

This directory contains the test suite for the personas backend package. The tests verify core functionality and ensure the package works correctly for researchers replicating the study.

## Running Tests

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/test_config_manager.py -v

# Run tests matching a pattern
uv run pytest -k "config" -v

# Run fast tests only (skip slow integration tests)
make test-fast
```

## Test Categories

### Configuration Tests
- `test_config_manager.py` - Configuration loading and validation

### Database Tests
- `test_db_handler.py` - Database connection and handler functionality
- `test_evaluate_registration.py` - Experiment registration workflow
- `test_evaluate_runner.py` - Questionnaire execution workflow

### Questionnaire Tests
- `test_bigfive_questionnaire.py` - Big Five personality questionnaire implementation

### Model Integration Tests
- `test_models_wrapper.py` - LLM provider integration (no network calls)

### Persona Generation Tests
- `test_persona_generation_modes.py` - Persona generation logic
- `test_persona_generation_utils.py` - Helper utilities for persona generation

### Documentation Tests
- `test_readme_links.py` - Validates all links in README.md point to existing files

## Test Requirements

Most tests require:
- Valid configuration file (`config.yaml`) or environment variables
- PostgreSQL database access (configured via `PERSONAS_PG__*` variables)
- For LLM tests: API keys (though most tests use mocks)

Some tests can run without database access (see individual test files for `@pytest.mark.skip` decorators).

## Adding New Tests

When adding functionality to the package:
1. Add corresponding tests to verify the feature works
2. Follow existing test patterns (see `test_config_manager.py` for examples)
3. Use pytest fixtures from `conftest.py` for common setup
4. Run `make test` to ensure all tests pass before committing

## Continuous Integration

Tests are automatically run via pre-commit hooks. To set up:

```bash
uv pip install pre-commit
pre-commit install
```

## Test Configuration

Test behavior can be modified via `pyproject.toml` in the `[tool.pytest.ini_options]` section.
