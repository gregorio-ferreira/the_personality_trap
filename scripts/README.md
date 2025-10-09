# Utility Scripts

This directory contains utility scripts for database management, configuration validation, and environment setup.

## Available Scripts

### `validate_config.py`
**Purpose**: Validate runtime configuration before running experiments.

**Usage**:
```bash
# Validate OpenAI and database configuration
uv run python scripts/validate_config.py --require-openai --require-db

# Validate all providers (OpenAI, Bedrock, database)
uv run python scripts/validate_config.py --all

# Use custom config file
uv run python scripts/validate_config.py --config /path/to/config.yaml --all
```

**Checks**:
- OpenAI API key presence
- AWS Bedrock credentials (access keys or profile)
- PostgreSQL connection parameters
- Configuration file existence and readability

**Return Codes**:
- `0`: All required configuration present
- `1`: Missing required values
- `2`: Config file not found or unreadable

### `manage_schemas.py`
**Purpose**: Create and manage database schemas for different environments.

**Usage**:
```bash
# List all schemas in the database
uv run python scripts/manage_schemas.py list

# Create a new schema with all required tables
uv run python scripts/manage_schemas.py create --schema my_experiment

# Drop a schema (use with caution!)
uv run python scripts/manage_schemas.py drop --schema test_schema
```

**Features**:
- Automatically creates all tables from SQLModel definitions
- Sets up proper schema isolation
- Useful for creating separate workspaces for multiple researchers

### `manage_test_schemas.py`
**Purpose**: Create disposable test schemas for validation and experimentation.

**Usage**:
```bash
# Create a test schema with sample data
uv run python scripts/manage_test_schemas.py create --schema test_run_001

# List test schemas
uv run python scripts/manage_test_schemas.py list

# Clean up test schema
uv run python scripts/manage_test_schemas.py cleanup --schema test_run_001
```

**Use Cases**:
- Testing new experiment configurations without affecting production data
- Validating migrations before applying to main schema
- Creating isolated environments for debugging

### `restore_database.sh`
**Purpose**: Restore the research database from backup files.

**Usage**:
```bash
# Restore from compressed SQL backup
bash scripts/restore_database.sh

# The script uses environment variables for database connection:
export PERSONAS_PG__HOST=localhost
export PERSONAS_PG__DATABASE=personas
export PERSONAS_PG__USER=personas_user
export PERSONAS_PG__PASSWORD=your_password
```

**See Also**: [`docs/DATABASE_BACKUP.md`](../docs/DATABASE_BACKUP.md) for detailed restoration instructions.

## Development Notes

These scripts are maintained as part of the research replication package. They assume:
- Python environment configured with `uv sync` or `pip install -e .`
- PostgreSQL database accessible via environment variables or config.yaml
- Appropriate permissions for schema creation/modification

For questions or issues, refer to the main [README.md](../README.md) or open an issue on the project repository.
