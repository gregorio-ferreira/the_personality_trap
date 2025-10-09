"""Personas backend package exports.

Centralized schema access:
- ACTIVE_SCHEMA: production schema (personality_trap) - use in evaluations
- get_default_schema: production schema (personality_trap) - use in Alembic
- get_experimental_schema: experimental schema for testing workflows
- get_table_args: helper for SQLModel __table_args__

Schema Usage Guidelines:
- Alembic migrations: Use get_default_schema() → reads YAML
  schema.default_schema
- Production queries/evaluations: Use ACTIVE_SCHEMA
- Experimental notebooks: Use get_experimental_schema() → reads YAML
  schema.target_schema
- All schema configuration is read from .yaml file via ConfigManager
"""

from .db.schema_config import (
    TARGET_SCHEMA as ACTIVE_SCHEMA,
)
from .db.schema_config import (
    get_default_schema,
    get_experimental_schema,
    get_table_args,
)

__all__ = [
    "ACTIVE_SCHEMA",
    "get_default_schema",
    "get_experimental_schema",
    "get_table_args",
]
