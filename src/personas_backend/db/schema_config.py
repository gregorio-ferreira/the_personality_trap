"""
Schema configuration for database models.

This module provides a centralized way to configure database schemas
for different environments (production, testing, development).

Configuration is read exclusively from YAML via ConfigManager.
No environment variables are used for schema resolution.
"""

from typing import Optional

try:
    # Local import to avoid hard dependency during early boot; best-effort.
    from personas_backend.utils.config import ConfigManager  # type: ignore
except ImportError:  # pragma: no cover - defensive import guard
    ConfigManager = None  # type: ignore[misc,assignment]


def get_target_schema() -> str:
    """
    Get the target schema for database operations.

    Priority order:
    1. YAML config: schema.default_schema (production/migration workflows)
    2. Fallback: "personality_trap"

    Note: This function returns default_schema by default, which is used
    for Alembic migrations and production queries. For experimental workflows,
    use get_experimental_schema() which reads schema.target_schema from YAML.

    Returns:
        str: Target schema name
    """
    # Read from YAML configuration via ConfigManager if available
    if ConfigManager is not None:
        try:
            cfg = ConfigManager()
            sc = cfg.schema_config
            # Use default_schema (production schema for migrations/queries)
            return str(sc.default_schema)
        except (RuntimeError, OSError, ValueError):
            # Ignore configuration errors; fall through to default
            pass

    return "personality_trap"


def get_default_schema() -> str:
    """
    Get the default/production schema name.

    This is ALWAYS used by Alembic migrations to create the production schema.
    It reads from YAML config: schema.default_schema or falls back to
    "personality_trap".

    Returns:
        str: Default production schema name
    """
    if ConfigManager is not None:
        try:
            cfg = ConfigManager()
            sc = cfg.schema_config
            return str(sc.default_schema)
        except (RuntimeError, OSError, ValueError):
            pass

    return "personality_trap"


def get_experimental_schema() -> str:
    """
    Get the experimental/testing schema for runtime operations.

    Priority order:
    1. YAML config: schema.target_schema (experimental schema)
    2. YAML config: schema.default_schema (fallback)
    3. Fallback: "personality_trap"

    Use this in notebooks and experimental scripts that want to work
    in a separate testing schema without affecting production data.

    Returns:
        str: Experimental schema name
    """
    # Read from YAML configuration
    if ConfigManager is not None:
        try:
            cfg = ConfigManager()
            sc = cfg.schema_config
            # Prefer target_schema (experimental), fallback to default
            return str(sc.target_schema or sc.default_schema)
        except (RuntimeError, OSError, ValueError):
            pass

    return "personality_trap"


def get_table_args(schema: Optional[str] = None) -> dict:
    """
    Get SQLModel __table_args__ for the specified schema.

    Args:
        schema: Optional schema override. If None, uses get_target_schema()

    Returns:
        dict: Table arguments dictionary with schema configuration
    """
    target_schema = schema or get_target_schema()
    return {"schema": target_schema}


# Export the current target schema for easy access
TARGET_SCHEMA = get_target_schema()
