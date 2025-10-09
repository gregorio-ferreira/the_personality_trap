"""Configuration loader supporting YAML files and environment overrides."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, cast

import yaml  # type: ignore[import-untyped]

ENV_PREFIX = "PERSONAS_"


@dataclass
class OpenAIConfig:
    api_key: str
    api_key_pain_narratives: str
    org_id: str


@dataclass
class BedrockConfig:
    aws_credentials: str
    aws_access_key: str
    aws_secret_key: str
    aws_region: str


@dataclass
class PostgreSQLConfig:
    password: str
    host: str
    database: str
    user: str
    port: int


@dataclass
class SchemaConfig:
    """Database schema configuration."""

    default_schema: str
    target_schema: str


class ConfigManager:
    """Load configuration from `.yaml` plus `PERSONAS_*` environment overrides."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize configuration manager.

        Load YAML configuration then apply environment variable overrides
        using the pattern PERSONAS_SECTION__KEY. Underscores are converted
        to hyphens to allow both styles in YAML keys.

        Args:
            config_path: Path to config file (defaults to .yaml in root)
            logger: Optional logger instance
        """
        if config_path is None:
            # Find project root by looking for pyproject.toml
            current_file = Path(__file__)
            project_root = current_file
            while project_root.parent != project_root:
                if (project_root / "pyproject.toml").exists():
                    break
                project_root = project_root.parent
            config_path = str(project_root / ".yaml")

        self.logger = logger
        if self.logger:
            self.logger.debug(f"Loading configuration from {config_path}")

        with open(config_path, encoding="utf-8") as f:
            loaded = yaml.load(f, Loader=yaml.FullLoader)

        if isinstance(loaded, MutableMapping):
            raw: Dict[str, Any] = cast(Dict[str, Any], dict(loaded))
        else:
            raw = {}

        self.config: Dict[str, Any] = raw

        # Allow override from environment variables (flat mapping)
        # e.g. PERSONAS_OPENAI__API_KEY overrides openai.api_key
        for key, value in os.environ.items():
            if not key.startswith(ENV_PREFIX):
                continue
            path = key[len(ENV_PREFIX) :].lower().split("__")
            cursor: Dict[str, Any] = self.config
            for part in path[:-1]:
                normalised = part.replace("_", "-")
                next_cursor = cursor.setdefault(normalised, {})
                if not isinstance(next_cursor, dict):
                    next_cursor = {}
                    cursor[normalised] = next_cursor
                cursor = cast(Dict[str, Any], next_cursor)
            cursor[path[-1].replace("_", "-")] = value

        # NOTE: Schema configuration is handled via schema_config property.
        # All schema resolution is done through get_experimental_schema() and
        # get_default_schema() functions in db/schema_config.py, which read
        # directly from this ConfigManager.
        #
        # No environment variables are used for schema configuration.
        # To use experimental/testing schema in notebooks:
        # - Configure schema.target_schema in .yaml, OR
        # - Use get_experimental_schema() instead of get_target_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, *path: str, default: Any = None) -> Any:
        """Retrieve a nested configuration value.

        Example: cfg.get("openai", "api-key")
        Hyphen / underscore differences are normalized.
        """
        cursor: Any = self.config
        for part in path:
            if not isinstance(cursor, dict):
                return default
            # Try several key variants to be tolerant to style.
            variants = {
                part,
                part.replace("_", "-"),
                part.replace("-", "_"),
            }
            found = None
            for candidate in variants:
                if candidate in cursor:
                    found = candidate
                    break
            if found is None:
                return default
            cursor = cursor[found]
        return cursor

    @property
    def openai_config(self) -> OpenAIConfig:
        """Get OpenAI configuration (values may be empty strings)."""
        openai_cfg = cast(Dict[str, Any], self.config.get("openai", {}))
        return OpenAIConfig(
            api_key=openai_cfg.get("api-key") or openai_cfg.get("api_key", ""),
            api_key_pain_narratives=openai_cfg.get("api-key-pain-narratives")
            or openai_cfg.get("api_key_pain_narratives", ""),
            org_id=openai_cfg.get("org-id") or openai_cfg.get("org_id", ""),
        )

    @property
    def bedrock_config(self) -> BedrockConfig:
        """Get Bedrock configuration (empty strings if unset)."""
        bedrock_cfg = cast(Dict[str, Any], self.config.get("bedrock", {}))
        return BedrockConfig(
            aws_credentials=bedrock_cfg.get("aws_credentials", ""),
            aws_access_key=bedrock_cfg.get("aws_access_key", ""),
            aws_secret_key=bedrock_cfg.get("aws_secret_key", ""),
            aws_region=bedrock_cfg.get("aws_region", ""),
        )

    @property
    def pg_config(self) -> PostgreSQLConfig:
        """Get PostgreSQL configuration (includes 'port')."""
        pg_cfg = cast(Dict[str, Any], self.config.get("pg", {}))
        # Port may appear as int or string; normalize.
        raw_port = pg_cfg.get("port", 5432)
        try:
            port = int(raw_port)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            port = 5432
        return PostgreSQLConfig(
            password=pg_cfg.get("password", ""),
            host=pg_cfg.get("host", "localhost"),
            database=pg_cfg.get("database", "personas"),
            user=pg_cfg.get("user", "personas"),
            port=port,
        )

    @property
    def schema_config(self) -> SchemaConfig:
        """Get database schema configuration."""
        schema_cfg = cast(Dict[str, Any], self.config.get("schema", {}))
        default_schema = schema_cfg.get("default_schema", "personality_trap")
        # Prefer explicit target_schema from config; fall back to
        # default_schema if missing
        target_schema = schema_cfg.get("target_schema", default_schema)
        return SchemaConfig(
            default_schema=default_schema,
            target_schema=target_schema,
        )
