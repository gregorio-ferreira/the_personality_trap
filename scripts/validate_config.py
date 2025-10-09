#!/usr/bin/env python
"""Validate personas backend runtime configuration.

Checks presence of required keys in ~/.yaml (or provided --config path) and
environment overrides (PERSONAS_*). Exits non-zero if required values missing.

Usage examples:
  uv run python scripts/validate_config.py --require-openai --require-db
  uv run python scripts/validate_config.py --config /path/to/custom.yaml --all

Return codes:
  0 = success / all required present
  1 = missing required values
  2 = config file not found / unreadable
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

from personas_backend.utils.config import (
    BedrockConfig,
    ConfigManager,
    OpenAIConfig,
    PostgreSQLConfig,
)


def check_openai(cfg: OpenAIConfig, missing: List[str]) -> None:
    if not cfg.api_key:
        missing.append("openai.api_key (or PERSONAS_OPENAI__API_KEY)")


def check_bedrock(cfg: BedrockConfig, missing: List[str]) -> None:
    # Accept either credentials profile or access key pair
    if not (cfg.aws_credentials or (cfg.aws_access_key and cfg.aws_secret_key)):
        missing.append("bedrock credentials (aws_credentials or access/secret pair)")
    if not cfg.aws_region:
        missing.append("bedrock.aws_region (or PERSONAS_BEDROCK__AWS_REGION)")


def check_db(cfg: PostgreSQLConfig, missing: List[str]) -> None:
    if not cfg.host:
        missing.append("pg.host (or PERSONAS_PG__HOST)")
    if not cfg.database:
        missing.append("pg.database (or PERSONAS_PG__DATABASE)")
    if not cfg.user:
        missing.append("pg.user (or PERSONAS_PG__USER)")
    if not cfg.password:
        missing.append("pg.password (or PERSONAS_PG__PASSWORD)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate personas backend configuration")
    p.add_argument("--config", help="Path to YAML config (default ~/.yaml)")
    p.add_argument("--require-openai", action="store_true", help="Require OpenAI settings")
    p.add_argument(
        "--require-bedrock",
        action="store_true",
        help="Require Bedrock settings",
    )
    p.add_argument("--require-db", action="store_true", help="Require PostgreSQL settings")
    p.add_argument("--all", action="store_true", help="Shortcut to require all subsystems")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.all:
        args.require_openai = args.require_bedrock = args.require_db = True

    config_path = args.config or os.path.expanduser("~/.yaml")
    if not Path(config_path).is_file():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return 2

    try:
        mgr = ConfigManager(config_path=config_path)
    except OSError as exc:  # file or permission problems
        print(
            f"[ERROR] Failed to load config (OS error): {exc}",
            file=sys.stderr,
        )
        return 2
    except ValueError as exc:  # yaml parsing issues
        print(f"[ERROR] Failed to parse config: {exc}", file=sys.stderr)
        return 2

    missing: List[str] = []

    if args.require_openai:
        check_openai(mgr.openai_config, missing)
    if args.require_bedrock:
        check_bedrock(mgr.bedrock_config, missing)
    if args.require_db:
        check_db(mgr.pg_config, missing)

    if missing:
        print("[FAIL] Missing required configuration values:")
        for item in missing:
            print(f"  - {item}")
        print("\nHints:")
        print(
            ("  * Override with env vars: PERSONAS_SECTION__KEY (e.g." " PERSONAS_OPENAI__API_KEY)")
        )
        print("  * See .yaml.example for structure")
        return 1

    print("[OK] Configuration validation passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
