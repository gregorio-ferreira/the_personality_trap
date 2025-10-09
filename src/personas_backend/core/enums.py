"""Shared enum definitions for model identifiers and persona-generation conditions."""

from __future__ import annotations

from enum import Enum


class ModelID(str, Enum):
    """Canonical identifiers for supported language models."""

    GPT4O = "gpt4o"
    GPT35 = "gpt35"
    CLAUDE35_SONNET = "claude35sonnet"
    LLAMA3_23B = "llama323B"
    LLAMA3_170B = "llama3170B"
    GPT4O_OLD = "gpt4oold"
    GPT4_TURBO = "gpt4turbo"


class PersonaGenerationCondition(str, Enum):
    """Supported trait configurations for persona generation."""

    MIN_L = "min_L"
    MAX_N = "max_N"
    MAX_P = "max_P"
    MAX_ALL = "max_all"
    MIN_E_MAX_REST = "min_E_max_rest"
