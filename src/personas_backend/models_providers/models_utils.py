"""Helpers for resolving configured LLM provider wrappers."""

import logging
from typing import Any, Dict, Optional, Union

from personas_backend.models_providers.aws_bedrock_client import (
    BedrockClientWrapper,  # type: ignore
)
from personas_backend.models_providers.models_config import (
    get_models_config,  # type: ignore
)
from personas_backend.models_providers.openai_client import (
    OpenAIClientWrapper,  # type: ignore
)
from personas_backend.utils.config import ConfigManager  # type: ignore


def get_model_wrapper(
    model: str,
    logger: logging.Logger,
    config_manager: Optional[ConfigManager] = None,
) -> Union[OpenAIClientWrapper, BedrockClientWrapper]:
    """
    Get the appropriate model wrapper based on the model name.

    Args:
        model (str): Model name

    Returns:
        Union[OpenAIClientWrapper, BedrockClientWrapper]: The appropriate model wrapper

    Raises:
        ValueError: If the configured provider is not supported.
    """
    model_config = get_models_config(model)
    cfg = config_manager or ConfigManager(logger=logger)
    provider = model_config.get("MODEL_PROVIDER")

    if provider == "openai":
        openai_wrapper = OpenAIClientWrapper(
            model_config=model_config, logger=logger, config_manager=cfg
        )
        return openai_wrapper
    elif provider == "aws-bedrock":
        bedrock_wrapper = BedrockClientWrapper(
            model_config=model_config, logger=logger, config_manager=cfg
        )
        return bedrock_wrapper
    else:
        raise ValueError(
            ("Unsupported model provider %s (use: openai or " "aws-bedrock)" % provider)
        )


def get_model_name_by_id(model_id: str) -> str:
    """
    Get model name from model ID.

    Args:
        model_id (str): Model ID

    Returns:
        str: Model name
    """
    models_id = {
        "anthropic.claude-3-5-sonnet-20240620-v1:0": "claude35sonnet",
        "eu.meta.llama3-2-3b-instruct-v1:0": "llama323B",
        "us.meta.llama3-1-70b-instruct-v1:0": "llama3170B",
        "gpt-4o-2024-11-20": "gpt4o",
        "gpt-3.5-turbo-0125": "gpt35",
        "gpt-4o-2024-05-13": "gpt4oold",
        "gpt-4-turbo-2024-04-09": "gpt4turbo",
    }
    return models_id[model_id]


def get_retry_config() -> Dict[str, Any]:
    """
    Get retry configuration.

    Returns:
        Dict[str, Any]: Retry configuration
    """
    retry_config = {
        "max_retries": 3,
        "success": False,
        "attempt": 0,
        "response": None,
    }

    return retry_config


def get_models_by_provider(provider: str) -> Dict[str, str]:
    """
    Get all models for a specific provider.

    Args:
        provider (str): Provider name ('openai' or 'aws-bedrock')

    Returns:
        Dict[str, str]: Dictionary of model names to model IDs
    """
    result = {}
    for model_name in get_all_models():
        config = get_models_config(model_name)
        if config["MODEL_PROVIDER"] == provider:
            result[model_name] = config["MODEL_ID"]
    return result


def get_all_models() -> list:
    """
    Get all available model names.

    Returns:
        list: List of model names
    """
    return [
        "claude35sonnet",
        "llama323B",
        "llama3170B",
        "gpt4o",
        "gpt35",
        "gpt4oold",
        "gpt4turbo",
    ]
