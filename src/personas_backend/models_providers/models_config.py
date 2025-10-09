"""Static configuration for supported LLM models used in experiments."""

from typing import Any, Dict


def get_models_config(model: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.

    Args:
        model (str): Model identifier

    Returns:
        Dict[str, Any]: Model configuration

    Raises:
        KeyError: If the model identifier is unknown.
    """
    models_list = {
        "claude35sonnet": {
            "PERSONALITY_ID": 0,
            "TOP_P": 1.0,
            "TEMPERATURE": 1.0,
            "MAX_TOKENS": 1000,
            "MODEL_PROVIDER": "aws-bedrock",
            "POPULATION": "anthropic",
            "MODEL_ID": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "AWS_REGION": None,  # Use default region from config
            "ADDITIONAL_REQUEST_FIELDS": None,
        },
        "llama3170B": {
            "PERSONALITY_ID": 0,
            "TOP_P": 0.9,
            "TEMPERATURE": 0.5,
            "MAX_TOKENS": 1200,
            "MODEL_PROVIDER": "aws-bedrock",
            "POPULATION": "llama3",
            "MODEL_ID": "us.meta.llama3-1-70b-instruct-v1:0",
            "AWS_REGION": "us-east-2",  # LLaMA 3.1 70B requires us-east-2
            "ADDITIONAL_REQUEST_FIELDS": None,
        },
        "llama323B": {
            "PERSONALITY_ID": 0,
            "TOP_P": 0.9,
            "TEMPERATURE": 0.5,
            "MAX_TOKENS": 1000,
            "MODEL_PROVIDER": "aws-bedrock",
            "POPULATION": "llama3",
            "MODEL_ID": "eu.meta.llama3-2-3b-instruct-v1:0",
            "AWS_REGION": None,  # Use default region from config
            "ADDITIONAL_REQUEST_FIELDS": None,
        },
        "gpt4o": {
            "PERSONALITY_ID": 0,
            "MODEL_PROVIDER": "openai",
            "POPULATION": "openai",
            "MODEL_ID": "gpt-4o-2024-11-20",
            "TEMPERATURE": 1.0,
            "TOP_P": 1.0,
            "MAX_TOKENS": 1000,
            "LOGPROBS": False,
            "TOP_LOGPROBS": 5,
        },
        "gpt35": {
            "PERSONALITY_ID": 0,
            "MODEL_PROVIDER": "openai",
            "POPULATION": "openai",
            "MODEL_ID": "gpt-3.5-turbo-0125",
            "TEMPERATURE": 1.0,
            "TOP_P": 1.0,
            "MAX_TOKENS": 1000,
            "LOGPROBS": False,
            "TOP_LOGPROBS": 5,
        },
        "gpt4oold": {
            "PERSONALITY_ID": 0,
            "MODEL_PROVIDER": "openai",
            "POPULATION": "openai",
            "MODEL_ID": "gpt-4o-2024-05-13",
            "TEMPERATURE": 1.0,
            "TOP_P": 1.0,
            "MAX_TOKENS": 1000,
            "LOGPROBS": False,
            "TOP_LOGPROBS": 5,
        },
        "gpt4turbo": {
            "PERSONALITY_ID": 0,
            "MODEL_PROVIDER": "openai",
            "POPULATION": "openai",
            "MODEL_ID": "gpt-4-turbo-2024-04-09",
            "TEMPERATURE": 1.0,
            "TOP_P": 1.0,
            "MAX_TOKENS": 1000,
            "LOGPROBS": False,
            "TOP_LOGPROBS": 5,
        },
    }
    return models_list[model]
