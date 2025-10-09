"""Typed wrapper around the OpenAI Chat Completions API."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI
from personas_backend.utils.config import ConfigManager  # type: ignore
from personas_backend.utils.post_processing_response import (  # type: ignore
    process_model_response,
    validate_schema,
)
from tenacity import retry, stop_after_attempt, wait_exponential


class OpenAIClientWrapper:
    """Thin orchestration layer for OpenAI chat completions."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        config_manager: Optional[ConfigManager] = None,
    ):
        """
        Initialize OpenAI client wrapper.

        Args:
            model_config (Dict[str, Any]): Model configuration
            logger (logging.Logger, optional): Logger instance
            config_manager (ConfigManager, optional): Configuration manager
        """
        # Get configuration
        self.config_manager = config_manager or ConfigManager()
        openai_config = self.config_manager.openai_config

        # Initialize OpenAI client
        self.client = OpenAI(
            timeout=60, api_key=openai_config.api_key, organization=openai_config.org_id
        )

        # Set up other attributes
        self.model_config = model_config
        self.logger = logger or logging.getLogger(__name__)
        self.model_id = self.model_config["MODEL_ID"]

    def generate_conversation(
        self,
        user_text_input: str,
        my_experiment: Dict[str, Any],
        system_role: Optional[str] = None,
        expected_schema: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
        """
        Generate a conversation with the model.

        Args:
            user_text_input (str): User input text
            my_experiment (Dict[str, Any]): Experiment configuration
            system_role (str, optional): System role prompt
            expected_schema (Dict[str, str], optional): Expected schema for validation

        Returns:
            Tuple[Dict[str, Any], Any, Dict[str, Any]]: Request, raw response, and validated response
        """
        request_json = self._generate_request_json(
            user_text_input=user_text_input,
            my_experiment=my_experiment,
            system_role=system_role,
        )
        self.logger.info(f"Generating message with model {request_json['model']}")

        try:
            # Call API with retry
            response = self._call_api_with_retry(request_json)

            text_content = self.extract_message_content(response)

            # First try direct JSON parsing since OpenAI can return well-formed JSON
            try:
                parsed_json = json.loads(text_content)
                self.logger.info("Response is valid JSON, skipping additional processing")

                # Still validate against schema if provided
                if expected_schema:
                    parsed_json = validate_schema(parsed_json, expected_schema, self.logger)
            except json.JSONDecodeError:
                # If direct parsing fails, use the full processing pipeline
                self.logger.info("Response is not valid JSON, applying full processing")
                parsed_json = process_model_response(text_content, expected_schema, self.logger)

            return request_json, response, parsed_json
        except Exception as e:
            self.logger.error(f"Error in generate_conversation: {str(e)}")
            return request_json, None, {"error": str(e)}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _call_api_with_retry(self, request_json: Dict[str, Any]) -> Any:
        """Helper method with retry logic for API calls"""
        return self.client.chat.completions.create(
            model=request_json["model"],
            messages=request_json["messages"],
            temperature=request_json["temperature"],
            response_format=request_json["response_format"],
            top_p=request_json.get("top_p", 1.0),
            max_tokens=request_json.get("max_tokens", 1000),
            logprobs=request_json.get("logprobs"),
            top_logprobs=(
                request_json.get("top_logprobs") if request_json.get("logprobs") else None
            ),
        )

    def extract_message_content(self, response: Any) -> str:
        """
        Extract the raw text content from an OpenAI API response.

        Args:
            response (Any): The complete API response from OpenAI

        Returns:
            str: The extracted text content
        """
        try:
            response_dict = response.model_dump()

            if response_dict["choices"][0]["finish_reason"] == "stop":
                self.logger.info("Stop reason: valid")
                message = response_dict["choices"][0]["message"]
                content = message.get("content", "")

                # Log token usage
                usage = response_dict["usage"]
                self.logger.info(f"Input tokens: {usage['prompt_tokens']}")
                self.logger.info(f"Output tokens: {usage['completion_tokens']}")
                self.logger.info(f"Total tokens: {usage['total_tokens']}")

                if content:
                    self.logger.info("Content is valid")
                    self.logger.debug(f"Output: {content[:100]}...")
                    return str(content)

            return ""
        except Exception as e:
            self.logger.error(f"Error extracting message content: {e}")
            return ""

    def _generate_request_json(
        self,
        user_text_input: str,
        my_experiment: Dict[str, Any],
        system_role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate request JSON for OpenAI.

        Args:
            user_text_input (str): User input text
            my_experiment (Dict[str, Any]): Experiment configuration
            system_role (str, optional): System role prompt

        Returns:
            Dict[str, Any]: Request JSON
        """
        if system_role is None:
            system_role = self._model_system_role()

        mycontext = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_text_input},
        ]

        request_json = {
            "model": self.model_id,
            "response_format": {"type": "json_object"},
            "messages": mycontext,
            "temperature": self.model_config["TEMPERATURE"],
            "top_p": self.model_config["TOP_P"],
            "max_tokens": self.model_config["MAX_TOKENS"],
        }

        logprobs = self.model_config.get("LOGPROBS", None)
        if logprobs:
            request_json["logprobs"] = self.model_config["LOGPROBS"]
            request_json["top_logprobs"] = self.model_config["TOP_LOGPROBS"]

        return request_json

    def get_validated_response(self, response: Any) -> Dict[str, Any]:
        """
        Extract and validate response from OpenAI.

        Args:
            response (Any): OpenAI response

        Returns:
            Dict[str, Any]: Validated response content
        """
        output = self.extract_message_content(response)

        # Use the centralized function for validation
        result = process_model_response(output, None, self.logger)
        # Ensure we return a dictionary
        if not isinstance(result, dict):
            self.logger.warning(
                "process_model_response did not return a dictionary, returning empty dict"
            )
            return {"error": "Failed to process model response"}
        return result

    def _model_system_role(self) -> str:
        """
        Get system role prompt for the model.

        Returns:
            str: System role prompt
        """
        cutoff_date = "2021-09"
        architecture = "GPT-3.5"

        # Map model IDs to their details
        model_details = {
            "gpt-3.5-turbo-0125": {"cutoff": "2021-09", "arch": "GPT-3.5"},
            "gpt-4-0613": {"cutoff": "2021-09", "arch": "GPT-4"},
            "gpt-4o-2024-05-13": {
                "cutoff": "2023-10",
                "arch": 'GPT-4o ("o" for "omni")',
            },
            "gpt-4o-2024-08-06": {
                "cutoff": "2024-08",
                "arch": 'GPT-4o ("o" for "omni")',
            },
            "gpt-4-0125-preview": {"cutoff": "2023-12", "arch": "GPT-4"},
            "gpt-4-turbo-2024-04-09": {"cutoff": "2024-04", "arch": "GPT-4"},
            "gpt-4o-2024-11-20": {
                "cutoff": "2024-10",
                "arch": 'GPT-4o ("o" for "omni")',
            },
        }

        # Get details for the current model or use defaults
        details = model_details.get(self.model_id, {"cutoff": cutoff_date, "arch": architecture})

        return f"""You are ChatGPT, a large language model trained by OpenAI, based on the {details['arch']} architecture.
Knowledge cutoff: {details['cutoff']}
Current date: {datetime.now().date()}"""
