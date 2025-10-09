"""Typed wrapper around the AWS Bedrock Converse API."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, cast

import boto3  # type: ignore
from personas_backend.utils.config import ConfigManager  # type: ignore
from personas_backend.utils.post_processing_response import (  # type: ignore
    process_model_response,
)
from tenacity import retry, stop_after_attempt, wait_exponential


class BedrockClientWrapper:
    """High-level interface for invoking Bedrock models with retries."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        config_manager: Optional[ConfigManager] = None,
    ):
        """
        Initialize Bedrock client wrapper.

        Args:
            model_config (Dict[str, Any]): Model configuration
            logger (logging.Logger, optional): Logger instance
            config_manager (ConfigManager, optional): Configuration manager
        """
        # self._generate_request_json = _generate_request_json
        # self._call_api_with_retry = _call_api_with_retry
        # Get configuration
        self.config_manager = config_manager or ConfigManager()
        bedrock_config = self.config_manager.bedrock_config
        self.logger = logger or logging.getLogger(__name__)

        # Set up session and client
        try:
            # Session configuration
            session_kwargs = {}

            # If explicit credentials are provided, use them
            if bedrock_config.aws_credentials:
                self.logger.info(f"Using AWS profile: {bedrock_config.aws_credentials}")
                session_kwargs["profile_name"] = bedrock_config.aws_credentials

            elif bedrock_config.aws_access_key and bedrock_config.aws_secret_key:
                self.logger.info("Using explicit AWS credentials")
                session_kwargs.update(
                    {
                        "aws_access_key_id": bedrock_config.aws_access_key,
                        "aws_secret_access_key": bedrock_config.aws_secret_key,
                    }
                )

            else:
                self.logger.info("Using default AWS credential chain")

            self.session = boto3.session.Session(**session_kwargs)

            # Validate that we have credentials
            credentials = self.session.get_credentials()
            if credentials is None:
                raise ValueError("No AWS credentials found")

            # Determine region from model config or use default
            model_region = model_config.get("AWS_REGION")
            if model_region:
                self.logger.info(f"Using model-specific region: {model_region}")
                bedrock_runtime_region = model_region
            else:
                default_region = bedrock_config.aws_region
                self.logger.info(f"Using default AWS region: {default_region}")
                bedrock_runtime_region = default_region

            model_id = model_config.get("MODEL_ID")
            self.logger.info(
                f"Creating Bedrock client for model {model_id} "
                f"in region: {bedrock_runtime_region}"
            )
            self.bedrock_runtime = self.session.client(
                service_name="bedrock-runtime", region_name=bedrock_runtime_region
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize AWS Bedrock client: {str(e)}")
            raise

        # Set up other attributes
        self.model_config = model_config
        self.model_id = self.model_config["MODEL_ID"]

    def generate_conversation(
        self,
        user_text_input: str,
        my_experiment: Dict[str, Any],
        system_role: Optional[str] = None,
        expected_schema: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Generate a conversation with the model and validate the response.

        Args:
            user_text_input (str): User input text
            my_experiment (Dict[str, Any]): Experiment configuration
            system_role (str, optional): System role prompt
            expected_schema (Dict[str, str], optional): Expected schema for validation

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]: Request, raw response, and validated response
        """
        # Create request
        request_json = self._generate_request_json(
            user_text_input=user_text_input,
            my_experiment=my_experiment,
            system_role=system_role,
        )

        self.logger.info(f"Generating message with model {request_json['modelId']}")

        try:
            # Call API with retry
            logging.info(f"Request JSON: {json.dumps(request_json, indent=2)}")
            self.logger.info(f"Calling Bedrock API with model {request_json['modelId']}")

            response = self._call_api_with_retry(request_json)
            logging.info(f"Response JSON: {json.dumps(response, indent=2)}")
            self.logger.info("Received response from Bedrock API")

            # Extract text content from response
            text_content = self.extract_message_content(response)

            # Process the response using the common utility
            parsed_json = process_model_response(
                response_text=text_content,
                expected_schema=expected_schema,
                logger=self.logger,
            )

            return request_json, response, parsed_json
        except Exception as e:
            self.logger.error(f"Error in generate_conversation: {str(e)}")
            return request_json, {}, {"error": str(e)}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _call_api_with_retry(self, request_json: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method with retry logic for API calls"""

        # Prepare arguments for the API call
        call_args = {
            "modelId": request_json["modelId"],
            "messages": request_json["messages"],
            "system": request_json["system"],
            "inferenceConfig": request_json["inferenceConfig"],
        }

        # Only add additionalModelRequestFields if present in request
        if "additionalModelRequestFields" in request_json:
            call_args["additionalModelRequestFields"] = request_json["additionalModelRequestFields"]

        result = self.bedrock_runtime.converse(**call_args)

        return cast(Dict[str, Any], result)

    def _generate_request_json(
        self,
        user_text_input: str,
        my_experiment: Dict[str, Any],
        system_role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate request JSON for AWS Bedrock.

        Args:
            user_text_input (str): User input text
            my_experiment (Dict[str, Any]): Experiment configuration
            system_role (str, optional): System role prompt

        Returns:
            Dict[str, Any]: Request JSON
        """
        if system_role is None:
            system_role = self._model_system_role()

        messages = []
        messages.append({"role": "user", "content": [{"text": user_text_input}]})

        request_json = {
            "modelId": self.model_id,
            "messages": messages,
            "system": [{"text": system_role}],
            "inferenceConfig": {
                "temperature": self.model_config["TEMPERATURE"],
                "maxTokens": self.model_config["MAX_TOKENS"],
                "topP": self.model_config["TOP_P"],
            },
        }

        # Only add additionalModelRequestFields if they exist and are not None
        additional_fields = self.model_config.get("ADDITIONAL_REQUEST_FIELDS")
        if additional_fields is not None:
            request_json["additionalModelRequestFields"] = additional_fields
        return request_json

    def extract_message_content(self, response: Dict[str, Any]) -> str:
        """
        Extract the raw text content from a Bedrock API response.

        Args:
            response (Dict[str, Any]): The complete API response from Bedrock

        Returns:
            str: The extracted text content
        """
        logging.info(f"Extracting message content from response: {json.dumps(response, indent=2)}")
        try:
            # Navigate through the nested structure to get the text content
            message = response.get("output", {}).get("message", {})
            content_list = message.get("content", [])

            if not content_list or len(content_list) == 0:
                self.logger.warning("No content found in response")
                return ""

            text_content = content_list[0].get("text", "")

            # Log token usage for monitoring
            if "usage" in response:
                token_usage = response["usage"]
                self.logger.info(f"Input tokens: {token_usage['inputTokens']}")
                self.logger.info(f"Output tokens: {token_usage['outputTokens']}")
                self.logger.info(f"Total tokens: {token_usage['totalTokens']}")

            return str(text_content)

        except Exception as e:
            self.logger.error(f"Error extracting message content: {e}")
            return ""

    def get_validated_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and validate response from Bedrock.

        Args:
            response (Dict[str, Any]): Bedrock response

        Returns:
            Dict[str, Any]: Validated response content
        """
        output = None
        if response.get("stopReason") == "end_turn":
            self.logger.info(f"Stop reason: {response['stopReason']}")
            message = response.get("output", {}).get("message", {})
            content = message.get("content", [])

            # Log token usage
            token_usage = response["usage"]
            self.logger.info(f"Input tokens: {token_usage['inputTokens']}")
            self.logger.info(f"Output tokens: {token_usage['outputTokens']}")
            self.logger.info(f"Total tokens: {token_usage['totalTokens']}")

            if content and len(content) > 0:
                self.logger.info("Content is valid")
                output = content[0].get("text", None)
                self.logger.debug(f"Output: {output}")

        # Use the centralized function for validation
        result = process_model_response(output, None, self.logger)
        return cast(Dict[str, Any], result)

    def _model_system_role(self) -> str:
        """
        Get system role prompt for the model.

        Returns:
            str: System role prompt
        """
        if self.model_id == "anthropic.claude-3-5-sonnet-20240620-v1:0":
            cutoff_date = "2024-04"
            model_name = "Claude Sonnet"
            architecture = "Claude 3.5"
        elif self.model_id == "eu.meta.llama3-2-3b-instruct-v1:0":
            cutoff_date = "2024-09"
            model_name = "Llama3-2-3b"
            architecture = "Llama3"
        elif self.model_id == "us.meta.llama3-1-70b-instruct-v1:0":
            cutoff_date = "2024-09"
            model_name = "Llama3-1-70b"
            architecture = "Llama3"
        else:
            self.logger.error(f"Model {self.model_id} not found.")
            return "You are an AI assistant."

        return (
            f"You are {model_name}, trained by Meta, "
            f"based on the {architecture} model family.\n"
            f"Knowledge cutoff: {cutoff_date}\n"
            f"Current date: {datetime.now().date()}"
        )
