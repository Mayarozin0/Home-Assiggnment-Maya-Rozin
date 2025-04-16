import os
import logging
from openai import AzureOpenAI
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIClient:
    """Class to handle interactions with Azure OpenAI."""

    def __init__(self):
        """Initialize the Azure OpenAI client."""
        # Check for required environment variables
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if not api_key:
            logger.error("Missing AZURE_OPENAI_API_KEY environment variable")
            raise ValueError("Missing AZURE_OPENAI_API_KEY environment variable. Please set it in your .env file.")

        if not endpoint:
            logger.error("Missing AZURE_OPENAI_ENDPOINT environment variable")
            raise ValueError("Missing AZURE_OPENAI_ENDPOINT environment variable. Please set it in your .env file.")

        # Initialize the client
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2023-05-15",
            azure_endpoint=endpoint
        )

        # Get deployment names
        self.gpt4_deployment = os.getenv("AZURE_GPT4_DEPLOYMENT", "gpt-4o")
        self.gpt4_mini_deployment = os.getenv("AZURE_GPT4_MINI_DEPLOYMENT", "gpt-4o-mini")

        logger.info(f"OpenAI client initialized with endpoint: {endpoint}")
        logger.info(f"Using deployments - GPT-4: {self.gpt4_deployment}, GPT-4 Mini: {self.gpt4_mini_deployment}")

    def get_chat_completion(
            self,
            messages: List[Dict[str, str]],
            functions: Optional[List[Dict[str, Any]]] = None,
            function_call: Optional[str] = None,
            temperature: float = 0.1,
            top_p: float = 0.4,
            model: Optional[str] = None
    ) -> Any:
        """
        Get a chat completion from Azure OpenAI.

        Args:
            messages: List of message dictionaries with role and content.
            functions: Optional list of function definitions.
            function_call: Optional specification of function to call.
            temperature: Temperature for response generation.
            top_p: Top_p for response generation.
            model: Model deployment to use.

        Returns:
            The OpenAI API response.
        """
        # Use GPT-4o by default if no model specified
        deployment = model or self.gpt4_deployment

        # Basic parameters for the chat completion
        params = {
            "model": deployment,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p
        }

        # Add functions if provided
        if functions:
            params["tools"] = [{"type": "function", "function": func} for func in functions]
            params["tool_choice"] = "auto" if function_call is None else function_call

        # Call the API
        return self.client.chat.completions.create(**params)