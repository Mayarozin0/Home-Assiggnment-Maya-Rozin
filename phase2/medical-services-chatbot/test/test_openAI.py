"""
Test script to verify the Azure OpenAI connection.
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Azure OpenAI settings
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
GPT4_MINI_DEPLOYMENT = os.getenv("AZURE_GPT4_MINI_DEPLOYMENT", "gpt-4o-mini")

def test_openai_connection():
    """Test the connection to Azure OpenAI."""
    logger.info("Starting OpenAI connection test")

    # Check if environment variables are set
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        logger.error("Azure OpenAI environment variables are not set")
        logger.info(f"AZURE_OPENAI_ENDPOINT: {'SET' if AZURE_OPENAI_ENDPOINT else 'NOT SET'}")
        logger.info(f"AZURE_OPENAI_API_KEY: {'SET' if AZURE_OPENAI_API_KEY else 'NOT SET'}")
        return False

    try:
        # Initialize client
        logger.info(f"Initializing Azure OpenAI client with endpoint: {AZURE_OPENAI_ENDPOINT}")
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-02-15-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

        # Test simple completion
        logger.info(f"Testing completion with model: {GPT4_MINI_DEPLOYMENT}")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello!"}
        ]

        start_time = time.time()
        response = client.chat.completions.create(
            model=GPT4_MINI_DEPLOYMENT,
            messages=messages,
            max_tokens=100
        )
        elapsed_time = time.time() - start_time

        # Check response
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            logger.info(f"Received response in {elapsed_time:.2f}s: {content}")
            logger.info("Connection test PASSED")
            return True
        else:
            logger.error("Received empty response")
            logger.info("Connection test FAILED")
            return False

    except Exception as e:
        logger.error(f"Error testing OpenAI connection: {e}")
        logger.info("Connection test FAILED")
        return False

if __name__ == "__main__":
    success = test_openai_connection()
    if success:
        print("\nAzure OpenAI connection is working properly.")
    else:
        print("\nAzure OpenAI connection test failed. Check the logs above for details.")