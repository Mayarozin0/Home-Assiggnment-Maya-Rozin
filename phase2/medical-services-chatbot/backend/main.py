"""
FastAPI application for the Medical Services Chatbot.
"""
import json
import logging
import os
from typing import List, Dict, Any, Optional

# Load environment variables at the very beginning
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai_client import OpenAIClient
from prompts import INFORMATION_COLLECTION_PROMPT, VERIFY_USER_INFORMATION_FUNCTION, GET_INFORMATION_FUNCTION
from utils import verify_user_information, prepare_messages_for_qa, extract_user_info_from_tool_call, get_information

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(title="Medical Services Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the OpenAI client
openai_client = OpenAIClient()


# Pydantic models for request and response
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    user_info: Optional[Dict[str, Any]] = None
    conversation_phase: str = "information_collection"


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, Any]


class ChatResponse(BaseModel):
    response: str
    conversation_phase: str
    user_info: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    message_to_add: Optional[Dict[str, str]] = None


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message and return a response.
    """
    try:
        # Log the request
        logger.info(f"Received chat request: {request}")

        # Process based on conversation phase
        if request.conversation_phase == "information_collection":
            return await process_information_collection(request)
        elif request.conversation_phase == "qa":
            return await process_qa(request)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid conversation phase: {request.conversation_phase}")

    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def process_information_collection(request: ChatRequest):
    """
    Process the information collection phase.
    """
    # Convert Pydantic messages to dictionaries
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    # Add system message if not present
    if not messages or messages[0]["role"] != "system":
        messages = [{"role": "system", "content": INFORMATION_COLLECTION_PROMPT}] + messages

    # Call OpenAI API
    functions = [VERIFY_USER_INFORMATION_FUNCTION]
    response = openai_client.get_chat_completion(messages, functions)

    # Extract the response message
    response_message = response.choices[0].message
    response_content = response_message.content or ""

    # Initialize variables
    tool_calls = []
    user_info = None
    conversation_phase = "information_collection"
    message_to_add = {"role": "assistant", "content": response_content}

    # Check if the model wants to call a function
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            # Extract the tool call details
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Add to tool calls list
            tool_calls.append({
                "id": tool_call.id,
                "function": {
                    "name": function_name,
                    "arguments": tool_call.function.arguments
                }
            })

            # If it's the verification function, process it
            if function_name == "verify_user_information":
                # Call the validation function
                validation_result = verify_user_information(
                    full_name=function_args.get("full_name"),
                    id_number=function_args.get("id_number"),
                    gender=function_args.get("gender"),
                    age=function_args.get("age"),
                    health_fund=function_args.get("health_fund"),
                    hmo_card_number=function_args.get("hmo_card_number"),
                    insurance_tier=function_args.get("insurance_tier")
                )

                # Extract user info
                user_info = function_args

                # Prepare the function response
                function_response = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": validation_result
                }

                # Add function response to messages
                messages.append(response_message.model_dump())
                messages.append(function_response)

                # Make a second API call with the validation result
                final_response = openai_client.get_chat_completion(messages)
                final_content = final_response.choices[0].message.content

                # Update message to add
                message_to_add = {"role": "assistant", "content": final_content}

                # If validation was successful, change phase to Q&A
                if "successful" in validation_result:
                    conversation_phase = "qa"

    # Return the chat response
    return ChatResponse(
        response=message_to_add["content"],
        conversation_phase=conversation_phase,
        user_info=user_info,
        tool_calls=tool_calls if tool_calls else None,
        message_to_add=message_to_add
    )


async def process_qa(request: ChatRequest):
    """
    Process the Q&A phase using the RAG approach.
    """
    logger.info("==== QA PHASE STARTED ====")
    logger.info(f"User info: {request.user_info}")
    logger.info(f"Messages: {[m.role for m in request.messages]}")

    # Ensure we have user info
    if not request.user_info:
        logger.error("Missing user info in QA phase")
        raise HTTPException(status_code=400, detail="User information required for Q&A phase")

    # Ensure we have user info
    if not request.user_info:
        raise HTTPException(status_code=400, detail="User information required for Q&A phase")

    # Convert Pydantic messages to dictionaries
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    # Get the last user message as the query
    user_query = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), None)
    if not user_query:
        raise HTTPException(status_code=400, detail="No user query found")

    qa_messages = prepare_messages_for_qa(user_query, request.user_info, messages)

    # Define the function to use
    functions = [GET_INFORMATION_FUNCTION]

    # Call OpenAI API with function calling
    response = openai_client.get_chat_completion(qa_messages, functions)

    # Extract the response message
    response_message = response.choices[0].message
    response_content = response_message.content or ""

    # Handle function calls
    tool_calls = []
    if response_message.tool_calls:
        # Add the assistant message with the function call to history
        messages.append(response_message.model_dump())

        for tool_call in response_message.tool_calls:
            # Extract the tool call details
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Add to tool calls list
            tool_calls.append({
                "id": tool_call.id,
                "function": {
                    "name": function_name,
                    "arguments": tool_call.function.arguments
                }
            })

            # Process the get_information function call
            if function_name == "get_information":
                # Call the function
                result = get_information(
                    query=function_args.get("query", ""),
                    hmo=function_args.get("hmo", request.user_info.get("health_fund", "")),
                    tier=function_args.get("tier", request.user_info.get("insurance_tier", ""))
                )

                # Add the function response to messages
                function_response = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result
                }
                messages.append(function_response)

        # Make a second API call with the function result
        final_response = openai_client.get_chat_completion(messages)
        response_content = final_response.choices[0].message.content

    # Return the chat response
    return ChatResponse(
        response=response_content,
        conversation_phase="qa",
        user_info=request.user_info,
        tool_calls=tool_calls if tool_calls else None,
        message_to_add={"role": "assistant", "content": response_content}
    )


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)