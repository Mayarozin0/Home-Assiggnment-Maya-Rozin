"""
Utility functions for the chatbot.
"""
from typing import Dict, Any, Tuple, List
from prompts import build_qa_prompt


def verify_user_information(
    full_name: str,
    id_number: str,
    gender: str,
    age: Any,
    health_fund: str,
    hmo_card_number: str,
    insurance_tier: str
) -> str:
    """
    Validate user information.

    Args:
        full_name: User's full name
        id_number: ID number (must be 9 digits)
        gender: Gender (must be one of: זכר, נקבה, אחר)
        age: Age (must be between 0 and 120)
        health_fund: Health fund (must be one of: מכבי, מאוחדת, כללית)
        hmo_card_number: HMO card number (must be 9 digits)
        insurance_tier: Insurance tier (must be one of: זהב, כסף, ארד)

    Returns:
        Validation result message
    """
    # Convert age to int if it's a string
    if isinstance(age, str) and age.isdigit():
        age = int(age)

    # Validate ID number: must be a 9-digit number
    if not (id_number.isdigit() and len(id_number) == 9):
        return "Invalid ID number format. It must be a 9-digit number."

    # Validate age: must be between 0 and 120
    if not (0 <= int(age) <= 120):
        return "Invalid age. It must be between 0 and 120."

    # Validate HMO card number: must be a 9-digit number
    if not (hmo_card_number.isdigit() and len(hmo_card_number) == 9):
        return "Invalid HMO card number format. It must be a 9-digit number."

    # Validate gender
    if gender not in ["זכר", "נקבה", "אחר"]:
        return "Invalid gender. Must be one of: זכר, נקבה, אחר."

    # Validate health fund
    if health_fund not in ["מכבי", "מאוחדת", "כללית"]:
        return "Invalid health fund. Must be one of: מכבי, מאוחדת, כללית."

    # Validate insurance tier
    if insurance_tier not in ["זהב", "כסף", "ארד"]:
        return "Invalid insurance tier. Must be one of: זהב, כסף, ארד."

    # All validations passed
    return "Validation successful."


def prepare_messages_for_qa(
    user_query: str,
    user_info: Dict[str, Any],
    conversation_history: List[Dict[str, str]] = None
) -> List[Dict[str, str]]:
    """
    Prepare messages for the Q&A phase.

    Args:
        user_query: The current user query
        user_info: User information dictionary
        conversation_history: Previous conversation history

    Returns:
        List of messages for the chat completion
    """
    # Build system message with user information
    system_message = {
        "role": "system",
        "content": build_qa_prompt(user_info)
    }

    # Use provided history or create a new history
    messages = conversation_history.copy() if conversation_history else []

    # Add system message at the beginning if not already there
    if not messages or messages[0]["role"] != "system":
        messages = [system_message] + messages

    # Add user query if not already added
    if not messages or messages[-1]["role"] != "user" or messages[-1]["content"] != user_query:
        messages.append({"role": "user", "content": user_query})

    return messages


def extract_user_info_from_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract user information from a tool call.

    Args:
        tool_call: Tool call from the OpenAI response

    Returns:
        Dictionary of user information
    """
    import json

    # Parse the arguments
    function_args = json.loads(tool_call["function"]["arguments"])

    # Extract user information
    user_info = {
        "full_name": function_args.get("full_name"),
        "id_number": function_args.get("id_number"),
        "gender": function_args.get("gender"),
        "age": function_args.get("age"),
        "health_fund": function_args.get("health_fund"),
        "hmo_card_number": function_args.get("hmo_card_number"),
        "insurance_tier": function_args.get("insurance_tier")
    }

    return user_info


def get_information(query, hmo, tier):
    """
    Retrieve information from the knowledge base.

    Args:
        query: The user's query
        hmo: The user's health fund
        tier: The user's insurance tier

    Returns:
        JSON-formatted string with the search results
    """
    from vector_search import search_knowledge_base
    import json

    # Search the knowledge base
    results = search_knowledge_base(query, hmo, tier)

    # Return the results as a JSON string
    return json.dumps(results, ensure_ascii=False)