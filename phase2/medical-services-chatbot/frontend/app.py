"""
Streamlit frontend for the medical services chatbot.
"""

import streamlit as st
import requests
import os
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Custom CSS
def load_css():
    st.markdown("""
    <style>
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: row;
            align-items: flex-start;
            position: relative;
        }
        .chat-message.user {
            background-color: #ECEEF7;
            border-radius: 0.5rem 0.5rem 0 0.5rem;
            align-self: flex-end;
            margin-left: 2rem;
        }
        .chat-message.assistant {
            background-color: #F5F5F5;
            border-radius: 0.5rem 0.5rem 0.5rem 0;
            align-self: flex-start;
            margin-right: 2rem;
        }
        .chat-message .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            object-fit: cover;
            margin-right: 1rem;
        }
        .chat-message .message {
            flex-grow: 1;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        .collection-phase {
            background-color: #e6f7ff;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .qa-phase {
            background-color: #f6ffed;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .stApp {
            direction: ltr;
        }
        .rtl-text {
            direction: rtl;
            text-align: right;
        }
        .ltr-text {
            direction: ltr;
            text-align: left;
        }
    </style>
    """, unsafe_allow_html=True)

def detect_rtl(text):
    """Detect if text contains RTL characters (primarily Hebrew)."""
    hebrew_chars = sum(1 for char in text if '\u0590' <= char <= '\u05FF')
    return hebrew_chars > len(text) / 3  # If more than 1/3 of chars are Hebrew

def display_message(role, content):
    """Display a message with the appropriate styling."""
    is_rtl = detect_rtl(content)
    direction_class = "rtl-text" if is_rtl else "ltr-text"

    if role == "user":
        st.markdown(f'<div class="chat-message user"><div class="message {direction_class}">{content}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message assistant"><div class="message {direction_class}">{content}</div></div>', unsafe_allow_html=True)

# Function to call the API
def chat_with_api(messages, user_info, conversation_phase):
    """Call the chat API with the given messages and user info."""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "messages": messages,
                "user_info": user_info,
                "conversation_phase": conversation_phase
            },
            timeout=60
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

# Initialize session state
def initialize_session_state():
    """Initialize the session state if it doesn't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "user_info" not in st.session_state:
        st.session_state.user_info = {}

    if "conversation_phase" not in st.session_state:
        st.session_state.conversation_phase = "information_collection"

    if "waiting_for_response" not in st.session_state:
        st.session_state.waiting_for_response = False

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

# Add a message to the chat
def add_message(role, content):
    """Add a message to the session state and display it."""
    st.session_state.messages.append({"role": role, "content": content})

# Main function to run the Streamlit app
def main():
    st.title("Medical Services Chatbot")

    # Load custom CSS
    load_css()

    # Initialize session state
    initialize_session_state()

    # Display phase indicator
    if st.session_state.conversation_phase == "information_collection":
        st.markdown('<div class="collection-phase">Information Collection Phase</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="qa-phase">Q&A Phase</div>', unsafe_allow_html=True)

    # Display the welcome message if this is the first interaction
    if not st.session_state.initialized:
        welcome_message = """Welcome to the Medical Services Chatbot! 

I'll help you find information about medical services available through your Israeli health fund (HMO).
"""
        add_message("assistant", welcome_message)
        st.session_state.initialized = True

    # Display chat history
    for message in st.session_state.messages:
        display_message(message["role"], message["content"])

    # Create the chat input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Type your message:", key="user_input")
        submit_button = st.form_submit_button("Send")

        if submit_button and user_input and not st.session_state.waiting_for_response:
            # Set waiting flag
            st.session_state.waiting_for_response = True

            # Add user message to the chat
            add_message("user", user_input)

            # Prepare API request
            api_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages
            ]

            # # Debug output
            # st.write(f"Debug - Sending messages: {api_messages}")
            # st.write(f"Debug - Conversation phase: {st.session_state.conversation_phase}")

            # Call the API
            response = chat_with_api(
                messages=api_messages,
                user_info=st.session_state.user_info,
                conversation_phase=st.session_state.conversation_phase
            )

            if response:
                st.write("Debug - Response received:", response)

                # Add the assistant's response to the chat
                add_message("assistant", response["response"])

                # Update user info if provided
                if response.get("user_info"):
                    st.session_state.user_info = response["user_info"]
                    st.write("Debug - Updated user info:", st.session_state.user_info)

                # Update conversation phase if changed
                if response.get("conversation_phase"):
                    old_phase = st.session_state.conversation_phase
                    new_phase = response["conversation_phase"]
                    st.write(f"Debug - Phase check: {old_phase} -> {new_phase}")

                    if old_phase != new_phase:
                        st.session_state.conversation_phase = new_phase
                        st.write("Debug - Phase changed")
            else:
                st.error("No response received from API")

            # Reset waiting flag
            st.session_state.waiting_for_response = False

            # Use this single rerun to update UI
            st.experimental_rerun()


    # Display a reset button
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.user_info = {}
        st.session_state.conversation_phase = "information_collection"
        st.session_state.waiting_for_response = False
        st.session_state.initialized = False
        st.experimental_rerun()

if __name__ == "__main__":
    main()