import time

import streamlit as st
import irc.client
from llama_cpp import Llama


@st.cache_resource
def load_model(model_path):
    """Load a Llama model from a given path.

    Args:
        model_path (str): The path to the model file.
    """
    llm = Llama(model_path, seed=-1)

    return llm


def section_load_model():
    """Load the LLM model as a Streamlit section."""

    if st.session_state.model_data:
        return

    st.header("Prompt")

    # Input box for model path
    model_path = st.text_input(
        "Enter the path to your GGUF model file:", "models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf")

    if st.button("Load"):
        # Check if a model path is provided
        if model_path:
            st.spinner("Loading model...")

            # Load the model
            llm = load_model(model_path)

            st.session_state.model_data = llm

            # Since the fundamental state of the app has changed, re-run the app.
            st.rerun()

    return


def add_to_irc_log(message, refresh=True):
    """Add a message to the IRC log in the Streamlit session state."""
    st.session_state.irc_log.append(message)

    if len(st.session_state.irc_log) > 50:
        st.session_state.irc_log.pop(0)

    if refresh:
        st.rerun()


def section_irc_connect():
    if st.session_state.irc_client or st.session_state.irc_connection:
        return

    st.header("IRC Connection")

    server = st.text_input("Server:", "irc.hivecom.net")
    port = st.number_input("Port:", 6697)
    nickname = st.text_input("Nickname:", "llm_bot")
    channel = st.text_input("Channel:", "#llm")

    irc_client = irc.client.Reactor()

    def on_connect(connection, event):
        add_to_irc_log("Connected to the server.")
        connection.join(channel)

    def on_join(connection, event):
        add_to_irc_log(f"Joined channel: {event.target}")

    def on_disconnect(connection, event):

        add_to_irc_log(
            f"Disconnected from the server. {
                event.arguments[0]
            }",
            # We don't want to refresh as this can interrupt the disconnect process.
            refresh=False
        )

    def on_privmsg(connection, event):
        add_to_irc_log(f"Private message from {
            event.source.nick}: {event.arguments[0]}")

    def on_pubmsg(connection, event):
        add_to_irc_log(f"Message in {event.target} from {
            event.source.nick}: {event.arguments[0]}")

    def on_action(connection, event):
        add_to_irc_log(f"Action in {event.target} from {
            event.source.nick}: {event.arguments[0]}")

    if st.button("Connect"):
        try:
            # Connect to the server.
            irc_connection = irc_client.server().connect(server, port, nickname)

            # Register event handlers.
            irc_connection.add_global_handler("welcome", on_connect)
            irc_connection.add_global_handler("join", on_join)
            irc_connection.add_global_handler("disconnect", on_disconnect)
            irc_connection.add_global_handler("privmsg", on_privmsg)
            irc_connection.add_global_handler("pubmsg", on_pubmsg)
            irc_connection.add_global_handler("action", on_action)

            # Make the IRC client and connection available in the Streamlit session state.
            st.session_state.irc_client = irc_client
            st.session_state.irc_connection = irc_connection

            st.rerun()

        except irc.client.ServerConnectionError as e:
            st.error(f"Could not connect to server: {e}")


def section_irc_content():
    if not st.session_state.irc_client or not st.session_state.irc_connection:
        return

    st.header("IRC")

    if st.button("Disconnect"):
        st.session_state.irc_connection.disconnect("Leaving")

        st.session_state.irc_client = None
        st.session_state.irc_connection = None

        st.rerun()

    st.write(st.session_state.irc_log)


def section_prompt_response(llm, role, prompt, parameters={
    "maximum_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "repeat_penalty": 1.1
}):
    """Generate a response from the LLM model as a Streamlit section.

        Args:
            llm (Llama): The Llama model to generate the response.
            role (str): The role of the LLM in the conversation.
            prompt (str): The user's prompt.
            parameters (dict): The parameters to use for the generation.

        Parameters:
            maximum_tokens (int): The maximum number of tokens to generate.
            temperature (float): The temperature to use for generation.
            top_p (float): The top-p value to use for generation.
            top_k (int): The top-k value to use for generation.
            repeat_penalty (float): The repeat penalty value to use for generation.
    """

    if st.session_state.regenerate or st.session_state.generated_content:
        st.header("Response")

    # Placeholder for streaming the response
    result_container = st.empty()

    if st.session_state.regenerate:
        streaming_result = llm.create_chat_completion(
            stream=True,
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt}
            ],
            max_tokens=parameters["maximum_tokens"],
            temperature=parameters["temperature"],
            top_p=parameters["top_p"],
            top_k=parameters["top_k"],
            repeat_penalty=parameters["repeat_penalty"]
        )

        # Reset the generated_content in the session state.
        st.session_state.generated_content = ""

        # Stream the response from the model
        for segment in streaming_result:
            choices = segment["choices"]

            if choices is None:
                break

            for choice in choices:
                if not "content" in choice["delta"]:
                    continue

                st.session_state.generated_content += choice["delta"]["content"]

                # Update container with the new content
                result_container.write(st.session_state.generated_content)

        # We're done generating the response.
        st.session_state.regenerate = False

    else:
        result_container.markdown(st.session_state.generated_content)


def clear_model():
    """Clear the loaded model from the Streamlit session state."""
    del st.session_state.model_data


def section_prompt():
    """Prompt the user for input and generate a response from the LLM model as a Streamlit section."""

    if not st.session_state.model_data:
        return

    st.header("Prompt")

    with st.sidebar:

        st.title("Model")

        st.code(st.session_state.model_data.model_path)

        st.button("Change Model", on_click=clear_model)

        st.title("Parameters")

        role_input = st.text_area(
            "LLM Role",
            "You are an assistant that will answer questions to the best of their abilities."
        )

        maximum_tokens_slider = st.slider("Maximum Tokens", 0, 8192, 1024, 1)

        temperature_slider = st.slider("Temperature", 0.0, 2.0, 0.7, 0.01)

        top_p_slider = st.slider("Top P", 0.0, 1.0, 0.95, 0.01)

        top_k_slider = st.slider("Top K", 0, 100, 50, 1)

        repeat_penalty_slider = st.slider(
            "Repeat Penalty", 0.0, 2.0, 1.1, 0.01)

    # Text Input for user's prompt
    user_input = st.text_area(
        "Enter a manual prompt:",
        "What can you tell me about the moon landing?"
    )

    if st.button("Generate!"):
        st.session_state.regenerate = True

    section_prompt_response(
        st.session_state.model_data,
        role_input,
        user_input,
        parameters={
            "maximum_tokens": maximum_tokens_slider,
            "temperature": temperature_slider,
            "top_p": top_p_slider,
            "top_k": top_k_slider,
            "repeat_penalty": repeat_penalty_slider
        }
    )


def initialize_state():
    """Initialize the Streamlit session state."""

    if "model_data" not in st.session_state:
        st.session_state.model_data = None

    if "regenerate" not in st.session_state:
        st.session_state.regenerate = False

    if "generated_content" not in st.session_state:
        st.session_state.generated_content = ""

    if "irc_client" not in st.session_state:
        st.session_state.irc_client = None

    if "irc_connection" not in st.session_state:
        st.session_state.irc_connection = None

    if "irc_log" not in st.session_state:
        st.session_state.irc_log = []


def main():
    """Run the Streamlit application."""

    initialize_state()

    # Streamlit application title
    st.title("Interactive LLM Prompt")

    section_irc_connect()

    section_irc_content()

    section_load_model()

    section_prompt()

    while True:
        time.sleep(1)

        if st.session_state.irc_connection:
            st.session_state.irc_client.process_once()

            st.rerun()


if __name__ == "__main__":
    main()
