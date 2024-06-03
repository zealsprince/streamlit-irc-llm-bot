import os
import time
import ssl
import socket

import streamlit as st
import irc.client

from llama_cpp import Llama

MODELS_PATH = "models"

DEFAULT_SERVER = "irc.hivecom.net"
DEFAULT_PORT = 6697
DEFAULT_NICKNAME = "llm_bot"
DEFAULT_CHANNEL = "#bots"
DEFAULT_BACKLOG_LENGTH = 25

DEFAULT_AUTORESPOND_ENABLED = True
DEFAULT_AUTORESPOND_INTERVAL = 5

DEFAULT_ROLE = "You are user in an IRC chat. Don't preface your response or use quotes. Limit your responses to a maximum of 32 words."
DEFAULT_PREPROMPT = ""
DEFAULT_PROMPT = "Tell a funny joke!"


def clean_message(message: str):
    """Clean a message for IRC by removing newlines, carriage returns, and the [INST] and [/INST] tags."""

    return message[:256].replace(  # Limit the message to 256 characters
        "\n", " "
    ).replace(
        "\r", " "
    ).replace(
        "[INST]", ""
    ).replace(
        "[/INST]", ""
    ).strip()


def send_to_channel(channel: str, nickname: str, message: str):
    """Send a message to the IRC channel in the Streamlit session state."""

    cleaned_message = clean_message(message)

    st.session_state.irc_connection.privmsg(channel, cleaned_message)

    add_to_irc_log(
        f"{channel} <{nickname}>: {cleaned_message}",
        refresh=False
    )


def send_to_user(user: str, nickname: str, message: str):
    """Send a message to a user."""

    cleaned_message = clean_message(message)

    st.session_state.irc_connection.privmsg(user, cleaned_message)

    add_to_irc_log(
        f"PRIVMSG <{nickname}>:<{user}>: {cleaned_message}",
        refresh=False
    )


def generate_general_response(nickname):
    """Generate a general response from the LLM model for the current backlog."""

    if not st.session_state.model_data:
        return "No model loaded!"

    llm = st.session_state.model_data

    result = llm.create_chat_completion(
        messages=[
            {"role": "assistant", "content": st.session_state.llm_role},
            {
                "role": "user",
                "content": f"The previous messages in this channel are the following:\n{"\n".join(st.session_state.irc_log)}\n\nYou are '{nickname}' in this chat.\n\n[INST]{st.session_state.llm_preprompt}What is your response to these messages?[/INST]"
            }
        ],
        max_tokens=st.session_state.llm_maximum_tokens,
        temperature=st.session_state.llm_temperature,
        top_p=st.session_state.llm_top_p,
        top_k=st.session_state.llm_top_k,
        repeat_penalty=st.session_state.llm_repeat_penalty
    )

    # I don't like this either. I'm just trying to get the first response.
    message = result["choices"][0]["message"]["content"]

    return message if len(message) > 0 else "(Empty response)"


def generate_direct_response(nickname, user, message):
    """Generate a direct response from the LLM model for a user ping."""

    if not st.session_state.model_data:
        return "No model loaded!"

    llm = st.session_state.model_data

    result = llm.create_chat_completion(
        messages=[
            {"role": "assistant", "content": st.session_state.llm_role},
            {
                "role": "user",
                "content": f"The previous messages in this channel are the following:\n{"\n".join(st.session_state.irc_log)}\n\nYou are '{nickname}' in this chat responding to '{user}' who just mentioned you saying '{message}'.\n\n[INST]{st.session_state.llm_preprompt}Respond to the message you were mentioned in.[/INST]"
            }
        ],
        max_tokens=st.session_state.llm_maximum_tokens,
        temperature=st.session_state.llm_temperature,
        top_p=st.session_state.llm_top_p,
        top_k=st.session_state.llm_top_k,
        repeat_penalty=st.session_state.llm_repeat_penalty
    )

    message = result["choices"][0]["message"]["content"]

    return message if len(message) > 0 else "(Empty response)"


def add_to_irc_log(message, refresh=True):
    """Add a message to the IRC log in the Streamlit session state."""
    st.session_state.irc_log.append(message)

    if len(st.session_state.irc_log) > st.session_state.irc_log_length:
        st.session_state.irc_log = st.session_state.irc_log[-st.session_state.irc_log_length:]

    if refresh:
        st.rerun()


def section_irc_connect():
    if st.session_state.irc_client or st.session_state.irc_connection:
        return

    st.header("IRC Connection")

    server = st.text_input("Server:", DEFAULT_SERVER, key="irc_server")
    port = st.number_input("Port:", 0, 65535, DEFAULT_PORT, 1, key="irc_port")
    nickname = st.text_input("Nickname:", DEFAULT_NICKNAME, key="irc_nickname")
    channel = st.text_input("Channel:", DEFAULT_CHANNEL, key="irc_channel")
    use_ssl = st.checkbox("Use SSL", True, key="irc_use_ssl")

    if st.button("Connect"):
        irc_client = irc.client.Reactor()

        def on_any(connection: irc.client.ServerConnection, event: irc.client.Event):
            add_to_irc_log(f"{event.arguments[0]}")

        def on_ping(connection: irc.client.ServerConnection, event: irc.client.Event):
            connection.pong("PONG")

        def on_connect(connection: irc.client.ServerConnection, event: irc.client.Event):
            add_to_irc_log(
                f"Connected to the server {server}. Joined {channel}."
            )

        def on_disconnect(connection: irc.client.ServerConnection, event: irc.client.Event):
            add_to_irc_log(
                f"Disconnected from the server: {
                    event.arguments[0]
                }",
                # We don't want to refresh as this can interrupt the disconnect process.
                refresh=False
            )

        def on_privmsg(connection: irc.client.ServerConnection, event: irc.client.Event):
            if event.source.nick != connection.username:
                add_to_irc_log(
                    f"PRIVMSG <{event.source.nick}>: {event.arguments[0]}",
                    refresh=False
                )

                response = generate_direct_response(
                    connection.username, event.source.nick, event.arguments[0]
                )

                send_to_user(
                    event.source.nick,
                    connection.username,
                    response
                )

        def on_pubmsg(connection: irc.client.ServerConnection, event: irc.client.Event):
            if event.source.nick != connection.username:
                # Add the incoming message to the IRC log but don't refresh yet since we're about to process it.
                add_to_irc_log(f"{event.target} <{
                    event.source.nick}>: {event.arguments[0]}",
                    refresh=False
                )

                if connection.username in event.arguments[0]:
                    response = generate_direct_response(
                        connection.username, event.source.nick, event.arguments[0]
                    )

                    send_to_channel(
                        event.target,
                        connection.username,
                        response
                    )

                if st.session_state.autorespond_enabled:
                    st.session_state.autorespond_counter = (
                        st.session_state.autorespond_counter + 1) % st.session_state.autorespond_interval

                    if st.session_state.autorespond_counter == 0:
                        response = generate_general_response(
                            connection.username
                        )

                        send_to_channel(
                            event.target,
                            connection.username,
                            response
                        )

                # send_to_channel blocks - if we reach this, we refresh since the last message is there.
                st.rerun()

        def on_action(connection: irc.client.ServerConnection, event: irc.client.Event):
            add_to_irc_log(
                f"Action in {event.target} from {
                    event.source.nick}: {event.arguments[0]}"
            )

        try:
            # Create the SSL factory for the IRC connection over TLS.
            def ssl_factory(address):
                context = ssl.create_default_context()

                # Optionally set if you want to verify the server's certificate
                context.verify_mode = ssl.CERT_REQUIRED

                sock = socket.create_connection(address)

                return context.wrap_socket(sock, server_hostname=address[0])

            # Connect to the server.
            irc_connection = irc_client.server().connect(
                server, port, nickname, connect_factory=ssl_factory if use_ssl else socket.create_connection)

            # Register event handlers.
            irc_connection.add_global_handler("welcome", on_connect)
            irc_connection.add_global_handler("disconnect", on_disconnect)
            irc_connection.add_global_handler("privmsg", on_privmsg)
            irc_connection.add_global_handler("pubmsg", on_pubmsg)
            irc_connection.add_global_handler("action", on_action)
            irc_connection.add_global_handler("ping", on_ping)

            # Debug handler. Keep in mind enabling this disables the other handlers due to st.rerun()
            # irc_connection.add_global_handler("all_events", on_any)

            # Make the IRC client and connection available in the Streamlit session state.
            st.session_state.irc_client = irc_client
            st.session_state.irc_connection = irc_connection

            # Join the channel.
            irc_connection.join(channel)

            st.rerun()

        except irc.client.ServerConnectionError as e:
            st.error(f"Could not connect to server: {e}")


def handle_irc_log_length():
    """Handle the IRC log length slider."""
    if len(st.session_state.irc_log) > st.session_state.irc_log_length:
        st.session_state.irc_log = st.session_state.irc_log[-st.session_state.irc_log_length:]


def section_irc_content():
    if not st.session_state.irc_client or not st.session_state.irc_connection:
        return

    st.header("IRC")

    st.checkbox(
        "Autorespond",
        value=DEFAULT_AUTORESPOND_ENABLED,
        key="autorespond_enabled"
    )

    st.slider(
        "Autorespond Interval",
        1, 100, DEFAULT_AUTORESPOND_INTERVAL, 1,
        key="autorespond_interval",
        disabled=not st.session_state.autorespond_enabled
    )

    st.slider("Backlog Length / Context size", 1, 250, DEFAULT_BACKLOG_LENGTH, 1,
              on_change=handle_irc_log_length, key="irc_log_length")

    if st.button("Disconnect"):
        st.session_state.irc_connection.disconnect("Leaving")

        st.session_state.irc_log = []
        st.session_state.irc_client = None
        st.session_state.irc_connection = None

        st.rerun()

    if len(st.session_state.irc_log) == 0:
        st.info("Connecting...")

    else:
        st.code("\n".join(st.session_state.irc_log), language="text")


@ st.cache_resource
def load_model(model_path, context_length=2048):
    """Load a Llama model from a given path.

    Args:
        model_path (str): The path to the model file.
    """
    llm = Llama(model_path, seed=-1, n_ctx=context_length)

    return llm


def section_model_load():
    """Load the LLM model as a Streamlit section."""

    if st.session_state.model_data:
        return

    st.header("Model")

    context_length = st.slider(
        "Context Length",
        0, 8192, 2048, 1,
        key="llm_context_length"
    )

    # Get the model files in the models directory.
    models = os.listdir(MODELS_PATH)

    model_path = st.selectbox(
        "Select a model to load:",
        models
    )

    direct_model_path = st.text_input(
        "Alternatively enter a direct path:", "")

    if st.button("Load"):
        if direct_model_path != "":
            model_path = direct_model_path

        # Check if a model path is provided
        if model_path:
            # Check if the model path is a direct path or a relative path.
            if direct_model_path == "":
                model_path = os.path.join(MODELS_PATH, model_path)

            st.spinner("Loading model...")

            # Load the model
            llm = load_model(model_path, context_length)

            st.session_state.model_data = llm

            # Since the fundamental state of the app has changed, re-run the app.
            st.rerun()

    return


def section_model_prompt_response(llm, role, prompt, parameters={
    "maximum_tokens": 256,
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

        if st.button("Send to IRC"):
            send_to_channel(
                st.session_state.irc_channel,
                st.session_state.irc_nickname,
                st.session_state.generated_content
            )

            st.session_state.generated_content = ""

            st.rerun()

    # Placeholder for streaming the response
    result_container = st.empty()

    if st.session_state.regenerate:
        streaming_result = llm.create_chat_completion(
            stream=True,
            messages=[
                {"role": "assistant", "content": role},
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


def section_model_prompt():
    """Prompt the user for input and generate a response from the LLM model as a Streamlit section."""

    if not st.session_state.model_data:
        return

    st.header("Model")

    with st.sidebar:

        st.title("Model Parameters")

        st.code(st.session_state.model_data.model_path)

        if st.button("Change Model"):
            clear_model()

            st.rerun()

        st.title("Parameters")

        role_input = st.text_area(
            "Role",
            DEFAULT_ROLE,
            key="llm_role"
        )

        preprompt_input = st.text_area(
            "Preprompt",
            DEFAULT_PREPROMPT,
            key="llm_preprompt"
        )

        maximum_tokens_slider = st.slider(
            "Maximum Tokens",
            0, 8192, 64, 1,
            key="llm_maximum_tokens"
        )

        temperature_slider = st.slider(
            "Temperature",
            0.0, 2.0, 0.7, 0.01,
            key="llm_temperature"
        )

        top_p_slider = st.slider(
            "Top P",
            0.0, 1.0, 0.95, 0.01,
            key="llm_top_p"
        )

        top_k_slider = st.slider(
            "Top K",
            0, 100, 50, 1,
            key="llm_top_k"
        )

        repeat_penalty_slider = st.slider(
            "Repeat Penalty",
            0.0, 2.0, 1.1, 0.01,
            key="llm_repeat_penalty"
        )

    # Text Input for user's prompt
    user_input = st.text_area(
        "Enter a prompt:",
        DEFAULT_PROMPT,
    )

    if st.button("Generate!"):
        st.session_state.regenerate = True

    section_model_prompt_response(
        st.session_state.model_data,
        role_input,
        f"[INST]{preprompt_input}[/INST]\n{user_input}",
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

    if "irc_nickname" not in st.session_state:
        st.session_state.irc_nickname = DEFAULT_NICKNAME

    if "irc_channel" not in st.session_state:
        st.session_state.irc_channel = DEFAULT_CHANNEL

    if "irc_server" not in st.session_state:
        st.session_state.irc_server = DEFAULT_SERVER

    if "irc_port" not in st.session_state:
        st.session_state.irc_port = DEFAULT_PORT

    if "irc_log" not in st.session_state:
        st.session_state.irc_log = []

    if "irc_log_length" not in st.session_state:
        st.session_state.irc_log_length = DEFAULT_BACKLOG_LENGTH

    if "autorespond_enabled" not in st.session_state:
        st.session_state.autorespond_enabled = DEFAULT_AUTORESPOND_ENABLED

    if "autorespond_interval" not in st.session_state:
        st.session_state.autorespond_interval = DEFAULT_AUTORESPOND_INTERVAL

    if "autorespond_counter" not in st.session_state:
        st.session_state.autorespond_counter = 0


def main():
    """Run the Streamlit application."""

    initialize_state()

    st.set_page_config(
        page_title="IRC LLM Bot"
    )

    # Streamlit application title
    st.title("IRC LLM Bot")

    section_irc_connect()

    section_irc_content()

    st.divider()

    section_model_load()

    section_model_prompt()

    while True:
        time.sleep(.1)

        if st.session_state.irc_connection:
            st.session_state.irc_client.process_once()


if __name__ == "__main__":
    main()
