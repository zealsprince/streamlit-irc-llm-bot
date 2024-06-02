import streamlit as st
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


def clear_model():
    """Clear the loaded model from the Streamlit session state."""
    del st.session_state.model_data


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
    """
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

    st.header("Response")

    # Capture the full response.
    result = ""

    # Placeholder for streaming the response
    result_container = st.empty()

    # Stream the response from the model
    for segment in streaming_result:
        choices = segment["choices"]

        if choices is None:
            break

        for choice in choices:
            if not "content" in choice["delta"]:
                continue

            result += choice["delta"]["content"]

            # Update container with the new content
            result_container.write(result)

    result_container.markdown(result)


def section_prompt():
    """Prompt the user for input and generate a response from the LLM model as a Streamlit section."""

    if not st.session_state.model_data:
        return

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


def run():
    """Run the Streamlit application."""

    initialize_state()

    # Streamlit application title
    st.title("Interactive LLM Prompt")

    section_load_model()

    section_prompt()


if __name__ == "__main__":
    run()
