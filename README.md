
# Streamlit IRC LLM Bot #

This is a bot written on top of Streamlit to interface with an IRC channel and respond to queries when mentioned.

## Setup ##

First, create your Python virtual environment via the following command:

    python3 -m venv venv

From there, enter this environment to install further requirements:

    source venv/bin/activate

Now install all requirements via the `venv` `pip3` installation:

    pip3 install -r requirements.txt

## Running ##

With the aforementioned requirements installed, you can invoke the app through `streamlit`:

    streamlit run app.py
