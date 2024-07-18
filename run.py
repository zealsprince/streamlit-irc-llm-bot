import os
import sys

import streamlit.web.cli as stcli
import streamlit.runtime.scriptrunner.magic_funcs

# Import dependencies here to make sure they're included in the build
import streamlit
import irc.client

from llama_cpp import Llama


def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))

    return resolved_path


if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("app.py"),
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())
