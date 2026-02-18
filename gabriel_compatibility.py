# gabriel_compatibility.py
import os

def gabriel_compatibility_env():
    """
    Make gabriel package work with either OpenAI or Azure credentials.

    If AZURE_OPENAI_* is set, map it into OPENAI_* variables some libraries rely on.
    No-op unless Azure vars exist. Never overwrites existing OPENAI_* vars.
    """
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_ep  = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_ver = os.getenv("AZURE_OPENAI_API_VERSION")

    # Map Azure -> OpenAI-compatible envs (only if missing)
    if azure_key:
        os.environ.setdefault("OPENAI_API_KEY", azure_key)

    if azure_ep:
        os.environ.setdefault("OPENAI_BASE_URL", azure_ep.rstrip("/") + "/openai")

    if azure_ver:
        os.environ.setdefault("OPENAI_API_VERSION", azure_ver)
