# gabriel_compatibility.py
"""
Compatibility shim:
- Keep env checks simple (don't remap Azure -> OPENAI_*).
- Patch upstream gabriel's internal get_response() so it can use Azure via AsyncAzureOpenAI.
"""

def gabriel_compatibility_env():
    # Import inside the function so this is safe even if gabriel isn't installed yet.
    import gabriel.utils.openai_utils as ou
    from llm_client_compat import get_response_compat

    # Monkeypatch: any internal gabriel calls to ou.get_response will use our compat client.
    ou.get_response = get_response_compat