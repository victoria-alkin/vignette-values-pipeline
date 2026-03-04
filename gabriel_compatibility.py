# gabriel_compatibility.py
"""
Compatibility:
- Patch upstream gabriel's internal get_response() so it can use Azure via AsyncAzureOpenAI.
- Also patch gabriel's internal OPENAI_API_KEY pre-checks so Azure-only setups work.
"""

import os

def gabriel_compatibility_env():
    # Safe import inside function
    import gabriel.utils.openai_utils as ou
    from llm_client_compat import get_response_compat

    ou.get_response = get_response_compat

    using_azure = bool(os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"))
    if using_azure:
        def _require_api_key_compat() -> str:
            k = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
            if not k:
                raise RuntimeError(
                    "Set OPENAI_API_KEY or Azure envs (AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT)."
                )
            return k

        ou._require_api_key = _require_api_key_compat

        def _get_rate_limit_headers_noop(*args, **kwargs):
            return None

        ou._get_rate_limit_headers = _get_rate_limit_headers_noop