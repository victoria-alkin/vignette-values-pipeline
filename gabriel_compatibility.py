# gabriel_compatibility.py
import os

def gabriel_compatibility_env():
    import gabriel.utils.openai_utils as ou
    from llm_client_compat import get_response_compat, get_response_compat_sync

    ou.get_response = get_response_compat
    ou.get_response_async = get_response_compat
    ou.get_response_sync = get_response_compat_sync

    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        def _require_api_key_compat() -> str:
            k = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
            if not k:
                raise RuntimeError("Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY.")
            return k
        ou._require_api_key = _require_api_key_compat
        
        def _get_rate_limit_headers_noop(*args, **kwargs):
            return {}
        ou._get_rate_limit_headers = _get_rate_limit_headers_noop