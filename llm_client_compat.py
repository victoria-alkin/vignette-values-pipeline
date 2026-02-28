# llm_client_compat.py
import os
import time
import asyncio
from typing import List, Optional, Tuple

from openai import AsyncOpenAI, AsyncAzureOpenAI


def _using_azure() -> bool:
    return bool(os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"))


def _get_model_name() -> str:
    # Your repo currently uses OPENAI_MODEL. Keep that as the single source of truth.
    m = os.getenv("OPENAI_MODEL")
    if not m:
        raise RuntimeError("OPENAI_MODEL must be set.")
    return m


async def get_response_compat(
    prompt: str,
    model: Optional[str] = None,
    n: int = 1,
    timeout: int = 60,
) -> Tuple[List[str], float]:
    """
    Drop-in-ish replacement for gabriel.utils.openai_utils.get_response(prompt,...)

    Returns: (responses: List[str], elapsed_seconds: float)

    Chooses Azure if AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT are set,
    otherwise uses OpenAI if OPENAI_API_KEY is set.
    """
    model = model or _get_model_name()
    t0 = time.time()

    if _using_azure():
        client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
            timeout=timeout,
        )
    else:
        # standard OpenAI path
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY or Azure envs (AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT).")
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=timeout)

    # Use Responses API (matches what your earlier gabriel logs showed: /openai/responses)
    async def _one() -> str:
        resp = await client.responses.create(model=model, input=prompt)
        return resp.output_text or ""

    # run n calls (sequential to keep behavior predictable)
    out: List[str] = []
    for _ in range(max(1, int(n))):
        out.append(await _one())

    return out, time.time() - t0


def get_response_compat_sync(*args, **kwargs):
    """Convenience sync wrapper for scripts that aren't already async."""
    return asyncio.run(get_response_compat(*args, **kwargs))