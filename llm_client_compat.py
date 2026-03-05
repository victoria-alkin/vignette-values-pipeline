import os, time, asyncio
from typing import List, Optional, Tuple, Dict, Any
from openai import AsyncOpenAI, AsyncAzureOpenAI
import concurrent.futures

def _using_azure() -> bool:
    return bool(os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"))

def _get_model_name() -> str:
    m = os.getenv("OPENAI_MODEL")
    if not m:
        raise RuntimeError("OPENAI_MODEL must be set.")
    return m

def _as_openai_base_url(endpoint: str) -> str:
    ep = endpoint.rstrip("/")
    # if the gateway already includes /v1, don't double it
    return ep if ep.endswith("/v1") else ep + "/v1"

async def get_response_compat(
    prompt: str,
    model: Optional[str] = None,
    n: int = 1,
    timeout: int = 60,
    json_mode: bool = False,
    **kwargs,
) -> Tuple[List[str], float, Dict[str, Any]]:
    model = model or _get_model_name()
    t0 = time.time()

    if json_mode and isinstance(prompt, str):
        prompt = prompt.rstrip() + "\n\nReturn ONLY valid JSON (no extra text)."

    async def _chat_call(client):
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "") if resp.choices else ""

    out: List[str] = []
    for _ in range(max(1, int(n))):
        if _using_azure():
            # First try Azure-style
            az = AsyncAzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
                timeout=timeout,
            )
            try:
                text = await _chat_call(az)
            except Exception as e:
                msg = str(e)
                # HMS gateway sometimes looks like Azure but routes OpenAI-style model names.
                # If Azure says DeploymentNotFound, try OpenAI-style /v1 routing with the same env vars.
                if "DeploymentNotFound" in msg or "The API deployment for this resource does not exist" in msg:
                    base_url = _as_openai_base_url(os.environ["AZURE_OPENAI_ENDPOINT"])
                    oa = AsyncOpenAI(
                        api_key=os.environ["AZURE_OPENAI_API_KEY"],  # reuse same key
                        base_url=base_url,
                        timeout=timeout,
                    )
                    text = await _chat_call(oa)
                else:
                    raise
        else:
            oa = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=timeout)
            text = await _chat_call(oa)

        out.append(text)

    elapsed = time.time() - t0
    return out, elapsed, {}  # 3-tuple for your installed gabriel

def get_response_compat_sync(*args, **kwargs):
    """
    Sync wrapper used by some gabriel code paths / older compatibility layers.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(get_response_compat(*args, **kwargs))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(lambda: asyncio.run(get_response_compat(*args, **kwargs)))
        return fut.result()