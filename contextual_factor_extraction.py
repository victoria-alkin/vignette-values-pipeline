"""
Extract contextual factors from free-text vignettes.
"""
# Output columns
OUTPUT_COLS = ["vignette_id", "factor_id", "text"]

import argparse

def get_args():
    ap = argparse.ArgumentParser(description="LLM-based contextual factor extractor.")
    ap.add_argument("--infile", required=True, help="Path to table with columns: vignette_id, vignette_text")
    ap.add_argument("--outdir", required=True, help="Directory to write outputs")
    ap.add_argument("--minlen", type=int, default=3, help="Minimum snippet length after trimming (default: 3)")
    ap.add_argument("--dedupe", action="store_true", help="Drop exact duplicate factors within each vignette")
    ap.add_argument("--save-name", default="factors_extracted_llm.csv", help="Output CSV filename")
    ap.add_argument("--model", default="gpt-4o-1120", help="Model (supports JSON Schema via Responses API)")
    return ap.parse_args()

import pandas as pd
from pathlib import Path

def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    ext = p.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(p)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    elif ext in (".jsonl", ".ndjson"):
        df = pd.read_json(p, lines=True)
    else:
        raise ValueError(f"Unsupported input type: {ext}")

    required = {"vignette_id", "vignette_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

# Contextual factor extraction guidelines
ISFACTOR_GUIDE = """
You will extract discrete clinical/contextual FACTS from the vignette.

DEFINITION — IsFactor = True:
- One discrete, decision-relevant clinical/contextual fact from the vignette.
- May include a short explanatory clause that directly clarifies that fact.
- Keep parentheticals if they clarify.
- Panels presented at the same timepoint (e.g., "Vitals: HR…, BP…, SpO₂…",
  "ABG: PaCO₂…, pH…", "Electrolytes: …, …, …") should be kept together as ONE item.
- Treat patient (or caregiver) statements as factors, including statements of preferences, advance directives, consent/refusal, agreements/commitments, and goals of care.

ATTRIBUTION (exposure → outcome) RULES:
- If a fact states a risk, benefit, contraindication, effect, or harm, and the text includes a precipitating
  fact, then the contextual factor MUST explicitly include
  the precipitating intervention/device/exposure (e.g., interventions,
  medications/therapies, procedures/devices) when present in the text.
- Preserve the causal connector that links exposure to outcome (e.g., "with", "from", "after",
  "during", "secondary to", "due to", "on", "under").
- Do not generalize such a fact into a free-floating statement if the exposure is specified.
- Preserve negations/qualifiers tied to exposure (e.g., contraindicated, avoid, continued/new/increased).

PANEL FIDELITY & QUALIFIERS:
- When a panel (e.g., vitals, ABG, labs, electrolytes) includes an explicit qualitative modifier
  or clinical qualifier (e.g., “elevated”, “low”, “inappropriately low”, “severe”, “marked”),
  preserve that qualifier together with the measurement it modifies.
- If a qualifier is stated for the panel as a whole, include it as a trailing clause within the same item.
- Do NOT invent or intensify qualifiers; include only those explicitly present in the text.
- Preserve parentheses and similar constructions that encode such qualifiers.

VERBATIM PREFERENCE:
- When possible, return the FACT using the exact wording from the vignette (verbatim substring).
- Do not substitute synonyms or rephrase clinical terms when a direct quote is available.
- Preserve capitalization, punctuation, units, symbols, and panel formatting as written.
- You may only trim surrounding whitespace and a trailing period; do not otherwise change the text.
- For panels (e.g., vitals/ABG/labs/electrolytes), keep separators and units exactly as written.

IsFactor = False (do NOT return):
- Pure explanation without a core fact.
- Text that clearly mixes different timepoints.
- Trivial fragments shorter than the configured minimum.

INSTRUCTIONS:
- Produce a LIST of FACT strings found in the vignette text.
- Each list item is ONE fact; keep panel items intact (do not split).
- Do not invent; quote only what is present in the text.
- Keep items concise; include key numbers/units if present.
- Do not include duplicates.
- Minimum length threshold = {minlen} characters (very short but meaningful items are allowed if ≥ {minlen}).
"""

# JSON schema for outputs
def make_factor_list_schema(minlen: int = 1) -> dict:
    """
    JSON Schema for structured outputs:
      {
        "factors": ["string", "string", ...]
      }
    `minlen` gently encourages non-trivial strings (SDK/model still may need code-side checks).
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "factors": {
                "type": "array",
                "items": {
                    "type": "string",
                    "minLength": max(1, int(minlen))
                }
            }
        },
        "required": ["factors"]
    }


def build_messages(vignette_text: str, minlen: int) -> list[dict]:
    system_msg = (
        "You are a careful information extractor. "
        "Work only with facts present in the text. Do not invent information."
    )
    user_msg = (
        ISFACTOR_GUIDE.format(minlen=minlen).strip()
        + "\n\nVIGNETTE:\n"
        + vignette_text.strip()
        + "\n\nReturn JSON only."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

import os
from typing import Tuple

try:
    from openai import OpenAI
    try:
        from openai import AzureOpenAI
    except Exception:
        AzureOpenAI = None
except ImportError as e:
    raise SystemExit("Please `pip install openai` (v1.x).") from e


def create_client() -> Tuple[object, bool]:
    """
    Returns (client, is_azure).
    - If Azure env vars are present, use AzureOpenAI with your endpoint/version.
    - Otherwise use the standard OpenAI client.

    Required env for Azure:
      AZURE_OPENAI_API_KEY
      AZURE_OPENAI_ENDPOINT      (e.g., https://<your-resource>.openai.azure.com/)
      AZURE_OPENAI_API_VERSION   (e.g., 2024-12-01-preview)
    Required env for OpenAI:
      OPENAI_API_KEY
    """
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_ep = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_ver = os.getenv("AZURE_OPENAI_API_VERSION")

    if azure_key and azure_ep:
        if AzureOpenAI is None:
            raise SystemExit("Your OpenAI SDK doesn't expose AzureOpenAI. Upgrade `openai` to v1.x.")
        if not azure_ver:
            raise SystemExit("Set AZURE_OPENAI_API_VERSION for Azure OpenAI (e.g., 2024-12-01-preview).")
        client = AzureOpenAI(
            azure_endpoint=azure_ep,
            api_key=azure_key,
            api_version=azure_ver,
        )
        return client, True

    # Fallback: standard OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY (or Azure env vars) before running.")
    client = OpenAI()
    return client, False


import json

def call_llm_factors(client, model: str, vignette_text: str, minlen: int) -> list[str]:
    """
    Call the LLM to extract factors as a JSON list of strings using
    Chat Completions + function-calling (tool calling), which is broadly supported.
    """
    schema = make_factor_list_schema(minlen=minlen)
    messages = build_messages(vignette_text, minlen=minlen)

    # Tool to force a JSON object with {"factors": [...]}
    tools = [{
        "type": "function",
        "function": {
            "name": "return_factor_list",
            "description": "Return only the factor list per the provided JSON Schema.",
            "parameters": schema,
        }
    }]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "return_factor_list"}},
        )
    except Exception as e:
        raise RuntimeError(f"LLM call failed (chat.completions): {e}") from e

    try:
        choice = resp.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None)
        if tool_calls:
            args_str = tool_calls[0].function.arguments
            data = json.loads(args_str)
            factors = data.get("factors", [])
        else:
            content = choice.message.content or ""
            data = json.loads(content)
            factors = data.get("factors", [])
    except Exception as e:
        raise RuntimeError(f"Could not parse factors from model response: {e}") from e

    if not isinstance(factors, list):
        raise RuntimeError("Model response did not include a 'factors' array.")

    return factors

import re
from typing import List

def _normalize_for_dedupe(s: str) -> str:
    """
    Normalization for exact-ish dedupe:
    - lowercase
    - collapse internal whitespace
    - trim leading/trailing spaces
    - drop trivial trailing punctuation (.,;:)
    """
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(".,;:")
    return s

def clean_factors(factors: List[str], minlen: int, dedupe: bool) -> List[str]:
    """Apply min-length filter and (optional) normalized dedupe, preserving order."""
    out: List[str] = []
    seen = set()
    for f in factors:
        f = (f or "").strip()
        if len(f) < minlen:
            continue
        if dedupe:
            key = _normalize_for_dedupe(f)
            if key in seen:
                continue
            seen.add(key)
        out.append(f)
    return out

from typing import List, Dict

def attach_ids(vignette_id: str, factors: List[str]) -> List[Dict]:
    """
    For each factor string, assign a per-vignette incremental ID:
      <vignette_id>-f001, <vignette_id>-f002, ...
    Returns a list of dicts with keys: vignette_id, factor_id, text
    """
    rows: List[Dict] = []
    for i, text in enumerate(factors, start=1):
        fid = f"{vignette_id}-f{i:03d}"
        rows.append({"vignette_id": vignette_id, "factor_id": fid, "text": text})
    return rows

import time, random
from typing import List, Dict

def with_retry(fn, tries: int = 4, base: float = 0.5, jitter: float = 0.25):
    """
    Retry wrapper with exponential backoff.
    base * (2^k) + uniform(0, jitter)
    """
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            if k == tries - 1:
                raise
            delay = base * (2 ** k) + random.uniform(0, jitter)
            time.sleep(delay)

def extract_all(df, client, model: str, minlen: int, dedupe: bool, rate_sleep: float = 0.2) -> List[Dict]:
    """
    Main extraction loop.
    Returns a list[dict] with columns: vignette_id, factor_id, text
    """
    all_rows: List[Dict] = []
    for _, r in df.iterrows():
        vid = str(r["vignette_id"])
        vtx = str(r["vignette_text"])

        # 1) LLM call (with retry)
        factors_raw = with_retry(lambda: call_llm_factors(client, model, vtx, minlen))

        # 2) Post-process (minlen + optional dedupe, preserve order)
        factors = clean_factors(factors_raw, minlen=minlen, dedupe=dedupe)

        # 3) Assign IDs and collect
        all_rows.extend(attach_ids(vid, factors))

        # 4) Rate limit between requests
        time.sleep(rate_sleep)

    return all_rows

import pandas as pd
from pathlib import Path
from typing import List, Dict

def save_results(rows: List[Dict], outdir: str, save_name: str) -> Path:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / save_name

    df = pd.DataFrame(rows, columns=OUTPUT_COLS)
    df.to_csv(out_file, index=False)
    return out_file

# ---- Programmatic wrapper for GUI use ----
def run_factor_extractor(
    infile: str,
    outdir: str,
    *,
    minlen: int = 3,
    dedupe: bool = False,
    save_name: str = "factors_extracted_llm.csv",
    model: str | None = None,
    rate_sleep: float = 0.2,
):
    """
    Runs the extractor and returns the Path to the written CSV
    with columns: vignette_id, factor_id, text.
    - If `model` is None, uses OPENAI_MODEL env var, else falls back to 'gpt-4o-1120'.
    - Works with both AzureOpenAI (deployment ID) and OpenAI (model name).
    """
    df = load_table(infile)
    client, _ = create_client()
    model_to_use = model or os.getenv("OPENAI_MODEL") or "gpt-4o-1120"

    rows = extract_all(
        df=df,
        client=client,
        model=model_to_use,
        minlen=minlen,
        dedupe=dedupe,
        rate_sleep=rate_sleep,
    )
    return save_results(rows, outdir, save_name)

def main():
    args = get_args()
    df = load_table(args.infile)

    client, is_azure = create_client()
    rows = extract_all(
        df=df,
        client=client,
        model=args.model,
        minlen=args.minlen,
        dedupe=args.dedupe,
    )

    out_file = save_results(rows, args.outdir, args.save_name)
    print(f"[ok] wrote {out_file} — {len(rows)} factor rows")

if __name__ == "__main__":
    main()
