"""
Extract discrete clinical decision options from free-text vignettes.

Input CSV schema:
  - vignette_id
  - vignette_text

Output CSV schema (wide, for the rater):
  - vignette_id
  - vignette_text
  - decision_1, decision_2

This script:
  1) lightly pre-scans the vignette for option-like sentences,
  2) prompts an LLM to return exactly 2 imperative decision options in JSON,
  3) cleans/deduplicates the options, and
  4) writes them to CSV.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List

import pandas as pd

# LLM client (Azure OpenAI or OpenAI)
def make_llm_client():
    """
    Returns a tuple: (backend, client, model, api_version)

    Chooses Azure OpenAI if AZURE_* env vars are present; otherwise falls back
    to the standard OpenAI API. Requires OPENAI_MODEL in both cases.
    """
    model = os.environ.get("OPENAI_MODEL")

    # Prefer Azure if its env vars are present
    if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
        from openai import AzureOpenAI
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=api_version,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
        if not model:
            raise RuntimeError("Set OPENAI_MODEL to your Azure deployment name.")
        return ("azure", client, model, api_version)

    # Otherwise use the standard OpenAI API
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY (or use Azure variables instead).")
    if not model:
        raise RuntimeError("Set OPENAI_MODEL (e.g., 'gpt-4o-mini').")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return ("openai", client, model, None)


# Prompting
SYSTEM_PROMPT = """You are a clinical decision extractor.

Goal:
- From a single clinical vignette, extract the discrete decision OPTIONS clinicians are considering or could reasonably consider given the text.
- Output exactly 2 options, each as a concise IMPERATIVE action phrase the clinician could take now.
- The two options should be MUTUALLY EXCLUSIVE separate options.

What counts as a decision:
- A concrete clinical action or explicit choice among actions
- If the vignette explicitly lists options, prefer those wordings.
- Do NOT invent options that have no grounding or are not mentioned in the vignette
- Do NOT include diagnostic impressions, facts, or predictions; only actionable choices.

Style:
- Imperative voice
- One option per item; ≤ 30 words each.
- Avoid duplicate or trivially overlapping options; prefer the broader umbrella action if needed.

Return JSON ONLY in the shape:
{
  "decisions": [
    {"id":"D1","text":"<option 1>"},
    {"id":"D2","text":"<option 2>"}
  ]
}
"""

USER_PROMPT_TEMPLATE = """VIGNETTE:
{vignette}

CANDIDATE HINTS (may refine/merge/ignore):
{hints}

Instructions:
- Produce exactly 2 decisions that a clinician could choose now.
- Keep each decision as an action phrase (imperative)
- JSON only
"""

# Candidate hints (pre-scan)
MODAL_PAT = re.compile(
    r'\b(?:option(?:s)?|alternatively|alternative|could|should|consider|may|might|'
    r'plan to|decide to|choose to|recommend|proceed with|initiate|start|stop|continue|'
    r'monitor|observe|intubate|extubate|ventilate|ECMO|tube|feed(?:ing)?|NG|IV)\b',
    flags=re.IGNORECASE
)

def rough_candidates(text: str) -> List[str]:
    """
    Pre-scan for likely option-bearing clauses.
    Returns short strings that the LLM can refine/merge/ignore.
    """
    s = re.sub(r'\s+', ' ', text).strip()
    # Split into sentences
    parts = re.split(r'(?<=[\.\?\!])\s+', s)

    hits = []
    for p in parts:
        if len(p) < 10:
            continue
        if MODAL_PAT.search(p):
            hits.append(p)

    # Also catch "One option is ..." / "Another option is ..."
    bulletish = re.findall(
        r'(?:one|another|an|the)\s+option\s+is\s+([^\.!\?]+)',
        s, flags=re.IGNORECASE
    )
    hits.extend(bulletish)

    # De-duplicate and cap
    out, seen = [], set()
    for h in hits:
        hh = h.strip()
        key = hh.lower()
        if hh and key not in seen:
            seen.add(key)
            out.append(hh)

    return out[:8]

# LLM call (JSON enforced)
def call_llm(client, backend: str, model: str, user_prompt: str) -> dict:
    """
    Sends SYSTEM_PROMPT + user_prompt to the chat API and returns parsed JSON.
    response_format='json_object' asks the model to return valid JSON only.
    """
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content
    return json.loads(content)

# Post-processing helpers
def clean_decision(txt: str) -> str:
    """
    Normalize a decision into a concise imperative action without trailing rationale.
    """
    t = txt.strip()

    # Remove simple leading numbering/bullets like "1) ", "- ", "2. "
    t = re.sub(r'^[\s\-\*\d\.\)]{0,4}', '', t)

    # If it starts with "to <verb>", make it imperative
    t = re.sub(r'^\s*to\s+', '', t, flags=re.IGNORECASE)

    # Normalize spaces
    t = re.sub(r'\s+', ' ', t).strip()

    # Capitalize first letter for consistency
    if t:
        t = t[0].upper() + t[1:]

    # Hard cap to avoid runaway strings
    return t[:300].rstrip()


def dedupe_keep_order(items: List[str]) -> List[str]:
    """
    Remove near-duplicates while preserving first occurrence.
    Normalizes to lowercase alphanumerics to catch trivial variants.
    """
    seen = set()
    out = []
    for x in items:
        key = re.sub(r'[^a-z0-9 ]+', '', x.lower())
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def keep_first_two(items: List[str]) -> List[str]:
    """
    Return two decisions.
    """
    return items[:2]

# Extract decisions for a single vignette
def extract_for_vignette(
    client,
    backend: str,
    model: str,
    vignette: str,
) -> List[str]:
    """
    1) Build 'hints' from the vignette
    2) Fill the USER_PROMPT_TEMPLATE with the vignette + hints.
    3) Call the LLM to get JSON decisions.
    4) Clean, dedupe, and cap the list.
    """
    # 1) Quick pass to find option-like sentences (breadcrumbs only)
    hints = rough_candidates(vignette)

    # 2) Construct the user prompt (show hints or <none>)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        vignette=vignette.strip(),
        hints="\n".join(f"- {h}" for h in hints) if hints else "<none>",
    )

    # 3) Ask the model for decisions (JSON enforced in call_llm)
    data = call_llm(client, backend, model, user_prompt)

    # Accept either [{"text": ...}, ...] or ["...", "..."]
    raw = []
    for item in data.get("decisions", []):
        if isinstance(item, dict):
            txt = item.get("text", "")
            if txt:
                raw.append(txt)
        elif isinstance(item, str):
            raw.append(item)

    # 4) Post-process
    cleaned = [clean_decision(t) for t in raw if t]
    cleaned = [t for t in cleaned if len(t) >= 3]  # drop tiny fragments
    cleaned = dedupe_keep_order(cleaned)
    cleaned = keep_first_two(cleaned)

    return cleaned

def get_args():
    ap = argparse.ArgumentParser(description="Extract discrete decision options from clinical vignettes.")
    ap.add_argument("--infile", required=True, help="CSV with columns: vignette_id, vignette_text")
    ap.add_argument("--outdir", required=True, help="Directory to write outputs")
    ap.add_argument("--save-name", default="vignettes_with_decisions.csv", help="Output CSV filename (wide format)")
    ap.add_argument("--dryrun", action="store_true", help="Print CSV to stdout instead of writing a file")
    return ap.parse_args()

# ---- Programmatic wrapper for GUI use ----
def run_decision_extractor(
    infile: str,
    outdir: str,
    *,
    save_name: str = "vignettes_with_decisions.csv",
):
    """
    Runs the decision extractor and returns CSV
    with columns: vignette_id, vignette_text, decision_1, decision_2.
    """
    # Ensure output directory exists
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # Initialize LLM client (Azure or OpenAI), using env vars
    backend, client, model, _ = make_llm_client()

    # Read input CSV and validate schema
    df = pd.read_csv(infile)
    required = {"vignette_id", "vignette_text"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Infile must contain columns: {sorted(required)}")

    # Process each vignette
    rows = []
    for _, r in df.iterrows():
        vid = r["vignette_id"]
        text = str(r["vignette_text"])

        decisions = extract_for_vignette(client, backend, model, text)

        if not decisions:
            # Keep row alignment even if nothing is extracted
            rows.append({"vignette_id": vid, "decision_id": "D1", "decision_text": ""})
            continue

        for i, d in enumerate(decisions, start=1):
            rows.append({"vignette_id": vid, "decision_id": f"D{i}", "decision_text": d})

    # Build long-format DF and pivot to wide
    long_df = pd.DataFrame(rows, columns=["vignette_id", "decision_id", "decision_text"])
    long_df = long_df[long_df["decision_text"].astype(str).str.len() > 0].copy()
    long_df["decision_idx"] = (
        long_df["decision_id"].str.extract(r"D(\d+)", expand=False).astype(int)
    )

    wide = (
        long_df
        .pivot(index="vignette_id", columns="decision_idx", values="decision_text")
        .sort_index(axis=1)
    )
    wide.columns = [f"decision_{i}" for i in wide.columns]  # decision_1, decision_2 expected

    # Bring back vignette_text and order columns
    base = df[["vignette_id", "vignette_text"]].drop_duplicates()
    final_df = base.merge(wide, on="vignette_id", how="left")
    decision_cols = [c for c in final_df.columns if c.startswith("decision_")]
    decision_cols = sorted(decision_cols, key=lambda x: int(x.split("_")[1]))[:2]  # keep only decision_1 and decision_2
    final_df = final_df[["vignette_id", "vignette_text", *decision_cols]]

    # Save and return the path
    save_path = outdir_path / save_name
    final_df.to_csv(save_path, index=False)
    return save_path

def main():
    args = get_args()

    # Ensure output directory exists
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Initialize LLM client (Azure or OpenAI)
    backend, client, model, _ = make_llm_client()

    # Read input CSV and validate schema
    df = pd.read_csv(args.infile)
    required = {"vignette_id", "vignette_text"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Infile must contain columns: {sorted(required)}")

    # Process each vignette
    rows = []
    for _, r in df.iterrows():
        vid = r["vignette_id"]
        text = str(r["vignette_text"])

        decisions = extract_for_vignette(client, backend, model, text)

        if not decisions:
            # Keep row alignment even if nothing is extracted
            rows.append({"vignette_id": vid, "decision_id": "D1", "decision_text": ""})
            continue

        for i, d in enumerate(decisions, start=1):
            rows.append({"vignette_id": vid, "decision_id": f"D{i}", "decision_text": d})

    # Write (or print) results
    # Long-format dataframe
    long_df = pd.DataFrame(rows, columns=["vignette_id", "decision_id", "decision_text"])

    # Drop empty decisions (placeholders) before pivot
    long_df = long_df[long_df["decision_text"].astype(str).str.len() > 0].copy()

    # Extract numeric index from decision_id like "D1" -> 1, "D2" -> 2
    long_df["decision_idx"] = (
        long_df["decision_id"].str.extract(r'D(\d+)', expand=False).astype(int)
    )

    # Pivot to wide: decision_idx -> decision_{k}
    wide = (
        long_df
        .pivot(index="vignette_id", columns="decision_idx", values="decision_text")
        .sort_index(axis=1)
    )

    # Rename columns to decision_1, decision_2, ...
    wide.columns = [f"decision_{i}" for i in wide.columns]

    # Bring back vignette_text from the original input df
    base = df[["vignette_id", "vignette_text"]].drop_duplicates()
    final_df = base.merge(wide, on="vignette_id", how="left")

    # Reorder columns: vignette_id, vignette_text, decision_1, decision_2 (only)
    decision_cols = [c for c in final_df.columns if c.startswith("decision_")]
    decision_cols = sorted(decision_cols, key=lambda x: int(x.split("_")[1]))[:2]
    final_df = final_df[["vignette_id", "vignette_text", *decision_cols]]

    # Write (or print) the format that rater expects
    if args.dryrun:
        print(final_df.to_csv(index=False))
    else:
        save_path = outdir / args.save_name
        final_df.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
