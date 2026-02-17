import gabriel
import os
import pandas as pd
import argparse
from pathlib import Path
import re

def _check_env():
    has_azure = all(os.environ.get(k) for k in [
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"
    ])
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_model = bool(os.environ.get("OPENAI_MODEL"))  # deployment id or model name
    if not has_model:
        raise RuntimeError("Set OPENAI_MODEL (Azure deployment id or OpenAI model name).")
    if not (has_azure or has_openai):
        raise RuntimeError("Set either Azure envs (AZURE_*) or OPENAI_API_KEY.")

LABELS = {
    "WithinPatient": (
        "Choose this when the vignette’s decisions concern a single identified patient. "
        "Multiple timepoints for the same patient still count as one patient. "
        "Mentions of family/surrogates/clinicians do NOT create additional patients. "
        "References to conserving resources for unspecified 'future patients' do NOT count as a second individuated patient. "
        "If unsure and there is no clear evidence of multiple distinct patients under consideration, prefer WithinPatient."
    ),
    "BetweenPatients": (
        "Choose this when the vignette asks you to compare or allocate a scarce resource or priority between TWO OR MORE different, individuated patients or candidates. "
        "Includes triage, organ allocation priority, choosing who receives a ventilator/ICU bed/dialysis slot, etc. when it is between two or more explicitly mentioned patients"
    ),
}

def get_args():
    ap = argparse.ArgumentParser(
        description="Classify clinical vignettes as WithinPatient vs BetweenPatients using GABRIEL."
    )
    ap.add_argument(
        "-i", "--input", required=True,
        help="Path to CSV with columns: vignette_id, vignette_text."
    )
    ap.add_argument(
        "-o", "--output",
        default="results_vignette_scope.csv",
        help="Path for the output CSV (default: results_vignette_scope.csv)"
    )
    return ap.parse_args()

REQUIRED_INPUT_COLS = ["vignette_id", "vignette_text"]

def load_and_prepare(input_path: str) -> pd.DataFrame:
    # Read CSV (utf-8 -> latin-1 fallback)
    try:
        df = pd.read_csv(input_path)
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding="latin-1")

    # Schema check
    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Expected {REQUIRED_INPUT_COLS}")

    # Normalize types and whitespace
    df["vignette_id"] = df["vignette_id"].astype(str)
    def _norm(s):
        s = "" if pd.isna(s) else str(s)
        return re.sub(r"[ \t]+", " ", s).strip()

    df["vignette_text"] = df["vignette_text"].apply(_norm)

    # The exact string we’ll feed to the LLM
    df["model_input"] = "VIGNETTE:\n" + df["vignette_text"]

    # Quick sanity
    if (df["model_input"].str.len() == 0).any():
        bad = df.loc[df["model_input"].str.len() == 0, "vignette_id"].tolist()[:5]
        raise ValueError(f"Some rows produced empty model_input. Examples: {bad}")

    return df

# --- Block 3: GABRIEL classify + write CSV (binary: WithinPatient / BetweenPatients) ---
import asyncio

async def main(args):
    _check_env()
    model = os.getenv("OPENAI_MODEL")  # Azure deployment id or OpenAI model name

    # Load vignettes and build the 'model_input' column we prepared earlier
    df = load_and_prepare(args.input)

    # Run GABRIEL classification on the 'model_input' text
    results = await gabriel.classify(
        df,
        column_name="model_input",     # <- IMPORTANT: we classify the prepared vignette text
        labels=LABELS,                 # <- your binary label rubric defined above
        save_dir="vignette_scope_runs",
        model=model,
        n_runs=1,
        use_dummy=False,
        reset_files=True,
    )

    # Force a single scope per row, with a clear tie-breaker
    label_cols = list(LABELS.keys())  # ["WithinPatient", "BetweenPatients"]

    def _pick_scope(row):
        flags = [c for c in label_cols if bool(row.get(c))]
        if len(flags) == 1:
            return flags[0]
        if len(flags) == 0:
            # Default conservative assumption when nothing is predicted
            return "WithinPatient"
        # If both True, prefer BetweenPatients (scarcity/triage cases are the priority)
        return "BetweenPatients"

    results["scope"] = results.apply(_pick_scope, axis=1)
    results["predicted_flags"] = (
        results[label_cols].fillna(False).astype(bool)
        .apply(lambda r: ", ".join([c for c in label_cols if r[c]]) or "None", axis=1)
    )

    # Minimal, ordered output
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results[["vignette_id", "scope", "predicted_flags"]].to_csv(out_path, index=False)
    print(f"Wrote {len(results)} rows → {out_path}")

# Optional: simple programmatic wrapper (handy for Streamlit/GUI)
def run_vignette_type_classifier(vignettes_csv: str, output_csv: str):
    class _Args:
        def __init__(self, input, output):
            self.input, self.output = input, output

    _check_env()
    _args = _Args(vignettes_csv, output_csv)
    try:
        asyncio.run(main(_args))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main(_args))
        else:
            raise
    return Path(output_csv)

if __name__ == "__main__":
    args = get_args()
    asyncio.run(main(args))
