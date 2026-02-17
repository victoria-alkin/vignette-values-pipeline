import gabriel
import os
import pandas as pd
import asyncio
import argparse
from pathlib import Path

def _check_env():
    has_azure = all(os.environ.get(k) for k in [
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"
    ])
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_model = bool(os.environ.get("OPENAI_MODEL"))
    if not has_model:
        raise RuntimeError("Set OPENAI_MODEL (Azure deployment id or OpenAI model name).")
    if not (has_azure or has_openai):
        raise RuntimeError("Set either Azure envs (AZURE_*) or OPENAI_API_KEY.")

#Labels for ethical principle classes
labels = {
    "Need": "The factor discusses or demonstrates which patient(s) are worst off or most in need.",
    "Maximal overall benefit": "The factor reflects which decision will achieve the greatest total good (e.g., most QALYs), for example by mentioning QALYs, etc.",
    "Equity": "The factor references ensuring proportionate outcomes across groups.",
    "Equality": "The factor relates to treating patients identically in distribution of resources.",
    "Reciprocity": "The factor references giving priority to those who contribute more (e.g., healthcare workers, organ donors), for example by referencing past contributions including organ donations or work in the healthcare field."
}

def get_args():
    ap = argparse.ArgumentParser(description="Classify contextual factors by ethical principles and save CSV.")
    ap.add_argument(
        "-o", "--output",
        default=r"C:/Users/spark/Dropbox/Victoria/Harvard/G2_Semester1/KohaneLabRotation/resultstest1.csv",
        help="Path to write the results CSV"
    )
    ap.add_argument(
        "-i", "--input",
        required=True,
        help="Path to contextual factors CSV from extractor (columns: vignette_id,factor_id,text)"
    )
    return ap.parse_args()

#Main call
async def main(args):
    _check_env()
    model = os.getenv("OPENAI_MODEL")
    # Load factors CSV created by the extractor and validate schema
    df = pd.read_csv(args.input)
    required_cols = {"vignette_id", "factor_id", "text"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input is missing required columns: {missing_cols}. Expected {required_cols}.")

    results = await gabriel.classify(
        df,
        column_name="text",   # name of column with text to classify
        labels=labels,        # dictionary of label definitions, defined above
        save_dir="test_classify",  # directory to save results
        model=model,       # model used for classification
        n_runs=3, # number of classification passes per text
        use_dummy=False,
        reset_files=True,
    )
    label_cols = list(labels.keys())
    results["predicted_classes"] = (
        results[label_cols].fillna(False).astype(bool)
        .apply(lambda row: ", ".join([c for c in label_cols if row[c]]) or "None", axis=1)
    )
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

# ---- Programmatic wrapper for GUI use ----
def run_factor_classifier(factors_csv: str, output_csv: str):
    class _Args:
        def __init__(self, input, output): self.input, self.output = input, output
    _check_env()
    _args = _Args(factors_csv, output_csv)
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