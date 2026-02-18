from gabriel_compatibility import gabriel_compatibility_env
gabriel_compatibility_env()

import os
import re
import asyncio
import pandas as pd
import gabriel
import argparse

# Env and model check
REQUIRED = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "OPENAI_MODEL",
]
def _check_env():
    has_azure = all(os.environ.get(k) for k in [
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"
    ])
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_model = bool(os.environ.get("OPENAI_MODEL"))

    if not has_model:
        raise RuntimeError("Set OPENAI_MODEL (Azure deployment id or OpenAI model name).")
    if not (has_azure or has_openai):
        raise RuntimeError(
            "Set either Azure envs (AZURE_OPENAI_API_KEY/ENDPOINT/API_VERSION) "
            "or OPENAI_API_KEY."
        )

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",  default=r"C:/Users/spark/Dropbox/Victoria/Harvard/G2_Semester1/KohaneLabRotation/vignettes_with_decisions.csv")
parser.add_argument("-o", "--output", default=r"C:/Users/spark/Dropbox/Victoria/Harvard/G2_Semester1/KohaneLabRotation/results_vignette_decision_ratings_ALL6.csv")

# Attributes on –10 to +10 scale
attributes = {
    "Need": (
        """Brief definition: Prioritize those who are worst off or most in need.
        Giving greater weight to patients with the greatest clinical need, severity, or disadvantage (e.g., highest risk without intervention, lack of viable alternatives). Emphasizes “sickest-first”/prioritarian approaches.
        Promoting this includes allocating scarce resources to those who are sickest or with the most urgent needs; expediting care for those who are worst off medically or have no alternatives; prioritizing severe or acute cases.
        Counteracting this includes diverting resources away from the worst-off toward less urgent cases or those with adequate alternatives.
        Rate –10 to +10 how the TARGET DECISION, given the vignette context, affects Need.
        (–10 = deprioritizes the worst-off/most in need; 0 = neutral; +10 = strongly prioritizes the worst-off/most in need)."""
    ),
    "Maximal overall benefit": (
        """Brief definition: Achieve the greatest total good (e.g., most QALYs)..
        Extended definition: Maximizing aggregate health outcomes across patients by choosing options with the largest expected total good (e.g., total QALYs).
        Promoting this includes allocating limited resources to patients with the highest expected benefit or probability of success, and favoring strategies that improve aggregate outcomes across patients with the largest total expected good.
        Counteracting this includes selecting options with lower expected aggregate benefit when higher-yield alternatives are available.
        Rate –10 to +10 how the TARGET DECISION, given the vignette context, affects Maximal overall benefit
        (–10 = reduces total expected benefit; 0 = neutral; +10 = maximizes total expected benefit)."""
    ),
    "Equity": (
        """Brief definition: Ensure proportionate outcomes across groups.
        Extended definition: Striving for equivalent health outcomes by reducing disparities between groups (e.g., by race, socioeconomic status, disability, geography) through proportionate allocation that may differentially support groups to achieve comparable outcomes.
        Promoting this includes prioritizing outreach/resources to underserved groups, removing access barriers, and using rules that narrow outcome gaps.
        Counteracting this includes choices that maintain or widen avoidable disparities across groups.
        Rate –10 to +10 how the TARGET DECISION, given the vignette context, affects Equity
        (–10 = widens disparities; 0 = neutral; +10 = reduces disparities)."""
    ),
    "Equality": (
        """Brief definition: Treat patients identically in distribution of resources.
        Extended definition: Giving each patient identical consideration and equal share/opportunity regardless of personal characteristics; emphasizes rules that treat patients identically, such as first-come/first-served or random selection.
        Promoting this includes using rules such as random allocation (allocating resources randomly through a lottery-based system) and first-come-first-served (allocating resources to those who have been waiting longer, arrived earlier, or called/requested evaluation earlier).
        Counteracting this includes giving unequal consideration to individuals or allocating resources preferentially to particular individuals or groups based on their characteristics.
        Rate –10 to +10 how the TARGET DECISION, given the vignette context, affects Equality
        If the TARGET DECISION uses an equality-based rule such as lottery, first-come-first-served, or waited-longest-first (for example by giving a slot or resource to someone who arrived first, called the clinic first, or has been on a list longer), then it PROMOTES equality and the score should be POSITIVE.
        If the TARGET DECISION chooses NOT to use a lottery or first-come-first-served rule but the alternative option uses a lottery or first-come-first-served rule, then the TARGET DECISION COUNTERACTS equality.
        If neither of the two stated DECISIONS options in the vignette use a lottery or first-come-first-served, then the score for both DECISIONS should be 0.
        (–10 = treats patients unequally; 0 = neutral; +10 = treats patients the same by using a lottery rule or prioritizing those who arrived/called first or waited longest)."""
    ),
    "Reciprocity": (
        """Brief definition: Give priority to those who contribute more (e.g., past organ donors, healthcare workers).
        Extended definition: Recognizing and reciprocating individuals’ past or ongoing contributions or assumed risks that benefit others, by giving them some priority when allocating scarce resources.
        Promoting this includes offering increased priority to individuals who have contributed more, such as frontline workers in healthcare at heightened occupational risk, living organ donors, etc.
        Counteracting this includes ignoring substantial contributions/risks when allocation decisions are made.
        Rate –10 to +10 how the TARGET DECISION, given the vignette context, affects Reciprocity
        (–10 = disregards legitimate contributions/risks; 0 = neutral; +10 = appropriately prioritizes contributors/at-risk helpers)."""
    )
}

additional_instructions = (
    """RATE ONLY the TARGET DECISION using the VIGNETTE as context.

    SCALE & SIGN
    - Integer −10 to +10 per attribute.
    - A positive (+) integer indicates that a decision promotes/advances/prioritizes the attribute.
    − A negative (-) integer indicates that a decision deprioritizes/undermines/counteracts the attribute (perhaps in favor of other attributes).
    - Use the full range when appropriate (±1–2 slight, ±3–4 mild, ±5–6 moderate, ±7–8 strong, ±9–10 very strong).
    - If an ethical principle is not CLEARLY/explicitly relevant to the vignette context and the TARGET DECISION, then the rating MUST be 0.

    GUIDE
    - Evaluate the decision’s effect, not the vignette alone.
    - Base magnitude on explicitness, immediacy, and likely impact.
    - Use the vignette as context. Take note of the alternative decisions as they may contain information on the ethical principles of the decision under consideration.
    - All vignettes will include choosing between two patients.
    - If helping the two patients EQUALLY affects an attribute, then the TARGET DECISION should be rater 0 for that attribute.

    ATTRIBUTES
    - For each attribute/ethical principle, you are given a brief definition, an extended definition, examples of actions that promote that ethical principle, and examples of actions that counteract that ethical principle.
    - Use these descriptions as a guide to discern whether each of the potential decisions promotes or counteracts each ethical principle.
    - For example, if a decision seems to match the actions that are listed as promoting an ethical principle, it should get a positive rating for that ethical principle.
    - For example, if a decision seems to match the actions that are listed as counteracting an ethical principle, it should get a negative rating for that ethical principle.

    COMPARISON CONTEXT
    - Above the TARGET DECISION you will see ALL DECISIONS for the same vignette. Use them as contextual comparators to understand trade-offs and comparisons.
    - Output ratings for the TARGET DECISION.
    - Make sure to look at ALL DECISIONS for a vignette since these represent ALL OPTIONS offered in a particular scenario.
    - If the TARGET DECISION does not promote an attribute more than the decision it is being compared to or counteract an attribute more than the decision it is being compared to, the score should be 0.

    OUTPUT
    - JSON with the exact attribute names defined above and their integer ratings; no extra text."""
)

# Helpers: load and reshape
def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return df

def decision_columns(df: pd.DataFrame):
    # Preserve original column order
    return [c for c in df.columns if re.match(r"(?i)^decision", str(c))]

def build_options_map(df_wide: pd.DataFrame) -> dict:
    """
    For each vignette_id, build a multi-line block listing ALL decisions (full text).
    """
    dcols = decision_columns(df_wide)
    options = {}
    for _, row in df_wide.iterrows():
        vid = row["vignette_id"]
        lines = []
        for i, c in enumerate(dcols, start=1):
            val = row.get(c, None)
            if pd.isna(val):
                continue
            txt = str(val).strip()
            if not txt:
                continue
            lines.append(f"{i}) {txt}")
        block_header = "ALL DECISIONS FOR CONTEXT (do not rate these; rate the TARGET DECISION only):"
        block_body = "\n".join(lines) if lines else "None listed."
        options[vid] = block_header + "\n" + block_body
    return options

def longify_decisions(df: pd.DataFrame) -> pd.DataFrame:
    # Required id/context columns
    if "vignette_id" not in df.columns or "vignette_text" not in df.columns:
        raise ValueError("Input must contain 'vignette_id' and 'vignette_text' columns.")

    dcols = decision_columns(df)
    if not dcols:
        raise ValueError("No decision columns found (expected columns like 'decision_1', 'decision_2', ...).")

    # Build the per-vignette options block
    options_map = build_options_map(df)

    # Long form
    long = df.melt(
        id_vars=["vignette_id", "vignette_text"],
        value_vars=dcols,
        var_name="decision_col",
        value_name="decision",
    )
    long = long.dropna(subset=["decision"])
    long["decision"] = long["decision"].astype(str).str.strip()
    long = long[long["decision"] != ""]

    long["decision_idx"] = long.groupby("vignette_id").cumcount() + 1
    long["decision_id"] = long.apply(lambda r: f"{r['vignette_id']}-d{int(r['decision_idx'])}", axis=1)

    # Compose model input: vignette + all decisions (full text) + target decision (full text)
    long["options_block"] = long["vignette_id"].map(options_map)
    long["text"] = (
        "VIGNETTE (context):\n" + long["vignette_text"].astype(str).str.strip()
        + "\n\n" + long["options_block"]
        + "\n\nTARGET DECISION (rate this using the vignette context):\n" + long["decision"].astype(str).str.strip()
    )

    return long[["vignette_id", "decision_id", "decision", "text"]]

# 7-point Likert labels (negative/counteracts vs positive/promotes)
LIKERT_7 = [
    "Strongly Counteracts",
    "Moderately Counteracts",
    "Slightly Counteracts",
    "Neutral",
    "Slightly Promotes",
    "Moderately Promotes",
    "Strongly Promotes",
]

def score_to_likert_7(x: float) -> str:
    # Map -10..+10 scale to 7 bins
    if x < -7: return LIKERT_7[0]  # strong counteracts
    if x < -4: return LIKERT_7[1]  # moderate counteracts
    if x < -1: return LIKERT_7[2]  # slight counteracts
    if x < 1:  return LIKERT_7[3]  # neutral
    if x <  4: return LIKERT_7[4]  # slight promotes
    if x <  7: return LIKERT_7[5]  # moderate promotes
    return LIKERT_7[6]             # strong promotes

# ---- Programmatic API for GUI use ----
async def rate_decisions(
    infile: str,
    outfile: str,
    *,
    model: str | None = None,
    n_runs: int = 2,
    save_dir: str = "ethics_rate_signed",
    reset_files: bool = True,
):
    _check_env()
    mdl = model or os.getenv("OPENAI_MODEL")
    if not mdl:
        raise RuntimeError("Set OPENAI_MODEL or pass model=...")

    src = load_table(infile)
    long_df = longify_decisions(src)

    rate_df = await gabriel.rate(
        long_df,
        column_name="text",
        attributes=attributes,
        model=mdl,
        n_runs=n_runs,
        save_dir=save_dir,
        reset_files=reset_files,
        use_dummy=False,
        additional_instructions=additional_instructions,
    )

    # ----- Post-processing -----
    label_cols = list(attributes.keys())
    scores = (
        rate_df[label_cols]
        .apply(pd.to_numeric, errors="coerce")
        .clip(lower=-10, upper=10)
        .fillna(0)
    )
    for c in label_cols:
        rate_df[c + "_likert"] = rate_df[c].apply(score_to_likert_7)
    thr = 0
    rate_df["promotes"]    = scores.apply(lambda r: ", ".join(r.index[r >  thr]) or "None", axis=1)
    rate_df["counteracts"] = scores.apply(lambda r: ", ".join(r.index[r < -thr]) or "None", axis=1)
    rate_df["best_abs"]   = scores.abs().max(axis=1)
    rate_df["best_score"] = scores.apply(lambda r: r[r.abs().idxmax()], axis=1)

    tie_mask = scores.abs().eq(rate_df["best_abs"], axis=0)
    def _format_ties(row):
        if row["best_abs"] <= 0: return "None"
        names = []
        for c in scores.columns:
            if tie_mask.loc[row.name, c]:
                v = scores.loc[row.name, c]
                sign = "+" if v > 0 else ""
                pretty = int(v) if float(v).is_integer() else round(float(v), 2)
                names.append(f"{c} ({sign}{pretty})")
        return ", ".join(names)
    rate_df["primary_principle(s)"] = rate_df.apply(_format_ties, axis=1)

    likert_cols = [c + "_likert" for c in label_cols]
    cols_order = [
        "vignette_id", "decision_id", "decision",
        *likert_cols, "promotes", "counteracts", "primary_principle(s)",
        "best_score", "best_abs", *label_cols
    ]
    cols_order = [c for c in cols_order if c in rate_df.columns] + \
                 [c for c in rate_df.columns if c not in cols_order]
    rate_df = rate_df[cols_order]

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    rate_df.to_csv(outfile, index=False)
    return outfile

def run_decision_rater(
    decisions_wide_csv: str,
    output_csv: str,
    **kwargs
):
    """Sync wrapper for GUI / notebooks."""
    return asyncio.run(rate_decisions(decisions_wide_csv, output_csv, **kwargs))

# Main
async def main():
    _check_env()
    args = parser.parse_args()
    model = os.getenv("OPENAI_MODEL")
    await rate_decisions(args.input, args.output, model=model)

if __name__ == "__main__":
    asyncio.run(main())