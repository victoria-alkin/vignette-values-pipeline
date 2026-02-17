# tools_consistency_runner.py
from pathlib import Path
import pandas as pd
import numpy as np

LIKERT_LABELS = [
    "Strongly Counteracts",
    "Moderately Counteracts",
    "Slightly Counteracts",
    "Neutral",
    "Slightly Promotes",
    "Moderately Promotes",
    "Strongly Promotes",
]

# Colors for PerRun_Likert
LIKERT_FILL_HEX = {
    "Strongly Counteracts": "#E57373",   # medium dark red
    "Moderately Counteracts": "#F2A7AC", # medium light red
    "Slightly Counteracts": "#FFDBE0",   # very light red
    "Neutral": "#E0E0E0",                # light grey
    "Slightly Promotes": "#D2F8DC",      # very light green
    "Moderately Promotes": "#A5D6A7",    # medium light green
    "Strongly Promotes": "#7BC96F",      # medium dark green
}

def sort_vignette_then_decision(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    df = df.copy()

    # vignette_id numeric key (A12 -> 12; non-numeric -> inf so they go last)
    if "vignette_id" in df.columns:
        vid = pd.to_numeric(
            df["vignette_id"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce"
        )
        df["_vid_sort"] = vid.fillna(np.inf)
    else:
        df["_vid_sort"] = np.arange(len(df))

    # decision index key (handle several column names; else all inf so order by vignette only)
    dec_series = None
    if "which_decision" in df.columns:
        dec_series = pd.to_numeric(df["which_decision"], errors="coerce")
    elif "decision_number" in df.columns:
        dec_series = pd.to_numeric(df["decision_number"], errors="coerce")
    elif "decision_id" in df.columns:
        dec_series = pd.to_numeric(df["decision_id"], errors="coerce")
    elif "decision" in df.columns:
        # works for strings like "decision_3"
        dec_series = pd.to_numeric(
            df["decision"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce"
        )

    if dec_series is None:
        df["_dec_sort"] = np.full(len(df), np.inf)
    else:
        df["_dec_sort"] = dec_series.fillna(np.inf).to_numpy()

    df.sort_values(["_vid_sort", "_dec_sort"], kind="mergesort", inplace=True)
    return df.drop(columns=["_vid_sort", "_dec_sort"], errors="ignore")

def _detect_likert_columns(df: pd.DataFrame) -> list[str]:
    """Object columns whose values are a subset of our allowed Likert labels."""
    like_cols = []
    label_set = set(LIKERT_LABELS)
    for c in df.columns:
        if df[c].dtype == object:
            vals = set(str(v) for v in df[c].dropna().unique())
            if vals and vals.issubset(label_set):
                like_cols.append(c)
    return like_cols

def _base_id_columns(df: pd.DataFrame) -> list[str]:
    ids = [c for c in [
        "vignette_id", "vignette_text", "decision", "decision_text",
        "which_decision", "decision_number", "decision_id"
    ] if c in df.columns]
    return ids or []

def _read_run_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = sort_vignette_then_decision(df)
    return df.reset_index().rename(columns={"index": "row_num"})

def _color_formats(workbook):
    fmts = {}
    for label, hex_color in LIKERT_FILL_HEX.items():
        fmts[label] = workbook.add_format({"bg_color": hex_color})
    return fmts

def _write_perrun_sheet(writer, perrun_df: pd.DataFrame):
    perrun_df.to_excel(writer, sheet_name="PerRun_Likert", index=False)
    ws = writer.sheets["PerRun_Likert"]
    wb = writer.book
    fmts = _color_formats(wb)

    run_cols = [i for i, c in enumerate(perrun_df.columns) if "__Run" in str(c)]
    # Color each Likert cell by its label
    for col_ix in run_cols:
        for row_ix in range(1, len(perrun_df) + 1):  # row 0 is header
            val = perrun_df.iloc[row_ix - 1, col_ix]
            if isinstance(val, str) and val in fmts:
                ws.write(row_ix, col_ix, val, fmts[val])

def build_consistency_workbook(run_csvs: list[Path], out_xlsx: Path):
    """
    Given run1_results.csv ... runR_results.csv, create a single XLSX
    with ONLY the color-coded PerRun_Likert sheet.
    """
    if not run_csvs:
        raise ValueError("No run CSVs provided.")

    first = _read_run_csv(run_csvs[0])
    like_cols = _detect_likert_columns(first)
    id_cols = _base_id_columns(first)
    if "row_num" not in id_cols:
        id_cols = ["row_num"] + id_cols

    merged = first[id_cols + like_cols].copy()
    merged.columns = id_cols + [f"{c}__Run1" for c in like_cols]

    # Merge subsequent runs (aligned on id_cols)
    for i, path in enumerate(run_csvs[1:], start=2):
        df_i = _read_run_csv(path)
        # Ensure consistent likert columns
        for c in like_cols:
            if c not in df_i.columns:
                df_i[c] = np.nan
        merged = merged.merge(df_i[id_cols + like_cols], on=id_cols, how="left")
        for c in like_cols:
            merged.rename(columns={c: f"{c}__Run{i}"}, inplace=True)
    
        # --- Reorder columns: group runs by VALUE (principle) ---
    # runs = 1..R inferred from how many CSVs we merged
    runs = list(range(1, len(run_csvs) + 1))

    # Build value-major order: [id cols] + [Autonomy__Run1..R] + [Justice__Run1..R] + ...
    ordered_run_cols = []
    for val in like_cols:
        for r in runs:
            col = f"{val}__Run{r}"
            if col in merged.columns:  # be safe
                ordered_run_cols.append(col)

    # Keep IDs first, then the grouped run columns
    col_order = [c for c in merged.columns if "__Run" not in c] + ordered_run_cols
    merged = merged[col_order]


    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        merged = sort_vignette_then_decision(merged)  # NEW
        _write_perrun_sheet(writer, merged)

    return out_xlsx

def run_replicates_and_build(
    *,
    input_csv: str,
    rater_func,           # callable: run_decision_rater(...)
    replicates: int,
    out_dir: str,
    run_prefix: str,      # "within" or "between"
    n_runs_per_call: int = 1,
    reset_files: bool = True,
):
    """
    Runs the rater 'replicates' times (fresh each time), writes run*_results.csv,
    then builds a single XLSX containing ONLY PerRun_Likert.
    """
    out_dir = Path(out_dir)
    runs_dir = out_dir / f"consistency_{run_prefix}" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_csvs = []
    for r in range(1, replicates + 1):
        out_csv = runs_dir / f"run{r}_results.csv"
        rater_func(
            decisions_wide_csv=input_csv,
            output_csv=str(out_csv),
            n_runs=int(n_runs_per_call),
            reset_files=bool(reset_files),
        )
        # NEW: enforce row order for each run file
        df_run = pd.read_csv(out_csv)
        df_run = sort_vignette_then_decision(df_run)
        df_run.to_csv(out_csv, index=False)

        run_csvs.append(out_csv)

    xlsx_path = out_dir / f"consistency_{run_prefix}" / f"consistency_{run_prefix}.xlsx"
    build_consistency_workbook(run_csvs, xlsx_path)
    return str(xlsx_path), [str(p) for p in run_csvs]