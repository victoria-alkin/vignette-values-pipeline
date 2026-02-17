# app.py — Step 1: env check + input upload/preview
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from datetime import datetime
import shutil
from typing import Union
from pandas.io.formats.style import Styler

st.set_page_config(page_title="Medical Vignettes — Ethics Pipeline", layout="wide")
PREVIEW_HEIGHT = 300

# --- Backend/env status (sidebar) ---
def detect_backend():
    has_azure = all(os.getenv(k) for k in [
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"
    ])
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL")

    if not model:
        return ("error", "Model not set",
                "Set OPENAI_MODEL (Azure deployment id or OpenAI model name).")
    if not (has_azure or has_openai):
        return ("error", "No API credentials",
                "Set Azure (AZURE_OPENAI_*) or OPENAI_API_KEY.")
    if has_azure:
        return ("success", "Azure OpenAI", f"Using deployment: {model}")
    else:
        return ("success", "OpenAI", f"Using model: {model}")

def download_csv_button(label: str, path: Union[str, Path], key: str):
    p = Path(path)
    if p.exists():
        st.download_button(
            label,
            data=p.read_bytes(),
            file_name=p.name,
            mime="text/csv",
            key=key,
        )

def _safe_len_csv(path: Union[str, Path], usecols=None) -> int:
    p = Path(path)
    if not p.exists(): 
        return 0
    try:
        if usecols:
            # stream in chunks to avoid loading whole file
            total = 0
            for chunk in pd.read_csv(p, usecols=usecols, chunksize=10_000):
                total += len(chunk)
            return total
        return len(pd.read_csv(p))
    except Exception:
        return 0

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

# ---- Likert styling for GUI previews ----
RATING_LABELS = [
    "Strongly Counteracts","Moderately Counteracts","Slightly Counteracts",
    "Neutral","Slightly Promotes","Moderately Promotes","Strongly Promotes",
]

# Pick the palette you want to mirror. This one matches your Excel summary workbook.
RATING_COLORS = {
    "Neutral":               "#E0E0E0",
    "Slightly Promotes":     "#D2F8DC",
    "Moderately Promotes":   "#A5D6A7",
    "Strongly Promotes":     "#81C784",
    "Slightly Counteracts":  "#FFDBE0",
    "Moderately Counteracts":"#F2A7AC",
    "Strongly Counteracts":  "#E57373",
}

def _detect_likert_columns(df: pd.DataFrame) -> list[str]:
    label_set = set(RATING_LABELS)
    like_cols = []
    for c in df.columns:
        if df[c].dtype == object:
            uniq = set(map(str, df[c].dropna().unique().tolist()))
            if uniq and uniq.issubset(label_set):
                like_cols.append(c)
    return like_cols

def style_likert(df: pd.DataFrame) -> Styler:
    like_cols = _detect_likert_columns(df)
    styler = df.style
    if not like_cols:
        return styler  # nothing to color
    def color_cell(v):
        return f"background-color: {RATING_COLORS.get(str(v), '')}"
    return styler.applymap(color_cell, subset=like_cols)

# --- Summary workbook builder (XLSX) ---
def build_summary_workbook(run_dir: Path) -> Path:
    """
    Creates summary.xlsx in run_dir with the requested sheets and formatting.
    Skips any missing CSVs and returns the path to the saved workbook.
    """
    import numpy as np

    run_dir = Path(run_dir)
    out_xlsx = run_dir / "summary.xlsx"

    # Expected files -> sheet names
    factors_dir = run_dir / "factors"
    decisions_dir = run_dir / "decisions"
    vt_dir = run_dir / "vignette_type"

    sources = [
        (vt_dir / "results_vignette_scope.csv",          "vignette_scopes"),
        (factors_dir / "factors_extracted.csv",          "factors_extracted"),
        (factors_dir / "factors_classified_within.csv",  "factors_classified_within"),
        (factors_dir / "factors_classified_between.csv", "factors_classified_between"),
        (factors_dir / "factors_classified_all.csv",     "factors_classified_all"),
        (decisions_dir / "decisions_extracted.csv",      "decisions_extracted"),
        (decisions_dir / "decisions_rated_within.csv",   "decisions_rated_within"),
        (decisions_dir / "decisions_rated_between.csv",  "decisions_rated_between"),
        (decisions_dir / "decisions_rated_all.csv",      "decisions_rated_all"),
    ]

    # Ratings palette (exact-capitalization labels)
    rating_labels = [
        "Strongly Counteracts",
        "Moderately Counteracts",
        "Slightly Counteracts",
        "Neutral",
        "Slightly Promotes",
        "Moderately Promotes",
        "Strongly Promotes",
    ]
    colors = {
        "Neutral":               "#E0E0E0",  # light grey
        "Slightly Promotes":     "#D2F8DC",  # very light green
        "Moderately Promotes":   "#A5D6A7",  # medium-light green
        "Strongly Promotes":     "#81C784",  # medium-dark green
        "Slightly Counteracts":  "#FFDBE0",  # very light red
        "Moderately Counteracts":"#F2A7AC",  # medium-light red
        "Strongly Counteracts":  "#E57373",  # medium-dark red
    }

    # Load available dataframes
    loaded = []
    missing = []
    for path, sheet in sources:
        try:
            if path.exists():
                df = pd.read_csv(path)
                loaded.append((sheet, df))
            else:
                missing.append((sheet, str(path)))
        except Exception as e:
            missing.append((sheet, f"{path} (error: {e})"))

    if missing:
        for sheet, note in missing:
            st.warning(f"Summary: skipping '{sheet}' — missing or unreadable: {note}")

    # Write workbook with XlsxWriter
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        workbook  = writer.book
        # Formats
        bold_fmt  = workbook.add_format({"bold": True})
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#F5F5F5", "border": 1})
        # Rating formats by label
        rating_fmts = {lab: workbook.add_format({"bg_color": col}) for lab, col in colors.items()}

        def write_df(sheet_name: str, df: pd.DataFrame):
            # Write DF with a header format
            df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1, startcol=0, header=False)
            ws = writer.sheets[sheet_name]
            # Write headers with style
            for c, col in enumerate(df.columns):
                ws.write(0, c, col, header_fmt)

            # Autofilter + freeze header
            nrows, ncols = df.shape
            if ncols > 0 and nrows > 0:
                ws.autofilter(0, 0, nrows, ncols - 1)
            ws.freeze_panes(1, 0)

            # Reasonable column widths
            for c, col in enumerate(df.columns):
                width = min(50, max(10, int(df[col].astype(str).map(len).max() if len(df) else 10)))
                ws.set_column(c, c, width)

            return ws, nrows, ncols

        # Helper: bold TRUE entries (covers True, "True", "TRUE")
        def bold_true_cells(ws, df: pd.DataFrame, start_row=1, start_col=0):
            nrows, ncols = df.shape
            if nrows == 0 or ncols == 0:
                return
            for c, col in enumerate(df.columns):
                # Apply only to "boolean-like" columns
                col_vals = df[col].dropna().unique().tolist()
                # If all values are in {True, False, "True", "False"} treat as boolean-like
                boolish = set(
                    [v if isinstance(v, (bool, np.bool_)) else str(v) for v in col_vals]
                ).issubset({True, False, "True", "False", "TRUE", "FALSE"})
                if not boolish:
                    continue
                # Range for this column (data only)
                first_row = start_row + 1
                last_row  = start_row + nrows
                cell_range = f"{xlsx_col(start_col + c)}{first_row}:{xlsx_col(start_col + c)}{last_row}"

                # Bold if text contains "True"
                ws.conditional_format(cell_range, {
                    "type": "text",
                    "criteria": "containing",
                    "value": "True",
                    "format": bold_fmt
                })
                ws.conditional_format(cell_range, {
                    "type": "text",
                    "criteria": "containing",
                    "value": "TRUE",
                    "format": bold_fmt
                })
                # Bold if the cell is a real boolean TRUE
                ws.conditional_format(cell_range, {
                    "type": "cell",
                    "criteria": "==",
                    "value": True,
                    "format": bold_fmt
                })

        # Helper: color rating columns that contain ONLY the known labels
        def color_rating_columns(ws, df: pd.DataFrame, start_row=1, start_col=0):
            nrows, ncols = df.shape
            if nrows == 0 or ncols == 0:
                return
            label_set = set(rating_labels)
            for c, col in enumerate(df.columns):
                series = df[col]
                if series.dtype == object:
                    # Unique non-null values
                    uniq = set(map(str, series.dropna().unique().tolist()))
                    if uniq and uniq.issubset(label_set):
                        # Apply text conditional formats for each label
                        first_row = start_row + 1
                        last_row  = start_row + nrows
                        cell_range = f"{xlsx_col(start_col + c)}{first_row}:{xlsx_col(start_col + c)}{last_row}"
                        for lab in rating_labels:
                            ws.conditional_format(cell_range, {
                                "type": "text",
                                "criteria": "containing",
                                "value": lab,
                                "format": rating_fmts[lab]
                            })

        # Utilities: Excel column letters
        def xlsx_col(idx: int) -> str:
            # 0->A, 1->B, ...
            s = ""
            idx += 1
            while idx:
                idx, rem = divmod(idx - 1, 26)
                s = chr(65 + rem) + s
            return s

        # Write each available sheet with appropriate styling
        for sheet, df in loaded:
            if sheet.startswith("decisions_rated") or sheet == "decisions_extracted":
                df = sort_vignette_then_decision(df)  # NEW
            ws, nrows, ncols = write_df(sheet, df)

            # Bolding for factors_classified_* sheets
            if sheet in ("factors_classified_within", "factors_classified_between", "factors_classified_all"):
                bold_true_cells(ws, df)

            # Color ratings for decisions_rated_* sheets
            if sheet in ("decisions_rated_within", "decisions_rated_between", "decisions_rated_all"):
                color_rating_columns(ws, df)

    return out_xlsx


with st.sidebar:
    st.subheader("Output")
    base_out = st.text_input("Output folder", value=str(Path("runs").absolute()))
    run_name = st.text_input("Run name", value=datetime.now().strftime("%Y%m%d_%H%M%S"))
    # One-click ZIP of the current run directory (if it exists)
    if "run_dir" in st.session_state:
        run_dir = Path(st.session_state["run_dir"])
        if st.button("Create run ZIP"):
            zip_base = run_dir / "artifacts"  # base name (no .zip)
            archive_path = shutil.make_archive(str(zip_base), "zip", root_dir=run_dir)
            st.session_state["run_zip"] = archive_path
            st.success(f"Created: {archive_path}")

        if "run_zip" in st.session_state and Path(st.session_state["run_zip"]).exists():
            st.download_button(
                "Download run ZIP",
                data=Path(st.session_state["run_zip"]).read_bytes(),
                file_name=Path(st.session_state["run_zip"]).name,
                mime="application/zip",
                key="dl_run_zip",
            )
    else:
        st.caption("Run something first to enable ZIP.")


with st.sidebar:
    st.title("Environment")
    status, title, detail = detect_backend()
    if status == "success":
        st.success(f"{title}\n\n{detail}")
    else:
        st.error(f"{title}\n\n{detail}")

st.title("Human Values Project — Vignette Ethics")
st.write("Upload a table with columns **vignette_id** and **vignette_text**.")

# --- File uploader ---
uploaded = st.file_uploader("Upload CSV / XLSX / JSONL", type=["csv", "xlsx", "xls", "jsonl", "ndjson"])

def read_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    if name.endswith((".jsonl", ".ndjson")):
        return pd.read_json(uploaded_file, lines=True)
    raise ValueError("Unsupported file type (use csv/xlsx/jsonl).")

if uploaded:
    try:
        # Clear only if it's a NEW file (filename changed)
        is_new_file = (st.session_state.get("input_name") != uploaded.name)
        if is_new_file:
            stale_keys = [
                "vignette_scope_csv",              # Step 1
                "factors_csv",                     # Step 2A
                "factors_classified_csv",          # Step 2B (merged)
                "factors_classified_within_csv",   # Step 2B (within)
                "factors_classified_between_csv",  # Step 2B (between)
                "decisions_csv",                   # Step 3A
                "decisions_rated_csv",             # Step 3B (merged)
                "decisions_rated_within_csv",      # Step 3B (within)
                "decisions_rated_between_csv",     # Step 3B (between)
                "run_zip",
                "run_dir",                         # optional: force a fresh run dir per upload
            ]
            for k in stale_keys:
                st.session_state.pop(k, None)

        df = read_table(uploaded)
        required = {"vignette_id", "vignette_text"}
        missing = required - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {sorted(missing)}. Found: {list(df.columns)}")
        else:
            st.success(f"Loaded {len(df):,} vignettes.")
            st.dataframe(df.head(20), use_container_width=True)
            st.session_state["input_df"] = df
            st.session_state["input_name"] = uploaded.name
    except Exception as e:
        st.error(f"Could not read file: {e}")

# Step 1 — Vignette Type (Within Patient vs Between Patient Scenario)
st.divider()
st.header("Step 1 — Vignette Type (Within Patient vs Between Patient Scenario)")

if "input_df" not in st.session_state:
    st.info("Upload a vignette file above to enable this step.")
else:
    df = st.session_state["input_df"]

    cols = st.columns([1, 1, 2])
    with cols[0]:
        limit_on_scope = st.checkbox("Limit N vignettes", value=True, key="limit_on_scope")
    with cols[1]:
        max_n_scope = len(df)
        n_limit_scope = st.number_input(
            "Run on the first N vignettes",
            min_value=1, max_value=max_n_scope,
            value=min(20, max_n_scope), step=1,
            disabled=not limit_on_scope,
            key="n_limit_scope",
        )
    with cols[2]:
        st.caption("Output: one row per vignette with columns: vignette_id, scope, predicted_flags.")

    if st.button("Run vignette type classifier", type="primary", disabled=(status != "success")):
        # Lazy import so initial app load is fast
        from vignette_type_classifier import run_vignette_type_classifier

        # Prepare run directory
        run_dir = Path(st.session_state.get("run_dir", Path(base_out) / run_name))
        vt_dir = run_dir / "vignette_type"
        run_dir.mkdir(parents=True, exist_ok=True)
        vt_dir.mkdir(parents=True, exist_ok=True)

        # Write the working input CSV (respect optional limit)
        work_df = df.head(int(n_limit_scope)) if limit_on_scope else df
        infile_path = run_dir / "input_vignettes.csv"
        work_df.to_csv(infile_path, index=False)

        # Output path
        out_path = vt_dir / "results_vignette_scope.csv"

        with st.spinner("Classifying vignette type…"):
            try:
                out = run_vignette_type_classifier(str(infile_path), str(out_path))
                st.session_state["run_dir"] = str(run_dir)
                st.session_state["vignette_scope_csv"] = str(out_path)
                st.success(f"Done. Wrote: {out}")
            except Exception as e:
                st.error(f"Vignette type classifier failed: {e}")

# --- Persistent preview for Step 1 (always shows if we have output) ---
if "vignette_scope_csv" in st.session_state:
    try:
        scope_df = pd.read_csv(st.session_state["vignette_scope_csv"])
        scope_df_disp = scope_df.rename(columns={"scope": "Vignette Type"})

        # Show only the renamed version
        st.dataframe(scope_df_disp.head(20), use_container_width=True, height=PREVIEW_HEIGHT)

        # Quick counts by label (binary only), using the renamed column
        if "Vignette Type" in scope_df_disp.columns:
            counts = scope_df_disp["Vignette Type"].value_counts(dropna=False)
            bp = int(counts.get("BetweenPatients", 0))
            wp = int(counts.get("WithinPatient", 0))
            cols = st.columns(2)
            cols[0].metric("BetweenPatients", f"{bp:,}")
            cols[1].metric("WithinPatient", f"{wp:,}")
    except Exception as e:
        st.warning(f"Could not preview vignette type CSV: {e}")

# Download button
if "vignette_scope_csv" in st.session_state:
    download_csv_button(
        "Download vignette type CSV",
        st.session_state["vignette_scope_csv"],
        key="dl_vignette_scope_persist",
    )


# Step 2 - Contextual Factors
st.divider()
st.header("Step 2A — Extract Contextual Factors")

disabled = "input_df" not in st.session_state
if disabled:
    st.info("Upload a vignette file above to enable this step.")
else:
    df = st.session_state["input_df"]

    cols = st.columns([1,2])
    with cols[0]:
         limit_on = st.checkbox("Limit N vignettes", value=True)
    with cols[1]:
        max_n = len(df)
        n_limit = st.number_input(
            "Run on the first N vignettes",
            min_value=1, max_value=max_n,
            value=min(20, max_n), step=1,
            disabled=not limit_on
        )

    if st.button("Run extractor", type="primary", disabled=(status != "success")):
        from contextual_factor_extraction import run_factor_extractor
        # Prepare run directory + input file on disk
        run_dir = Path(st.session_state.get("run_dir", Path(base_out) / run_name))
        st.session_state["run_dir"] = str(run_dir)
        factors_dir = run_dir / "factors"
        run_dir.mkdir(parents=True, exist_ok=True)
        factors_dir.mkdir(parents=True, exist_ok=True)

        # Write the working input CSV (respect optional limit)
        work_df = df.head(int(n_limit)) if limit_on else df
        infile_path = run_dir / "input_vignettes.csv"
        work_df.to_csv(infile_path, index=False)

        # Call your wrapper
        with st.spinner("Extracting contextual factors…"):
            try:
                out_csv = run_factor_extractor(
                    infile=str(infile_path),
                    outdir=str(factors_dir),
                    dedupe=True,
                    save_name="factors_extracted.csv",
                )
                st.session_state["factors_csv"] = str(out_csv)
                st.session_state["run_dir"] = str(run_dir)
                st.success(f"Done. Wrote: {out_csv}")
            except Exception as e:
                st.error(f"Extractor failed: {e}")
# --- Persistent preview for Step 2A (always shows if we have output) ---
if "factors_csv" in st.session_state:
    try:
        df_all = pd.read_csv(st.session_state["factors_csv"])

        if "vignette_id" not in df_all.columns:
            st.warning("Extracted factors file is missing 'vignette_id'. Showing first 20 rows.")
            st.dataframe(df_all.head(20), use_container_width=True, height=PREVIEW_HEIGHT)
        else:
            vid_options = [str(v) for v in sorted(df_all["vignette_id"].astype(str).unique())]
            chosen_vid = st.selectbox(
                "Preview contextual factors per vignette_id",
                vid_options,
                index=0,
                key="prev_step2a_vid",
            )
            view = df_all[df_all["vignette_id"].astype(str) == chosen_vid]
            st.dataframe(view, use_container_width=True, height=PREVIEW_HEIGHT)
    except Exception as e:
        st.warning(f"Could not preview factors CSV: {e}")
# Persistent download: raw extracted factors
if "factors_csv" in st.session_state:
    download_csv_button(
        "Download extracted contextual factors CSV",
        st.session_state["factors_csv"],
        key="dl_factors_extracted_persist",
    )

st.divider()
st.header("Step 2B — Classify Contextual Factors (by Vignette Type)")

if "factors_csv" not in st.session_state:
    st.info("Run the extractor first to enable this step.")
elif "vignette_scope_csv" not in st.session_state:
    st.info("Run Step 1 (Vignette Type) first to enable branching here.")
else:
    factors_csv = st.session_state["factors_csv"]
    run_dir = Path(st.session_state["run_dir"])
    factors_dir = run_dir / "factors"

    # Standard output paths
    out_within = factors_dir / "factors_classified_within.csv"
    out_between = factors_dir / "factors_classified_between.csv"
    out_all = factors_dir / "factors_classified_all.csv"

    # Controls
    cols = st.columns([2, 2, 2])
    with cols[0]:
        st.caption("Outputs will be written to the run folder:")
        st.code(str(factors_dir), language="text")
    with cols[1]:
        st.caption("File names:")
        st.code(
            "factors_classified_within.csv\n"
            "factors_classified_between.csv\n"
            "factors_classified_all.csv",
            language="text",
        )
    with cols[2]:
        st.caption("Each branch runs only on its Vignette Type subset, joined by vignette_id.")

    if st.button("Run factor classifier by Vignette Type", type="primary", disabled=(status != "success")):
        # Lazy imports for speed
        from classifier_ethics_withinpatient import run_factor_classifier as run_factor_classifier_within
        from classifier_ethics_betweenpatients import run_factor_classifier as run_factor_classifier_between

        with st.spinner("Preparing subsets…"):
            scope_df = pd.read_csv(st.session_state["vignette_scope_csv"])[["vignette_id", "scope"]]
            scope_df["vignette_id"] = scope_df["vignette_id"].astype(str)

            fac_df = pd.read_csv(factors_csv)
            fac_df["vignette_id"] = fac_df["vignette_id"].astype(str)

            joined = fac_df.merge(scope_df, on="vignette_id", how="inner")
            within_df = joined[joined["scope"] == "WithinPatient"].copy()
            between_df = joined[joined["scope"] == "BetweenPatients"].copy()

            # Write subset inputs to disk for the two branch calls
            in_within = factors_dir / "factors_within_input.csv"
            in_between = factors_dir / "factors_between_input.csv"
            if len(within_df):
                within_df[["vignette_id", "factor_id", "text"]].to_csv(in_within, index=False)
            if len(between_df):
                between_df[["vignette_id", "factor_id", "text"]].to_csv(in_between, index=False)

        # Branch: WithinPatient
        if len(within_df):
            with st.spinner(f"Classifying factors (WithinPatient)… ({len(within_df):,} rows)"):
                try:
                    run_factor_classifier_within(str(in_within), str(out_within))
                    st.success(f"Wrote: {out_within}")
                    st.session_state["factors_classified_within_csv"] = str(out_within)
                except Exception as e:
                    st.error(f"WithinPatient classifier failed: {e}")
        else:
            st.info("WithinPatient subset: 0 rows — skipped.")

        # Branch: BetweenPatients
        if len(between_df):
            with st.spinner(f"Classifying factors (BetweenPatients)… ({len(between_df):,} rows)"):
                try:
                    run_factor_classifier_between(str(in_between), str(out_between))
                    st.success(f"Wrote: {out_between}")
                    st.session_state["factors_classified_between_csv"] = str(out_between)
                except Exception as e:
                    st.error(f"BetweenPatients classifier failed: {e}")
        else:
            st.info("BetweenPatients subset: 0 rows — skipped.")

        # Merge outputs (attach Vignette Type) — force consistent dtype
        try:
            parts = []

            # read scope table with vignette_id as str
            scope_df = pd.read_csv(
                st.session_state["vignette_scope_csv"],
                usecols=["vignette_id", "scope"],
                dtype={"vignette_id": str},
            )

            if out_within.exists():
                w = pd.read_csv(out_within, dtype={"vignette_id": str})
                w = w.merge(scope_df, on="vignette_id", how="left").rename(columns={"scope": "Vignette Type"})
                parts.append(w)

            if out_between.exists():
                b = pd.read_csv(out_between, dtype={"vignette_id": str})
                b = b.merge(scope_df, on="vignette_id", how="left").rename(columns={"scope": "Vignette Type"})
                parts.append(b)

            if parts:
                merged = pd.concat(parts, ignore_index=True)
                merged.to_csv(out_all, index=False)
                st.success(f"Merged: {out_all}")
                st.session_state["factors_classified_csv"] = str(out_all)
        except Exception as e:
            st.warning(f"Could not build merged factors file: {e}")

# --- Persistent preview for Step 2B (by Vignette Type) ---
if "run_dir" in st.session_state:
    run_dir = Path(st.session_state["run_dir"])
    factors_dir = run_dir / "factors"
    out_within = factors_dir / "factors_classified_within.csv"
    out_between = factors_dir / "factors_classified_between.csv"
    out_all = factors_dir / "factors_classified_all.csv"

    tabs = st.tabs(["WithinPatient", "BetweenPatients", "Merged (all)"])
    with tabs[0]:
        if out_within.exists():
            _df = pd.read_csv(out_within)
            _df = sort_vignette_then_decision(_df)  # NEW
            st.dataframe(_df.head(20), use_container_width=True, height=PREVIEW_HEIGHT)
            download_csv_button("Download factors_classified_within.csv", out_within, key="dl_factors_within")
        else:
            st.caption("No WithinPatient output yet.")
    with tabs[1]:
        if out_between.exists():
            _df = pd.read_csv(out_between)
            _df = sort_vignette_then_decision(_df)  # NEW
            st.dataframe(_df.head(20), use_container_width=True, height=PREVIEW_HEIGHT)
            download_csv_button("Download factors_classified_between.csv", out_between, key="dl_factors_between")
        else:
            st.caption("No BetweenPatients output yet.")
    with tabs[2]:
        if out_all.exists():
            _df = pd.read_csv(out_all)
            _df = sort_vignette_then_decision(_df)  # NEW
            st.dataframe(_df.head(20), use_container_width=True, height=PREVIEW_HEIGHT)
            download_csv_button("Download factors_classified_all.csv", out_all, key="dl_factors_all")
        else:
            st.caption("No merged output yet.")

st.divider()
st.header("Step 3A — Extract Decisions")

if "input_df" not in st.session_state:
    st.info("Upload a vignette file above to enable this step.")
else:
    df = st.session_state["input_df"]

    cols = st.columns([1,1,2])
    with cols[0]:
        limit_on_dec = st.checkbox(
            "Process only the first N vignettes",
            value=True,
            key="limit_on_dec",
            help="When on, only the first N rows (vignettes) from the uploaded file are sent to the decision extractor."
        )
    with cols[1]:
        max_n_dec = len(df)
        n_limit_dec = st.number_input(
            "N vignettes to process",
            min_value=1,
            max_value=max_n_dec,
            value=min(20, max_n_dec),
            step=1,
            disabled=not limit_on_dec,
            key="n_limit_dec",
            help="This limits the number of vignettes, not the number of decision options per vignette."
        )
    with cols[2]:
        st.caption("Output: one row per vignette with columns decision_1..k.")


    if st.button("Run decision extractor", type="primary", disabled=(status != "success")):
        # Lazy import to speed initial load
        from decision_extraction import run_decision_extractor

        # Prepare run directory
        run_dir = Path(st.session_state.get("run_dir", Path(base_out) / run_name))
        decisions_dir = run_dir / "decisions"
        run_dir.mkdir(parents=True, exist_ok=True)
        decisions_dir.mkdir(parents=True, exist_ok=True)

        # Write working input CSV (respect optional limit)
        work_df = df.head(int(n_limit_dec)) if limit_on_dec else df
        infile_path = run_dir / "input_vignettes.csv"
        work_df.to_csv(infile_path, index=False)

        with st.spinner("Extracting decisions…"):
            try:
                out_csv = run_decision_extractor(
                    infile=str(infile_path),
                    outdir=str(decisions_dir),
                    save_name="decisions_extracted.csv",
                )
                st.session_state["run_dir"] = str(run_dir)
                st.session_state["decisions_csv"] = str(out_csv)
                st.success(f"Done. Wrote: {out_csv}")
            except Exception as e:
                st.error(f"Decision extractor failed: {e}")
# Persistent download: wide decisions CSV
if "decisions_csv" in st.session_state:
    download_csv_button(
        "Download decisions CSV",
        st.session_state["decisions_csv"],
        key="dl_decisions_wide_persist",
    )
# --- Persistent preview for Step 3A (always shows if we have output) ---
if "decisions_csv" in st.session_state:
    try:
        df_dec = pd.read_csv(st.session_state["decisions_csv"])
        if "vignette_id" in df_dec.columns:
            vid_options_3a = [str(v) for v in sorted(df_dec["vignette_id"].astype(str).unique())]
            chosen_vid_3a = st.selectbox(
                "Preview decisions per vignette_id",
                vid_options_3a,
                index=0,
                key="prev_step3a_vid",
            )
            view_3a = df_dec[df_dec["vignette_id"].astype(str) == chosen_vid_3a]
        else:
            st.warning("Decisions file is missing 'vignette_id'. Showing first 20 rows.")
            view_3a = df_dec.head(20)
        st.dataframe(view_3a, use_container_width=True, height=PREVIEW_HEIGHT)
    except Exception as e:
        st.warning(f"Could not preview decisions CSV: {e}")

st.divider()
st.header("Step 3B — Rate Decisions by Ethical Principles (by Vignette Type)")

if "decisions_csv" not in st.session_state:
    st.info("Run the decision extractor first to enable this step.")
elif "vignette_scope_csv" not in st.session_state:
    st.info("Run Step 1 (Vignette Type) first to enable branching here.")
else:
    decisions_csv = st.session_state["decisions_csv"]
    run_dir = Path(st.session_state["run_dir"])
    decisions_dir = run_dir / "decisions"

    # Output paths
    out_within = decisions_dir / "decisions_rated_within.csv"
    out_between = decisions_dir / "decisions_rated_between.csv"
    out_all = decisions_dir / "decisions_rated_all.csv"

    # Controls
    cols = st.columns([1, 2])
    with cols[0]:
        n_runs = st.number_input("n_runs (per decision)", min_value=1, max_value=5, value=2, step=1, key="rate_n_runs")
    with cols[1]:
        st.caption("Each branch runs only on its Vignette Type subset, joined by vignette_id.")

    if st.button("Run decision rater by Vignette Type", type="primary", disabled=(status != "success")):
        # Lazy imports
        from decisions_rater_withinpatient import run_decision_rater as run_decision_rater_within
        from decisions_rater_betweenpatients import run_decision_rater as run_decision_rater_between

        with st.spinner("Preparing subsets…"):
            scope_df = pd.read_csv(st.session_state["vignette_scope_csv"])[["vignette_id", "scope"]]
            scope_df["vignette_id"] = scope_df["vignette_id"].astype(str)

            dec_df = pd.read_csv(decisions_csv)
            dec_df["vignette_id"] = dec_df["vignette_id"].astype(str)

            joined = dec_df.merge(scope_df, on="vignette_id", how="inner")
            within_df = joined[joined["scope"] == "WithinPatient"].copy()
            between_df = joined[joined["scope"] == "BetweenPatients"].copy()

            # Write subset inputs (wide) for the two branch calls
            in_within = decisions_dir / "decisions_within_input.csv"
            in_between = decisions_dir / "decisions_between_input.csv"
            if len(within_df):
                within_df.to_csv(in_within, index=False)
            if len(between_df):
                between_df.to_csv(in_between, index=False)

        # Branch: WithinPatient
        if len(within_df):
            with st.spinner(f"Rating decisions (WithinPatient)… ({len(within_df):,} vignettes)"):
                try:
                    run_decision_rater_within(
                        decisions_wide_csv=str(in_within),
                        output_csv=str(out_within),
                        n_runs=int(n_runs),
                        reset_files=True,
                    )
                    # NEW: enforce vignette-then-decision row order
                    _tmp = pd.read_csv(out_within)
                    _tmp = sort_vignette_then_decision(_tmp)
                    _tmp.to_csv(out_within, index=False)

                    st.success(f"Wrote: {out_within}")
                    st.session_state["decisions_rated_within_csv"] = str(out_within)
                except Exception as e:
                    st.error(f"WithinPatient rater failed: {e}")
        else:
            st.info("WithinPatient subset: 0 rows — skipped.")

        # Branch: BetweenPatients
        if len(between_df):
            with st.spinner(f"Rating decisions (BetweenPatients)… ({len(between_df):,} vignettes)"):
                try:
                    run_decision_rater_between(
                        decisions_wide_csv=str(in_between),
                        output_csv=str(out_between),
                        n_runs=int(n_runs),
                        reset_files=True,
                    )
                    # NEW: enforce vignette-then-decision row order
                    _tmp = pd.read_csv(out_between)
                    _tmp = sort_vignette_then_decision(_tmp)
                    _tmp.to_csv(out_between, index=False)

                    st.success(f"Wrote: {out_between}")
                    st.session_state["decisions_rated_between_csv"] = str(out_between)
                except Exception as e:
                    st.error(f"BetweenPatients rater failed: {e}")
        else:
            st.info("BetweenPatients subset: 0 rows — skipped.")

        # Merge outputs (attach Vignette Type) — force consistent dtype
        try:
            parts = []

            scope_df = pd.read_csv(
                st.session_state["vignette_scope_csv"],
                usecols=["vignette_id", "scope"],
                dtype={"vignette_id": str},
            )

            if out_within.exists():
                w = pd.read_csv(out_within, dtype={"vignette_id": str})
                w = w.merge(scope_df, on="vignette_id", how="left").rename(columns={"scope": "Vignette Type"})
                parts.append(w)

            if out_between.exists():
                b = pd.read_csv(out_between, dtype={"vignette_id": str})
                b = b.merge(scope_df, on="vignette_id", how="left").rename(columns={"scope": "Vignette Type"})
                parts.append(b)

            if parts:
                merged = pd.concat(parts, ignore_index=True)
                merged = sort_vignette_then_decision(merged)  # NEW
                merged.to_csv(out_all, index=False)
                st.success(f"Merged: {out_all}")
                st.session_state["decisions_rated_csv"] = str(out_all)
        except Exception as e:
            st.warning(f"Could not build merged decisions file: {e}")

# --- Persistent preview for Step 3B (by Vignette Type) ---
if "run_dir" in st.session_state:
    run_dir = Path(st.session_state["run_dir"])
    decisions_dir = run_dir / "decisions"
    out_within = decisions_dir / "decisions_rated_within.csv"
    out_between = decisions_dir / "decisions_rated_between.csv"
    out_all = decisions_dir / "decisions_rated_all.csv"

    tabs = st.tabs(["WithinPatient", "BetweenPatients", "Merged (all)"])
    with tabs[0]:
        if out_within.exists():
            _df = pd.read_csv(out_within)
            _df = sort_vignette_then_decision(_df).head(20)
            st.dataframe(style_likert(_df), use_container_width=True, height=PREVIEW_HEIGHT)
            download_csv_button("Download decisions_rated_within.csv", out_within, key="dl_decisions_within")
        else:
            st.caption("No WithinPatient output yet.")

    with tabs[1]:
        if out_between.exists():
            _df = pd.read_csv(out_between)
            _df = sort_vignette_then_decision(_df).head(20)
            st.dataframe(style_likert(_df), use_container_width=True, height=PREVIEW_HEIGHT)
            download_csv_button("Download decisions_rated_between.csv", out_between, key="dl_decisions_between")
        else:
            st.caption("No BetweenPatients output yet.")

    with tabs[2]:
        if out_all.exists():
            _df = pd.read_csv(out_all)
            _df = sort_vignette_then_decision(_df).head(20)
            st.dataframe(style_likert(_df), use_container_width=True, height=PREVIEW_HEIGHT)
            download_csv_button("Download decisions_rated_all.csv", out_all, key="dl_decisions_all")
        else:
            st.caption("No merged output yet.")

# --- Build and Download Summary Workbook ---
st.divider()
st.header("Summary workbook (Excel)")

if "run_dir" not in st.session_state:
    st.info("Run at least one step to create a run folder, then you can build the summary workbook.")
else:
    run_dir = Path(st.session_state["run_dir"])

    # Build button -> persist path
    if st.button("Build summary.xlsx", type="primary"):
        try:
            out_xlsx = build_summary_workbook(run_dir)
            st.session_state["summary_xlsx"] = str(out_xlsx)
            st.success(f"Built: {out_xlsx}")
        except Exception as e:
            st.error(f"Failed to build summary workbook: {e}")

    # Persistent download (works across reruns)
    sx = st.session_state.get("summary_xlsx")
    if not sx and (run_dir / "summary.xlsx").exists():
        sx = str(run_dir / "summary.xlsx")
        st.session_state["summary_xlsx"] = sx

    if sx and Path(sx).exists():
        st.download_button(
            "⬇️ Download summary.xlsx",
            data=Path(sx).read_bytes(),
            file_name=Path(sx).name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_summary_xlsx",
        )
        st.caption(f"Size: {Path(sx).stat().st_size:,} bytes")
    else:
        st.caption("No summary.xlsx yet. Click **Build summary.xlsx**.")

st.divider()
with st.expander("Run Summary", expanded=True):
    # input vignettes
    n_vignettes = len(st.session_state.get("input_df", pd.DataFrame()))
    # factors extracted
    n_factors = _safe_len_csv(st.session_state.get("factors_csv", ""))
    # decisions (wide): count decision columns present
    dec_path = st.session_state.get("decisions_csv", "")
    if dec_path:
        try:
            peek = pd.read_csv(dec_path, nrows=1)
            n_decision_cols = sum(str(c).lower().startswith("decision_") for c in peek.columns)
        except Exception:
            n_decision_cols = 0
    else:
        n_decision_cols = 0
    # rated decisions rows
    n_rated = _safe_len_csv(st.session_state.get("decisions_rated_csv", ""))

    cols = st.columns(4)
    cols[0].metric("Vignettes loaded", f"{n_vignettes:,}")
    cols[1].metric("Factors extracted (rows)", f"{n_factors:,}")
    cols[2].metric("Decision columns per vignette", f"{n_decision_cols:,}")
    cols[3].metric("Rated decisions (rows)", f"{n_rated:,}")

# ======================
# Tools — Consistency Checks (PerRun Likert only)
# ======================
st.divider()
st.header("Tools")
st.header("Consistency Checks")

if "decisions_csv" not in st.session_state:
    st.info("Run Step 3A (Extract Decisions) first to enable consistency checks.")
elif "vignette_scope_csv" not in st.session_state:
    st.info("Run Step 1 (Vignette Type) first so we can split inputs by Within/Between.")
else:
    run_dir = Path(st.session_state["run_dir"])
    decisions_dir = run_dir / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)

    decisions_csv = st.session_state["decisions_csv"]
    scope_csv = st.session_state["vignette_scope_csv"]

    # Controls
    cols = st.columns([1, 2])
    with cols[0]:
        replicates = st.number_input(
            "Replicates",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="How many separate full reruns of the rater to include in the PerRun sheet."
        )
    with cols[1]:
        st.caption("This builds an XLSX with a single color-coded PerRun_Likert sheet.")

    # Make/refresh the split inputs (same logic as Step 3B)
    with st.spinner("Preparing Within/Between inputs…"):
        scope_df = pd.read_csv(scope_csv, usecols=["vignette_id", "scope"])
        scope_df["vignette_id"] = scope_df["vignette_id"].astype(str)

        dec_df = pd.read_csv(decisions_csv)
        dec_df["vignette_id"] = dec_df["vignette_id"].astype(str)

        joined = dec_df.merge(scope_df, on="vignette_id", how="inner")
        within_df = joined[joined["scope"] == "WithinPatient"].copy()
        between_df = joined[joined["scope"] == "BetweenPatients"].copy()

        in_within = decisions_dir / "decisions_within_input.csv"
        in_between = decisions_dir / "decisions_between_input.csv"
        if len(within_df):
            within_df.to_csv(in_within, index=False)
        if len(between_df):
            between_df.to_csv(in_between, index=False)

    # Buttons row
    bcols = st.columns(2)
    with bcols[0]:
        run_within_btn = st.button(
            "Run consistency — WithinPatient rater",
            type="primary",
            disabled=(status != "success" or len(within_df) == 0),
            help="Runs the WithinPatient rater multiple times and builds a PerRun_Likert workbook."
        )
    with bcols[1]:
        run_between_btn = st.button(
            "Run consistency — BetweenPatients rater",
            type="primary",
            disabled=(status != "success" or len(between_df) == 0),
            help="Runs the BetweenPatients rater multiple times and builds a PerRun_Likert workbook."
        )

    # Actions
    if run_within_btn:
        try:
            from tools_consistency_runner import run_replicates_and_build
            from decisions_rater_withinpatient import run_decision_rater as rater_within

            with st.spinner(f"Running {int(replicates)} replicates (WithinPatient)…"):
                xlsx_path, run_csvs = run_replicates_and_build(
                    input_csv=str(in_within),
                    rater_func=rater_within,
                    replicates=int(replicates),
                    out_dir=str(decisions_dir),
                    run_prefix="within",
                    n_runs_per_call=1,
                    reset_files=True,   # always fresh for replicates
                )
            st.success(f"PerRun_Likert workbook (WithinPatient) ready: {xlsx_path}")
            st.session_state["consistency_within_xlsx"] = xlsx_path
        except Exception as e:
            st.error(f"WithinPatient consistency run failed: {e}")

    if run_between_btn:
        try:
            from tools_consistency_runner import run_replicates_and_build   
            from decisions_rater_betweenpatients import run_decision_rater as rater_between

            with st.spinner(f"Running {int(replicates)} replicates (BetweenPatients)…"):
                xlsx_path, run_csvs = run_replicates_and_build(
                    input_csv=str(in_between),
                    rater_func=rater_between,
                    replicates=int(replicates),
                    out_dir=str(decisions_dir),
                    run_prefix="between",
                    n_runs_per_call=1,
                    reset_files=True,   # always fresh for replicates
                )
            st.success(f"PerRun_Likert workbook (BetweenPatients) ready: {xlsx_path}")
            st.session_state["consistency_between_xlsx"] = xlsx_path
        except Exception as e:
            st.error(f"BetweenPatients consistency run failed: {e}")

    # Download buttons (persist across reruns)
    dlcols = st.columns(2)
    with dlcols[0]:
        xp = st.session_state.get("consistency_within_xlsx")
        if xp and Path(xp).exists():
            st.download_button(
                "Download PerRun_Likert (WithinPatient)",
                data=Path(xp).read_bytes(),
                file_name=Path(xp).name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_consistency_within",
            )
    with dlcols[1]:
        xp = st.session_state.get("consistency_between_xlsx")
        if xp and Path(xp).exists():
            st.download_button(
                "Download PerRun_Likert (BetweenPatients)",
                data=Path(xp).read_bytes(),
                file_name=Path(xp).name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_consistency_between",
            )