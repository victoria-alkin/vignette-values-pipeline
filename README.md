# Vignette Values Pipeline (Human Values Project)

This repository contains a pipeline for processing clinical ethics vignettes used in the Human Values Project in AI for clinical medicine.

## Components

Core modules (Python):
- `app_gui_interactive.py` — Streamlit app that runs the pipeline end-to-end and writes outputs to a run directory.
- `vignette_type_classifier.py` — Step 1: classifies vignette type/scope (WithinPatient vs BetweenPatients).
- `contextual_factor_extraction.py` — Step 2A: extracts contextual factors from each vignette.
- `classifier_ethics_withinpatient.py` / `classifier_ethics_betweenpatients.py` — Step 2B: classifies extracted factors for each vignette type.
- `decision_extraction.py` — Step 3A: extracts decision options from each vignette.
- `decisions_rater_withinpatient.py` / `decisions_rater_betweenpatients.py` — Step 3B: rates decisions by the extent to which they promote/neutral/counteract ethical principles for each vignette type.
- `tools_consistency_runner.py` — tool for running replicate rating to check consistency and exporting a PerRun_Likert workbook.

Example input:
- `examples/sample_vignettes.csv`

## Input format

The Streamlit app expects a table with at least these columns:
- `vignette_id`
- `vignette_text`

Supported upload formats in the app: CSV, XLSX/XLS, JSONL/NDJSON.

## Environment variables

On startup, the app checks for:
- `OPENAI_MODEL` (required)
- Either:
  - `OPENAI_API_KEY`, or
  - `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`

## Running the app

From the repository root:

```bash
streamlit run app_gui_interactive.py
