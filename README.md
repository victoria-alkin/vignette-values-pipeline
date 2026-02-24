# Vignette Values Pipeline (Human Values Project)

This repository contains a pipeline for processing clinical ethics vignettes used in the Human Values Project in AI for clinical medicine.

## Getting started

### 1. Get the repository

**Option 1: Clone with Git (recommended)**

```bash
git clone https://github.com/victoria-alkin/vignette-values-pipeline.git
cd vignette-values-pipeline
```

**Option 2: Or download as ZIP (no Git required)**

1. Open the repository page in your browser.
2. Click **Code → Download ZIP**
3. Unzip the folder.
4. Open a terminal inside the unzipped folder (the folder containing `requirements.txt`).

### 2. Create and activate a virtual environment (recommended)

**Windows (PowerShell)**

```bash
python -m venv values_env
.\values_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Mac / Linux**

```bash
python3 -m venv values_env
source values_env/bin/activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

From the repository root (the folder that contains `requirements.txt`):

```bash
pip install -r requirements.txt
```

### 4. Set necessary environment variables
On startup, the app checks for:

- `OPENAI_MODEL` (required)

And **either**:

- `OPENAI_API_KEY`

**OR**

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`

The included `gabriel_compatibility.py` file automatically maps Azure credentials into OpenAI-compatible environment variables without overwriting existing values.


### Using Azure OpenAI (PowerShell example)

```bash
$env:AZURE_OPENAI_API_KEY     = "<your key>"
$env:AZURE_OPENAI_ENDPOINT    = "https://your-resource.openai.azure.com"
$env:AZURE_OPENAI_API_VERSION = "2025-03-01-preview"

$env:OPENAI_MODEL = "gpt-4.1-mini"
```

Do **not** manually append `/openai` to the endpoint.  
The compatibility layer handles routing.


### Using OpenAI directly (PowerShell example)

```bash
$env:OPENAI_API_KEY = "<your key>"
$env:OPENAI_MODEL   = "gpt-4.1-mini"
```

### 5. Prepare input file

The Streamlit app expects a table with at least these columns:

- `vignette_id`
- `vignette_text`

Supported upload formats in the app:

- CSV  
- XLSX/XLS  
- JSONL/NDJSON

### 6. Running the app

From the repository root (the folder containing `app_gui_interactive.py`):

```bash
streamlit run app_gui_interactive.py
```
After running the command, your browser should automatically open to http://localhost:8501.
---

## Components

Core modules (Python):

- `app_gui_interactive.py` — Streamlit app that runs the pipeline end-to-end and writes outputs to a run directory.
- `vignette_type_classifier.py` — Step 1: classifies vignette type/scope (WithinPatient vs BetweenPatients).
- `contextual_factor_extraction.py` — Step 2A: extracts contextual factors from each vignette.
- `classifier_ethics_withinpatient.py` / `classifier_ethics_betweenpatients.py` — Step 2B: classifies extracted factors for each vignette type.
- `decision_extraction.py` — Step 3A: extracts decision options from each vignette.
- `decisions_rater_withinpatient.py` / `decisions_rater_betweenpatients.py` — Step 3B: rates decisions by the extent to which they promote/neutral/counteract ethical principles for each vignette type.
- `tools_consistency_runner.py` — Tool for running replicate rating to check consistency and exporting a `PerRun_Likert` workbook.
- `gabriel_compatibility.py` — Compatibility layer allowing either Azure OpenAI or OpenAI credentials without modifying upstream GABRIEL.

Example input:
- `examples/sample_vignettes.csv`

---

## Troubleshooting

- Ensure your virtual environment is activated.
- Ensure you ran:

```bash
pip install -r requirements.txt
```

- Confirm environment variables are set in the same terminal session used to launch Streamlit.
- For Azure errors, verify:
  - Correct endpoint URL
  - Correct API version
  - `OPENAI_MODEL` matches your Azure deployment name
