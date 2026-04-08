---
title: Meta Hack Clinical Trial Triage
emoji: "🧪"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Clinical Trial Triage (OpenEnv)

Production-style OpenEnv environment for clinical trial operations and baseline evaluation.

This project simulates real workflows in pharmacovigilance and trial quality operations with typed APIs, deterministic graders, robust baseline inference, and Hugging Face Space deployment support.

## Live Deployment

- Hugging Face Space repo: https://huggingface.co/spaces/vvinayakkkkk/meta-hack
- Space app URL: https://vvinayakkkkk-meta-hack.hf.space

If the app URL is sleeping or rebuilding, wait for the Space to become healthy and refresh.

## What This Project Covers

- Adverse event triage
- Protocol deviation audit
- Safety narrative generation (ICSR-style)
- Dense reward grading in 0.0 to 1.0
- OpenEnv-compatible endpoints and metadata
- Submission-safe baseline script at project root: `inference.py`

## Core Features

- FastAPI server with typed request/response models
- OpenEnv adapter mounted at `/openenv/*`
- Session isolation support via `X-Session-ID`
- Deterministic heuristic fallback when LLM/provider fails
- Dockerized deployment for Hugging Face Spaces
- Local pre-submission validator script

## Repository Layout

```text
clinical-trial-triage/
  client.py
  inference.py
  models.py
  openenv.yaml
  Dockerfile
  requirements.txt
  server/
    app.py
    environment.py
    openenv_env.py
  tasks/
    case_bank.py
    production_cases.py
    graders.py
  scripts/
    validate_submission.py
    test_generalization.py
    baseline_inference.py
  ui/
    index.html
    triage.html
    docs.html
    performance.html
```

## Tasks and Graders

### 1) `adverse_event_triage` (easy)

- Severity classification
- Reporting timeline
- MedDRA-like coding fields
- Seriousness determination

### 2) `protocol_deviation_audit` (medium)

- Deviation classification
- CAPA requirement
- Site risk scoring
- Finding escalation IDs

### 3) `safety_narrative_generation` (hard)

- Structured narrative quality
- Causality assessment
- Temporal evidence flags
- Regulatory completeness

All graders output bounded rewards and normalized scoring in `[0.0, 1.0]`.

## Requirements

- Python 3.10+
- Docker (for HF-equivalent local run)
- Optional: Hugging Face token for router inference
- Optional: Groq key(s) for alternate baseline script

## Local Setup

### 1) Clone and enter project

```bash
git clone <your-repo-url>
cd clinical-trial-triage
```

### 2) Create virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Configure environment

Create `.env` at project root (do not commit this file):

```env
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
HF_TOKEN=hf_xxx

# Optional
ENV_SERVER_URL=http://localhost:8000
API_KEY=
GROQ_API_KEY=
GROQ_API_KEYS=
BASELINE_MODEL=llama-3.3-70b-versatile
```

## Run Locally

### Start API server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Quick checks

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/tasks
```

### UI routes

- Browser root `/` redirects to `/ui/`
- Main triage view: `/ui/triage.html`
- API web card view: `/web`

## API Endpoints

### Core

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /grader`
- `GET /health`
- `POST /baseline`
- `POST /infer/step`

### OpenEnv

- `GET /openenv/metadata`
- `GET /openenv/schema`
- `POST /openenv/reset`
- `POST /openenv/step`
- `GET /openenv/state`
- `GET /openenv/health`

## Baseline Inference

Submission baseline script is root `inference.py`.

What it does:

- Uses OpenAI SDK against HF router variables
- Tries LLM response with bounded retries
- Normalizes actions to strict task schemas
- Uses deterministic fallback if provider call fails
- Writes `outputs/baseline_results.json`
- Emits structured stdout markers: `[START]`, `[STEP]`, `[END]`

Run:

```bash
python inference.py
```

Expected output artifact:

- `outputs/baseline_results.json`

Expected score keys:

- `mean_score`
- `overall_mean_reward`
- `tasks` object with per-task metrics

## Validation and Testing

### Pre-submission validator

Start server first, then run:

```bash
python scripts/validate_submission.py
```

Validator checks:

- API health and task endpoints
- OpenEnv metadata/schema/reset/step/state/health
- 3+ tasks with grader outputs in range `[0.0, 1.0]`
- Root `inference.py` runs and outputs valid schema under runtime budget

### Generalization run

```bash
python scripts/test_generalization.py
```

Output:

- `outputs/generalization_results.json`

## Docker and HF-Equivalent Local Run

Build:

```bash
docker build -t clinical-trial-triage .
```

Run:

```bash
docker run --rm -p 7860:7860 clinical-trial-triage
```

Smoke test:

```bash
curl http://127.0.0.1:7860/health
curl -X POST http://127.0.0.1:7860/reset -H "Content-Type: application/json" -d '{"task_id":"adverse_event_triage"}'
```

## Hugging Face Spaces: What To Do Now

Follow this exact sequence.

### 1) Space configuration

- Space SDK: `Docker`
- Port: `7860` (already aligned in Dockerfile)
- Hardware: CPU basic works, CPU upgraded is safer for faster builds/inference

### 2) Push code to Space repo

- Remote target: `https://huggingface.co/spaces/vvinayakkkkk/meta-hack`
- Required files must exist in root of submitted project:
  - `Dockerfile`
  - `openenv.yaml`
  - `inference.py`
  - `server/`, `tasks/`, `models.py`, `requirements.txt`

### 3) Set Space Variables/Secrets

In Space settings, add these:

Required:

- `API_BASE_URL` = `https://router.huggingface.co/v1`
- `MODEL_NAME` = `meta-llama/Llama-3.3-70B-Instruct`
- `HF_TOKEN` = your Hugging Face token

Optional:

- `ENV_SERVER_URL` (usually not needed in Space)
- `GROQ_API_KEY` or `GROQ_API_KEYS` (only if using Groq script)
- `BASELINE_MODEL` (only for Groq baseline script)

Security:

- Put tokens in Secrets, not plain Variables.
- Never commit `.env`.

### 4) Rebuild and verify

After push + env setup:

- Watch build logs until healthy
- Confirm these URLs return `200`:
  - `https://vvinayakkkkk-meta-hack.hf.space/health`
  - `https://vvinayakkkkk-meta-hack.hf.space/openenv/metadata`
  - `https://vvinayakkkkk-meta-hack.hf.space/openenv/schema`
- Browser UI should load at:
  - `https://vvinayakkkkk-meta-hack.hf.space/`
  - `https://vvinayakkkkk-meta-hack.hf.space/ui/`

### 5) Submission readiness checks

Before final submission:

- Run local validator: `python scripts/validate_submission.py`
- Ensure runtime is under 20 minutes for `inference.py`
- Ensure output scores are bounded in `[0.0, 1.0]`
- Ensure root script name is exactly `inference.py`

## Troubleshooting

### UI not visible on Space

- Check Space is healthy and not sleeping
- Confirm root now redirects browsers to `/ui/`
- Open `/ui/` directly if needed
- Confirm build includes `ui/` directory

### HF router 402 or credit errors

- This usually means HF Inference Provider credit exhaustion
- Script falls back to deterministic logic and still completes
- For better LLM-assisted scores, use a funded token/provider

### Build failures

- Verify `requirements.txt` resolves cleanly
- Confirm Dockerfile starts `uvicorn server.app:app --port ${PORT:-7860}`
- Confirm no secrets are hardcoded in repository files

### Validator failures

- Start server before running validator
- Verify `/openenv/*` endpoints respond
- Verify `outputs/baseline_results.json` gets written

## Security and Compliance Notes

- Do not store real patient-identifying information
- Do not commit API keys, tokens, or `.env`
- Treat this as synthetic benchmarking data and workflow simulation

## Contribution Workflow

- Create a feature branch
- Keep changes scoped and testable
- Run validator before opening PR
- Include notes on reward/behavior impact for grading changes

## License

MIT

## Acknowledgements

- OpenEnv ecosystem
- FastAPI and Pydantic
- Hugging Face Spaces
