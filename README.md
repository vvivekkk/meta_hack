<div align="center">

<img src="https://img.shields.io/badge/version-1.0.0-blue?style=for-the-badge" alt="Version"/>
<img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License"/>
<img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
<img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
<img src="https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace"/>

<br/><br/>

<h1>🧪 Clinical Trial Triage</h1>

<p><strong>Production-grade OpenEnv environment for clinical trial operations, pharmacovigilance, and baseline evaluation.</strong></p>

<p>
  <a href="https://vvinayakkkkk-meta-hack.hf.space"><strong>🚀 Live Demo</strong></a> ·
  <a href="https://vvinayakkkkk-meta-hack.hf.space/docs"><strong>📖 API Docs</strong></a> ·
  <a href="#-quick-start"><strong>⚡ Quick Start</strong></a> ·
  <a href="#-tasks--graders"><strong>📋 Tasks</strong></a> ·
  <a href="#-contributing"><strong>🤝 Contribute</strong></a>
</p>

<br/>

> A simulation platform for real-world pharmacovigilance and trial quality workflows — with typed APIs, deterministic graders, dense reward scoring, and Hugging Face Space deployment support.

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Live Deployment](#-live-deployment)
- [Features](#-features)
- [Architecture](#-architecture)
- [Repository Structure](#-repository-structure)
- [Tasks & Graders](#-tasks--graders)
- [Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
  - [Running the Server](#running-the-server)
- [API Reference](#-api-reference)
  - [Core Endpoints](#core-endpoints)
  - [OpenEnv Endpoints](#openenv-endpoints)
- [Baseline Inference](#-baseline-inference)
- [Validation & Testing](#-validation--testing)
- [Docker](#-docker)
- [Hugging Face Spaces Deployment](#-hugging-face-spaces-deployment)
- [Troubleshooting](#-troubleshooting)
- [Security & Compliance](#-security--compliance)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🔭 Overview

**Clinical Trial Triage** is a production-style [OpenEnv](https://openenv.dev) environment designed for benchmarking and evaluating AI agents on real-world clinical operations tasks. It simulates end-to-end workflows found in regulated pharmacovigilance and trial quality settings, including:

- **Adverse Event Triage** — classifying severity, seriousness, and reporting timelines
- **Protocol Deviation Auditing** — risk scoring, CAPA flagging, escalation tracking
- **Safety Narrative Generation** — structured ICSR-style narrative drafting with causality assessment

All tasks ship with deterministic graders that emit bounded reward scores in `[0.0, 1.0]`, making this suitable for offline benchmarking, reinforcement learning environments, and hackathon evaluation pipelines.

---

## 🌐 Live Deployment

| Resource | URL |
|---|---|
| 🤗 HF Space Repo | [huggingface.co/spaces/vvinayakkkkk/meta-hack](https://huggingface.co/spaces/vvinayakkkkk/meta-hack) |
| 🚀 Live App | [vvinayakkkkk-meta-hack.hf.space](https://vvinayakkkkk-meta-hack.hf.space) |
| 📋 Triage UI | [/ui/triage.html](https://vvinayakkkkk-meta-hack.hf.space/ui/triage.html) |
| 📖 API Docs | [/docs](https://vvinayakkkkk-meta-hack.hf.space/docs) |
| 🏥 Health Check | [/health](https://vvinayakkkkk-meta-hack.hf.space/health) |

> **Note:** If the Space is sleeping, navigate to the app URL and wait for it to become healthy before refreshing.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧩 **OpenEnv-Compatible** | Full adapter mounted at `/openenv/*` with metadata, schema, and session endpoints |
| 🔒 **Session Isolation** | Per-request session context via `X-Session-ID` header |
| 🎯 **Deterministic Graders** | Heuristic-based fallback graders that work even without an LLM provider |
| 🤖 **LLM-Powered Baseline** | Pluggable inference via HF Router, Groq, or any OpenAI-compatible endpoint |
| 📦 **Dockerized** | One-command build and run for local or cloud deployment |
| ✅ **Validator Script** | Pre-submission validation that checks all endpoints, schemas, and scoring bounds |
| 🖥️ **Browser UI** | Triage, docs, and performance views served at `/ui/` |
| 📊 **Dense Reward Scoring** | All graders output normalized `[0.0, 1.0]` rewards for RL compatibility |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                        │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │  Core API   │   │  OpenEnv     │   │   Static UI          │ │
│  │  /reset     │   │  Adapter     │   │   /ui/triage.html    │ │
│  │  /step      │   │  /openenv/*  │   │   /ui/docs.html      │ │
│  │  /state     │   │              │   │   /ui/performance    │ │
│  │  /tasks     │   │              │   │                      │ │
│  └──────┬──────┘   └──────┬───────┘   └──────────────────────┘ │
│         │                 │                                       │
│  ┌──────▼─────────────────▼─────────────────────┐               │
│  │              Environment Layer                │               │
│  │   environment.py  ·  openenv_env.py           │               │
│  └──────────────────────┬────────────────────────┘               │
│                         │                                         │
│  ┌──────────────────────▼────────────────────────┐               │
│  │              Tasks & Graders                   │               │
│  │  adverse_event_triage  ·  protocol_deviation   │               │
│  │  safety_narrative_generation                   │               │
│  └────────────────────────────────────────────────┘               │
└────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  LLM Provider      │
                    │  HF Router / Groq  │
                    │  (+ heuristic      │
                    │   fallback)        │
                    └────────────────────┘
```

---

## 📁 Repository Structure

```text
clinical-trial-triage/
│
├── 📄 inference.py              # Submission baseline script (root-level, required)
├── 📄 client.py                 # Python client for interacting with the server
├── 📄 models.py                 # Pydantic request/response models
├── 📄 openenv.yaml              # OpenEnv environment declaration
├── 📄 Dockerfile                # Container definition for HF Spaces / local Docker
├── 📄 requirements.txt          # Python dependencies
│
├── 📂 server/
│   ├── app.py                   # FastAPI application entrypoint
│   ├── environment.py           # Core environment logic and state management
│   └── openenv_env.py           # OpenEnv adapter implementation
│
├── 📂 tasks/
│   ├── case_bank.py             # Synthetic case data for all tasks
│   ├── production_cases.py      # Extended production-quality case scenarios
│   └── graders.py               # Deterministic reward graders (returns [0.0, 1.0])
│
├── 📂 scripts/
│   ├── validate_submission.py   # Pre-submission end-to-end validator
│   ├── test_generalization.py   # Generalization test suite
│   └── baseline_inference.py   # Alternate Groq-based baseline script
│
├── 📂 ui/
│   ├── index.html               # Root redirect to /ui/
│   ├── triage.html              # Interactive triage task UI
│   ├── docs.html                # API documentation view
│   └── performance.html         # Score and performance dashboard
│
└── 📂 outputs/                  # Generated artifacts (gitignored)
    ├── baseline_results.json
    └── generalization_results.json
```

---

## 📋 Tasks & Graders

All tasks follow the same request/response contract. Each grader returns a bounded reward and normalized scoring in `[0.0, 1.0]`.

---

### 1. `adverse_event_triage` · **Easy**

Evaluate an AI agent's ability to perform front-line adverse event (AE) classification as in a pharmacovigilance unit.

**Graded Fields:**

| Field | Description |
|---|---|
| `severity` | Classification: mild / moderate / severe / life-threatening |
| `reporting_timeline` | Expedited (7/15 days) vs. routine (30 days) |
| `meddra_code` | MedDRA-like PT/SOC coding accuracy |
| `seriousness` | Boolean seriousness determination per ICH E2A criteria |

**Reward Breakdown:**

```
severity_score       × 0.30
timeline_score       × 0.25
meddra_score         × 0.25
seriousness_score    × 0.20
─────────────────────────────
total_reward ∈ [0.0, 1.0]
```

---

### 2. `protocol_deviation_audit` · **Medium**

Simulate a site audit scenario where deviations from the clinical protocol must be identified, classified, and escalated appropriately.

**Graded Fields:**

| Field | Description |
|---|---|
| `deviation_class` | Category: minor / major / critical |
| `capa_required` | Whether a Corrective and Preventive Action is mandated |
| `site_risk_score` | Normalized risk score for the investigative site |
| `finding_escalation_ids` | Correct escalation IDs for flagged findings |

**Reward Breakdown:**

```
deviation_class_score      × 0.30
capa_score                 × 0.25
site_risk_score            × 0.25
escalation_ids_score       × 0.20
─────────────────────────────────
total_reward ∈ [0.0, 1.0]
```

---

### 3. `safety_narrative_generation` · **Hard**

Generate a structured Individual Case Safety Report (ICSR)-style narrative meeting regulatory completeness standards (EMA, FDA).

**Graded Fields:**

| Field | Description |
|---|---|
| `narrative_quality` | Structural completeness and clinical coherence |
| `causality_assessment` | WHO-UMC or similar causality classification |
| `temporal_evidence_flags` | Correct identification of temporal relationships |
| `regulatory_completeness` | Coverage of mandatory ICH E2B(R3) fields |

**Reward Breakdown:**

```
narrative_quality_score       × 0.30
causality_score               × 0.25
temporal_evidence_score       × 0.25
regulatory_completeness_score × 0.20
─────────────────────────────────────
total_reward ∈ [0.0, 1.0]
```

---

## ⚡ Quick Start

### Prerequisites

| Dependency | Version | Required |
|---|---|---|
| Python | 3.10+ | ✅ Yes |
| Docker | 20.10+ | Optional (for container run) |
| HF Token | — | Optional (for LLM inference) |
| Groq API Key | — | Optional (alternate baseline) |

---

### Installation

**1. Clone the repository**

```bash
git clone <your-repo-url>
cd clinical-trial-triage
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
```

<details>
<summary>Linux / macOS</summary>

```bash
source .venv/bin/activate
```
</details>

<details>
<summary>Windows (PowerShell)</summary>

```powershell
.\.venv\Scripts\Activate.ps1
```
</details>

**3. Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Environment Configuration

Create a `.env` file at the project root. **Do not commit this file.**

```env
# ─── Required ──────────────────────────────────────────────────
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# ─── Optional ──────────────────────────────────────────────────
ENV_SERVER_URL=http://localhost:8000
API_KEY=

# Groq-based alternate baseline
GROQ_API_KEY=
GROQ_API_KEYS=
BASELINE_MODEL=llama-3.3-70b-versatile
```

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | ✅ | Base URL for the LLM inference router |
| `MODEL_NAME` | ✅ | Model identifier for completions |
| `HF_TOKEN` | ✅ | Hugging Face token for HF Router access |
| `ENV_SERVER_URL` | ❌ | Override for local server URL |
| `GROQ_API_KEY` | ❌ | Groq API key for alternate baseline |
| `GROQ_API_KEYS` | ❌ | Comma-separated list of Groq keys (rotation) |
| `BASELINE_MODEL` | ❌ | Model name for Groq baseline script |

---

### Running the Server

**Start the API server:**

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

**Verify it's running:**

```bash
# Health check
curl http://127.0.0.1:8000/health

# List available tasks
curl http://127.0.0.1:8000/tasks
```

**Access the browser UI:**

| View | URL |
|---|---|
| Main UI | [http://localhost:8000/ui/](http://localhost:8000/ui/) |
| Triage | [http://localhost:8000/ui/triage.html](http://localhost:8000/ui/triage.html) |
| API Card | [http://localhost:8000/web](http://localhost:8000/web) |
| Swagger Docs | [http://localhost:8000/docs](http://localhost:8000/docs) |

---

## 📡 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset` | Reset environment state for a task |
| `POST` | `/step` | Submit an action and receive a reward |
| `GET` | `/state` | Get current environment state |
| `GET` | `/tasks` | List all available tasks |
| `GET` | `/grader` | Retrieve grader metadata |
| `GET` | `/health` | Liveness check |
| `POST` | `/baseline` | Run built-in baseline action |
| `POST` | `/infer/step` | Run LLM inference for a step |

**Example: Reset and step through `adverse_event_triage`**

```bash
# 1. Reset the environment
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "adverse_event_triage"}'

# 2. Submit an action
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "severity": "severe",
      "seriousness": true,
      "reporting_timeline": "15_days",
      "meddra_code": "10019211"
    }
  }'
```

**Example response from `/step`:**

```json
{
  "observation": { "case_id": "AE-2024-001", "status": "evaluated" },
  "reward": 0.87,
  "done": true,
  "info": {
    "severity_score": 1.0,
    "timeline_score": 1.0,
    "meddra_score": 0.75,
    "seriousness_score": 1.0
  }
}
```

---

### OpenEnv Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/openenv/metadata` | Environment metadata (name, version, tasks) |
| `GET` | `/openenv/schema` | Action/observation JSON schemas |
| `POST` | `/openenv/reset` | OpenEnv-compatible reset |
| `POST` | `/openenv/step` | OpenEnv-compatible step |
| `GET` | `/openenv/state` | OpenEnv-compatible state retrieval |
| `GET` | `/openenv/health` | OpenEnv health check |

> All OpenEnv endpoints follow the [OpenEnv specification](https://openenv.dev/spec). Session isolation is supported via the `X-Session-ID` request header.

---

## 🤖 Baseline Inference

The root-level `inference.py` is the **submission baseline script**. It is designed to be run as a standalone evaluation against the live environment.

**What it does:**

1. Connects to the environment server via `ENV_SERVER_URL`
2. Iterates over all configured tasks
3. Constructs prompts and calls the LLM via OpenAI-compatible SDK (HF Router)
4. Normalizes model outputs to match strict task action schemas
5. Falls back to deterministic heuristics if the LLM call fails or times out
6. Writes structured results to `outputs/baseline_results.json`
7. Emits structured stdout markers for pipeline consumption: `[START]`, `[STEP]`, `[END]`

**Run:**

```bash
python inference.py
```

**Expected output artifact** — `outputs/baseline_results.json`:

```json
{
  "mean_score": 0.74,
  "overall_mean_reward": 0.74,
  "tasks": {
    "adverse_event_triage": {
      "mean_reward": 0.81,
      "episodes": 5,
      "scores": [0.87, 0.75, 0.80, 0.85, 0.78]
    },
    "protocol_deviation_audit": { ... },
    "safety_narrative_generation": { ... }
  }
}
```

> **Runtime budget:** `inference.py` must complete in under **20 minutes** for valid submission.

---

## ✅ Validation & Testing

### Pre-Submission Validator

Start the server first, then run:

```bash
python scripts/validate_submission.py
```

**What the validator checks:**

- [ ] `/health` endpoint responds with `200 OK`
- [ ] `/tasks` lists at least 3 tasks
- [ ] All OpenEnv endpoints respond correctly (`/openenv/metadata`, `/openenv/schema`, `/openenv/reset`, `/openenv/step`, `/openenv/state`, `/openenv/health`)
- [ ] Grader outputs are bounded in `[0.0, 1.0]` for all tasks
- [ ] Root `inference.py` runs and produces valid output schema within runtime budget

---

### Generalization Test Suite

```bash
python scripts/test_generalization.py
```

Runs inference across a held-out set of cases to measure generalization outside the training distribution.

**Output:** `outputs/generalization_results.json`

---

## 🐳 Docker

### Build

```bash
docker build -t clinical-trial-triage .
```

### Run

```bash
docker run --rm -p 7860:7860 \
  -e HF_TOKEN=hf_xxx \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
  clinical-trial-triage
```

### Smoke Test

```bash
# Health
curl http://127.0.0.1:7860/health

# Reset a task
curl -X POST http://127.0.0.1:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "adverse_event_triage"}'
```

---

## 🤗 Hugging Face Spaces Deployment

Follow this exact sequence for a clean deployment.

### Step 1 — Configure Your Space

- **SDK:** `Docker`
- **Port:** `7860` (already aligned in `Dockerfile`)
- **Hardware:** CPU Basic works; CPU Upgraded recommended for faster builds and inference

### Step 2 — Push Code

Target remote: `https://huggingface.co/spaces/vvinayakkkkk/meta-hack`

Required files in the root of your pushed project:

```
✅ Dockerfile
✅ openenv.yaml
✅ inference.py
✅ server/
✅ tasks/
✅ models.py
✅ requirements.txt
```

### Step 3 — Set Space Variables & Secrets

In your Space **Settings → Variables and Secrets**:

| Key | Type | Value |
|---|---|---|
| `API_BASE_URL` | Variable | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Variable | `meta-llama/Llama-3.3-70B-Instruct` |
| `HF_TOKEN` | **Secret** | Your HF token |
| `GROQ_API_KEY` | **Secret** | *(Optional)* Groq key |
| `BASELINE_MODEL` | Variable | *(Optional)* `llama-3.3-70b-versatile` |

> ⚠️ Always store tokens in **Secrets**, never in plain Variables. Never commit `.env` to the repository.

### Step 4 — Rebuild and Verify

After pushing and setting secrets:

1. Watch the build logs until status is **Healthy**
2. Confirm the following return `200`:

```
https://vvinayakkkkk-meta-hack.hf.space/health
https://vvinayakkkkk-meta-hack.hf.space/openenv/metadata
https://vvinayakkkkk-meta-hack.hf.space/openenv/schema
```

3. Browser UI should be accessible at `/` and `/ui/`

### Step 5 — Final Submission Checklist

```
[ ] python scripts/validate_submission.py passes all checks
[ ] inference.py runtime < 20 minutes
[ ] All reward scores bounded in [0.0, 1.0]
[ ] Root script is named exactly inference.py
[ ] No secrets committed to repository
[ ] outputs/baseline_results.json contains mean_score, overall_mean_reward, tasks
```

---

## 🔧 Troubleshooting

<details>
<summary><strong>UI not visible on the Space</strong></summary>

- Verify the Space is **Healthy** (not sleeping or building)
- Navigate to `/ui/` directly: `https://your-space.hf.space/ui/`
- Confirm the `ui/` directory was included in your push
- Check that the Dockerfile copies `ui/` to the working directory

</details>

<details>
<summary><strong>HF Router 402 or credit errors</strong></summary>

- This indicates Hugging Face Inference Provider credit exhaustion on your account
- The script **automatically falls back** to deterministic heuristics and will still complete
- For higher LLM-assisted scores, top up HF credits or switch to a Groq key via `GROQ_API_KEY`

</details>

<details>
<summary><strong>Docker build failures</strong></summary>

- Verify `requirements.txt` resolves cleanly: `pip install -r requirements.txt --dry-run`
- Confirm the Dockerfile `CMD` starts uvicorn on `${PORT:-7860}`
- Ensure no secrets are hardcoded in any repository file
- Check Python version compatibility (3.10+ required)

</details>

<details>
<summary><strong>Validator script failures</strong></summary>

- **Start the server before running the validator** — it makes live HTTP requests
- Confirm all `/openenv/*` endpoints respond (check server logs for 404s)
- Confirm `outputs/baseline_results.json` is written after `inference.py` runs
- Check for import errors in `server/app.py` on startup

</details>

<details>
<summary><strong>LLM responses not matching task schema</strong></summary>

- `inference.py` normalizes LLM outputs before submission — check normalization logic for your task
- Inspect raw model output by temporarily adding print statements before normalization
- If the provider is rate-limiting, increase retry delay in `inference.py`

</details>

---

## 🔐 Security & Compliance

| Requirement | Status |
|---|---|
| No real patient-identifying information (PII) stored | ✅ All data is synthetic |
| API keys / tokens not committed to repository | ✅ Use `.env` and HF Secrets |
| `.env` excluded from version control | ✅ Add to `.gitignore` |
| Synthetic benchmarking data only | ✅ No real clinical records used |
| HIPAA / GDPR scope | ⚠️ Not applicable — simulation only |

> This project is a **simulation platform only**. It does not process, store, or transmit real patient data. Treat all case data as synthetic benchmarking material.

---

## 🤝 Contributing

Contributions are welcome. Please follow this workflow:

**1. Fork and branch**

```bash
git checkout -b feature/your-feature-name
```

**2. Make your changes**

Keep changes scoped, testable, and documented. For grading changes, include notes on reward / behavior impact.

**3. Validate locally**

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 &
python scripts/validate_submission.py
python scripts/test_generalization.py
```

**4. Open a Pull Request**

- Describe what changed and why
- Include sample output or score comparisons if graders were modified
- Reference any relevant issues

**Areas we'd welcome contributions in:**

- 🧪 New task types (e.g., consent form review, lab value flagging)
- 📊 Additional grader dimensions
- 🌍 Multilingual case bank entries
- 🧰 Client SDK improvements
- 📖 Documentation and examples

---

## 📄 License

This project is licensed under the **MIT License**. See [`LICENSE`](./LICENSE) for full terms.

---

## 🙏 Acknowledgements

- [OpenEnv](https://openenv.dev) — Environment specification and ecosystem
- [FastAPI](https://fastapi.tiangolo.com) — High-performance async API framework
- [Pydantic](https://docs.pydantic.dev) — Data validation and schema generation
- [Hugging Face Spaces](https://huggingface.co/spaces) — Hosted deployment platform
- [Meta LLaMA](https://llama.meta.com) — Foundation model for baseline inference
- [Groq](https://groq.com) — Ultra-fast LLM inference API

---

<div align="center">

Made with ❤️ for the OpenEnv hackathon community

<sub>If this project helped you, consider giving it a ⭐</sub>

</div>
