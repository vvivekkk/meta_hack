# Clinical Trial Triage - OpenEnv

A production-style OpenEnv environment for clinical operations automation.

This repository simulates three high-impact trial operations tasks and provides:
- A typed FastAPI/OpenEnv runtime
- Deterministic graders and dense rewards
- A fail-safe baseline (`inference.py`)
- A Groq-based baseline (`scripts/baseline_inference.py`)
- PPO RL training/evaluation stack
- A submission validator (`scripts/validate_submission.py`)

---

## 1) What This Project Actually Does

Clinical studies create continuous operational load across safety and compliance workflows. This environment models those workflows as RL episodes:

1. **Adverse Event Triage**
- Severity class (CTCAE-like labels)
- Expedited reporting timeline (7-day, 15-day, routine)
- MedDRA-like coding fields

2. **Protocol Deviation Audit**
- Major/minor/protocol-amendment classification
- CAPA requirement
- Site risk score (0-10)
- Finding IDs to escalate

3. **Safety Narrative Generation**
- Structured ICSR-style narrative text
- Causality class
- Temporal flags
- Dechallenge/rechallenge indicators

The environment is intentionally multi-modal:
- Works as plain REST API
- Works as OpenEnv-native `/openenv/*`
- Works inside RL loops via `gymnasium` wrapper

---

## 2) Current Project Status (as implemented)

As of the latest local runs on unseen production cases (`python scripts/test_generalization.py`):
- **LLM-enabled best run**: mean **0.9854**
  - `adverse_event_triage`: **1.0000**
  - `protocol_deviation_audit`: **0.9983**
  - `safety_narrative_generation`: **0.9580**
- **Provider-failure fallback run** (HF router call failures / 402 path): mean **0.9534**
  - `adverse_event_triage`: **1.0000**
  - `protocol_deviation_audit`: **0.9483**
  - `safety_narrative_generation`: **0.9120**
- **Current artifact in repo**: `outputs/generalization_results.json` (latest run state)

Validator status:
- `python scripts/validate_submission.py` passes end-to-end checks.
- Validator runs root `inference.py` (submission-required script name/location) and verifies output schema and runtime budget.

---

## 3) End-to-End Runtime Architecture

```
+-----------------------+
|   Client / Script     |
|  (curl, UI, Python)   |
+-----------+-----------+
            |
            v
+-----------------------+
| FastAPI app           |
| server/app.py         |
| /reset /step /state   |
| /grader /baseline     |
+-----------+-----------+
            |
            v
+-----------------------+
| ClinicalTrialEnvironment
| server/environment.py |
| state machine, cases, |
| reward shaping        |
+-----------+-----------+
            |
            +----------------------+
            |                      |
            v                      v
+-----------------------+  +-----------------------+
| tasks/case_bank.py    |  | tasks/graders.py      |
| synthetic cases       |  | deterministic scoring |
+-----------------------+  +-----------------------+

OpenEnv path:
/openenv/* -> server/openenv_env.py -> same core environment semantics
```

Important implementation behavior:
- Main REST API supports explicit session isolation with `X-Session-ID`.
- OpenEnv mount uses shared adapter with serialized concurrency to preserve reset-step continuity.

---

## 4) Data: What Exists and How It Flows

### 4.1 Source Data

All task data is local in-code case banks:
- `tasks/case_bank.py` (base cases)
- `tasks/production_cases.py` (extra production-style cases)

The environment loads:
- `AE_CASES`
- `DEVIATION_CASES`
- `NARRATIVE_CASES`

and extends these with `EXTRA_*` case sets when available.

### 4.2 Episode Data Flow

For each episode:
1. `POST /reset` chooses task and first case.
2. `POST /step` receives typed action payload.
3. Correct grader is invoked:
- `grade_ae_triage`
- `grade_protocol_deviation`
- `grade_safety_narrative`
4. Reward shaping applies bonuses/penalties.
5. `GET /grader` reports normalized episode score.

### 4.3 Output Artifacts

Primary output artifacts:
- `outputs/baseline_results.json`
- `outputs/groq_key_usage.json` (Groq key-pool state)
- `outputs/rl/*` (RL training/evaluation artifacts)

---

## 5) Models and Schemas

Typed contracts are in `models.py`.

Top-level action model:
- `TriageAction`
- exactly one of:
  - `ae_triage`
  - `deviation_audit`
  - `safety_narrative`

Top-level observation model:
- `TriageObservation`
- task-specific nested payload:
  - `ae_observation`
  - `deviation_observation`
  - `narrative_observation`

Reward model:
- `TriageReward` with task-specific sub-scores and penalty flags.

State model:
- `TriageState` containing episode metadata and action log.

---

## 6) Algorithms: What Is Actually Running

## 6.1 Deterministic Graders (`tasks/graders.py`)

### AE grading
Weighted score from:
- severity
- timeline
- is_serious
- SOC fuzzy match
- PT fuzzy match

### Deviation grading
Weighted score from:
- deviation type
- CAPA correctness
- risk score proximity
- violation recall
- violation precision

### Narrative grading
Composite score from:
- narrative completeness
- temporal coverage
- causality accuracy
- regulatory compliance

Generalization upgrades implemented:
- Dynamic keyword extraction from case payload
- Case-specific regulatory flag handling
- Non-warfarin narratives now score correctly (for expanded data)

## 6.2 Reward Shaping (`server/environment.py`)

Shaping logic includes:
- small positive bonus for partial progress (0.3-0.6 raw)
- penalty deduction when action is structurally invalid
- anti-gaming severity penalty on suspicious inflation
- anti-loop penalty for repeated identical actions

## 6.3 Inference Baseline (`inference.py`)

`inference.py` now uses a **hybrid fail-safe strategy** per case:
1. Try a retry-limited LLM call (`safe_llm_call`, up to 2 attempts) using OpenAI SDK + HF router variables.
2. Normalize action JSON to the strict task schema.
3. Apply task-specific guardrails:
- `protocol_deviation_audit`: calibrate LLM output against deterministic risk anchors.
- `safety_narrative_generation`: repair/enrich narrative text and temporal flags to satisfy regulatory grading criteria.
4. If LLM output is unavailable or unusable after normalization/guardrails, immediately use deterministic heuristic fallback.
5. Continue episode execution without crashing.

Reliability guarantees:
- bounded retry only (no unbounded stalls)
- deterministic fallback for all tasks
- task-level post-processing to keep LLM outputs rubric-aligned
- output is always written, even when API access is unavailable

This design keeps compliance support for OpenAI/HF variables while making baseline execution reproducible and robust.

## 6.4 RL Pipeline (`rl/*`)

- `rl/gym_env.py`: task wrapper as `gymnasium.Env`
- `rl/featurizer.py`: 128-d vector encoding
- `rl/action_templates.py`: parameterized `MultiDiscrete` action mapping
- `rl/train.py`: PPO training loop (`stable-baselines3`)
- `rl/evaluate.py`: per-task metrics for saved policies

---

## 7) Which LLM Are We Using, Exactly, and For What?

Two different LLM paths exist in this repo.

### Path A: `inference.py` (HF Router via OpenAI SDK)

Used for:
- retry-limited candidate action generation per case
- schema normalization + task-specific LLM guardrails
- immediate deterministic fallback if LLM is unavailable, invalid, or provider-limited

Environment variables:
- `API_BASE_URL` (default `https://router.huggingface.co/v1`)
- `HF_TOKEN` (or `API_KEY` fallback)
- `MODEL_NAME` (default `meta-llama/Llama-3.3-70B-Instruct`)

Client:
- `openai.OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)`

### Path B: `scripts/baseline_inference.py` (Groq SDK)

Used for:
- Groq baseline script for alternative provider testing

Environment variables:
- `GROQ_API_KEY` or `GROQ_API_KEYS`
- `BASELINE_MODEL` (default `llama-3.3-70b-versatile`)

Client:
- `groq.Groq(...)` via `scripts/groq_key_pool.py`

### Summary Table

| Script | Provider path | Default model | Key env var(s) | Purpose |
|---|---|---|---|---|
| `inference.py` | HF router (OpenAI SDK) | `meta-llama/Llama-3.3-70B-Instruct` | `HF_TOKEN` | Hybrid baseline (retry-limited LLM + normalization/guardrails + deterministic fallback) |
| `scripts/baseline_inference.py` | Groq SDK | `llama-3.3-70b-versatile` | `GROQ_API_KEY(S)` | Groq baseline for alternative provider benchmarking |

---

## 8) Submission Requirements: What Is Checked and Which Script Matters

The local submission gate is `scripts/validate_submission.py`.

It checks:
1. Core endpoints respond (`/`, `/health`, `/tasks`)
2. OpenEnv endpoints respond (`/openenv/metadata`, `/openenv/schema`, reset/step/state/health)
3. Each task can complete and score in `[0.0, 1.0]`
4. Baseline script runs and writes `outputs/baseline_results.json`

Important detail:
- Validator executes root `inference.py`.
- It expects both `mean_score` and `overall_mean_reward` in output JSON.

Compatibility note:
- `inference.py` now writes both `mean_score` and `overall_mean_reward` with identical values.
- This avoids downstream schema mismatch for score readers.

Runtime constraint:
- `inference.py` is expected to complete in under 20 minutes on constrained infra (2 vCPU / 8 GB RAM).

Recommended submission run order:
1. Start server
2. Run `python inference.py`
3. Run `python scripts/validate_submission.py`

---

## 9) Why 402 Happened: HF Token or Groq Key?

Short answer from observed behavior:
- The `402` came from **HF Inference Providers credits** in the `inference.py` path.
- That path uses `HF_TOKEN` against `https://router.huggingface.co/v1`.

The typical error text is:
- "depleted your monthly included credits..."

So this is tied to Hugging Face account credit state, not Groq key parsing.

How Groq behaves in this repo:
- If Groq keys are missing in `scripts/baseline_inference.py`, it falls back to heuristic baseline.
- Groq failures are handled via key-pool retries/backoff and usually present as auth/rate-limit/provider errors, not the HF credit message.

Clear interpretation:
1. `inference.py` + `HF_TOKEN` + HF router -> **402 means HF credits exhausted**.
2. `scripts/baseline_inference.py` + `GROQ_API_KEY(S)` -> Groq path; different failure modes.

Runtime behavior in this repo when HF path fails:
1. Exceptions (including 402) are caught in `safe_llm_call`.
2. The episode does not crash; task action falls back to deterministic logic.
3. Results are still emitted to output JSON.
4. On unseen production cases, this fallback profile still stays strong (observed mean around **0.9534**).

---

## 10) Full File-by-File Map (What Each File Does)

### Root

- `models.py`
- all typed action/observation/reward/state models

- `client.py`
- async and sync client wrappers
- now includes `session_id` support and `X-Session-ID` header propagation

- `inference.py`
- primary fail-safe baseline for this repo
- retry-limited LLM call + schema guardrails + deterministic fallback on any failure
- writes `outputs/baseline_results.json`

- `openenv.yaml`
- metadata for OpenEnv packaging/discovery

- `Dockerfile`
- image build and runtime entrypoint (`uvicorn server.app:app --port 7860`)

### `server/`

- `server/app.py`
- FastAPI app
- REST endpoints and OpenEnv mount
- leaderboard and web UI endpoint

- `server/environment.py`
- core environment state machine
- reset/step/state
- reward shaping
- in-memory session store

- `server/openenv_env.py`
- OpenEnv adapter wrapper over core environment

### `tasks/`

- `tasks/case_bank.py`
- base synthetic cases and ground truth

- `tasks/production_cases.py`
- extra realism cases for generalization

- `tasks/graders.py`
- deterministic graders for all tasks
- narrative generalization logic and dynamic compliance flags

### `scripts/`

- `scripts/heuristic_baseline.py`
- deterministic no-LLM baseline

- `scripts/groq_key_pool.py`
- key parsing, cooldown, usage tracking

- `scripts/baseline_inference.py`
- Groq LLM baseline with fallback

- `scripts/validate_submission.py`
- end-to-end pre-submit checks

### `rl/`

- `rl/featurizer.py` -> observation encoding
- `rl/action_templates.py` -> policy vector to typed action mapping
- `rl/gym_env.py` -> gym wrapper
- `rl/train.py` -> PPO training
- `rl/evaluate.py` -> policy evaluation
- `rl/smoke_test.py` -> quick smoke run

### `tests/`

- `test_openenv_adapter.py` -> OpenEnv integration tests
- `test_rl_env.py` -> RL env shape/step checks
- `test_session_isolation.py` -> main API session isolation
- `test_narrative_grader_generalization.py` -> narrative grading non-warfarin case

---

## 11) API Endpoints

### Main environment API

- `GET /`
- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /grader`
- `POST /baseline`
- `GET /leaderboard`
- `GET /web`

Main API supports `X-Session-ID` for episode isolation.

### OpenEnv-native API

- `POST /openenv/reset`
- `POST /openenv/step`
- `GET /openenv/state`
- `GET /openenv/schema`
- `GET /openenv/metadata`
- `GET /openenv/health`

---

## 12) Setup and Run

## 12.1 Local setup

```bash
# from repo root
python -m venv .venv
# activate venv (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 12.2 Start server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## 12.3 Run optimized baseline

```bash
python inference.py
```

## 12.4 Run submission validator

```bash
python scripts/validate_submission.py
```

## 12.5 Run tests

```bash
python -m pytest -q
```

---

## 13) Environment Variables and Secrets

Submission-required environment variables for LLM path:
- `API_BASE_URL`: API endpoint for the LLM provider.
- `MODEL_NAME`: model identifier used by `inference.py`.
- `HF_TOKEN`: Hugging Face/API key.

LLM client requirement:
- `inference.py` uses `openai.OpenAI(...)` for all LLM calls.
- The required submission inference script is root `inference.py`.

| Variable | Used by | Meaning |
|---|---|---|
| `HF_TOKEN` | `inference.py` | HF token for router-based LLM calls |
| `API_BASE_URL` | `inference.py` | HF router URL (default set) |
| `MODEL_NAME` | `inference.py` | HF model string |
| `ENV_SERVER_URL` | `inference.py` | Server URL for reset/step/grader |
| `GROQ_API_KEY` | `scripts/baseline_inference.py` | single Groq key |
| `GROQ_API_KEYS` | `scripts/baseline_inference.py` | key pool list |
| `BASELINE_MODEL` | `scripts/baseline_inference.py` | Groq model name |
| `VALIDATOR_BASE_URL` | `scripts/validate_submission.py` | validator target base URL |

Never hardcode keys in source.

---

## 14) Scoring Interpretation

In this environment:
- scores are normalized in `[0.0, 1.0]`
- higher is better
- deterministic graders are strict on structural correctness and domain-specific content

Practical interpretation:
- `>= 0.85` is very strong for this benchmark version
- high variance usually indicates prompt/parse/provider instability, not grader randomness

---

## 15) What To Do If You See Low Scores

1. Verify server health and no stale session collisions.
2. Ensure correct model/provider env vars are set for the script you are running.
3. Run `python scripts/test_generalization.py` and inspect per-task deltas in `outputs/generalization_results.json`.
4. If `402` appears in `inference.py`, add HF provider credits or use a valid paid route; until then, fallback mode remains active and stable.
5. Re-run after connectivity/provider recovery to regain peak LLM-assisted scores.
6. Validate with `scripts/validate_submission.py` to ensure compliance.

---

## 16) Known Caveats

- HF provider `402` credit exhaustion can force deterministic fallback mode even with valid key syntax.
- `inference.py` now includes both `mean_score` and `overall_mean_reward` for compatibility.
- Both scripts write to the same file path (`outputs/baseline_results.json`), so run order matters for artifact format.
- OpenEnv mount is serialized for continuity safety in current implementation.

---

## 17) Minimal Repro Commands (Copy/Paste)

```bash
# 1) start server
uvicorn server.app:app --host 127.0.0.1 --port 8000

# 2) optimized baseline
python inference.py

# 3) compliance validation
python scripts/validate_submission.py

# 4) tests
python -m pytest -q
```

---

## 18) Submission Compliance Checklist

This project is implemented to satisfy the required submission constraints:

1. Real-world task simulation:
- Clinical trial operations workflows (AE triage, protocol deviations, ICSR narrative writing).

2. OpenEnv spec compliance:
- Typed Pydantic models in `models.py`.
- Endpoints implemented: `reset()`, `step()`, `state()` and `/openenv/*` surfaces.
- Metadata declared in `openenv.yaml`.

3. Minimum 3 tasks with deterministic graders:
- `adverse_event_triage` (easy)
- `protocol_deviation_audit` (medium)
- `safety_narrative_generation` (hard)
- Reward outputs are normalized to `[0.0, 1.0]`.

4. Meaningful reward function:
- Dense partial-credit scoring via task-specific graders plus reward shaping.

5. Baseline inference script requirement:
- Root `inference.py` is the submission baseline script.
- LLM calls use `openai.OpenAI(...)` and env vars (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`).

6. Deployable container:
- `Dockerfile` starts the FastAPI app on HF Spaces-compatible port (`7860` by default).

### Hugging Face Spaces Deployment Steps

1. Create a new Space on Hugging Face:
- SDK: `Docker`
- Visibility: your choice (private/public)

2. Push this repository contents to the Space repo (must include `Dockerfile`, `openenv.yaml`, root `inference.py`).

3. In Space settings, configure Variables/Secrets:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

4. Wait for Space build logs to complete successfully.

5. Smoke test the deployed Space:
- `GET /health` must return 200
- `POST /reset` must return 200 and include an observation payload

6. Run local validator before final submission:
- `python scripts/validate_submission.py`

### Explicit Submission Pass Conditions

All of the following must pass:

1. HF Space deploys successfully from submitted repo.
2. Automated ping to Space URL returns HTTP 200 and reset works.
3. OpenEnv spec checks pass (`openenv.yaml`, typed models, reset/step/state/OpenEnv endpoints).
4. Docker build succeeds from submitted repository.
5. Root `inference.py` runs to completion and writes reproducible score output.
6. At least 3 tasks exist with graders and each grader score is in `[0.0, 1.0]`.
7. Runtime stays under 20 minutes on constrained infra (~2 vCPU / 8 GB RAM).

### Pre-Submission Commands

```bash
# 1) Build container
docker build -t clinical-trial-triage .

# 2) Run container locally (HF Spaces equivalent)
docker run --rm -p 7860:7860 clinical-trial-triage

# 3) Ping + reset checks (must return 200)
curl -i http://127.0.0.1:7860/health
curl -i -X POST http://127.0.0.1:7860/reset -H "Content-Type: application/json" -d '{"task_id":"adverse_event_triage"}'

# 4) Validate OpenEnv + baseline + graders
python scripts/validate_submission.py

# 5) Reproducible baseline and unseen-case checks
python inference.py
python scripts/test_generalization.py
```

HF Space URL smoke test pattern:

```bash
curl -i https://<your-space>.hf.space/health
curl -i -X POST https://<your-space>.hf.space/reset -H "Content-Type: application/json" -d '{"task_id":"adverse_event_triage"}'
```

---

## 19) Generalization & Robustness

This repository includes a dedicated unseen-case evaluation script:

```bash
python scripts/test_generalization.py
```

What it does:
- evaluates only unknown production cases from `tasks/production_cases.py`
- reuses the same action policy path from `inference.py` (retry-limited LLM, normalization/guardrails, deterministic fallback)
- remains runnable when no valid LLM key is available (falls back automatically)

Output artifact:
- `outputs/generalization_results.json`

### Observed Score Bands (Local)

| Mode | `adverse_event_triage` | `protocol_deviation_audit` | `safety_narrative_generation` | `mean_score` |
|---|---:|---:|---:|---:|
| LLM-assisted best run | 1.0000 | 0.9983 | 0.9580 | **0.9854** |
| LLM unavailable / provider failure fallback | 1.0000 | 0.9483 | 0.9120 | **0.9534** |
| Mixed run (latest checked-in artifact example) | 1.0000 | 0.9983 | 0.9120 | **0.9701** |

### API Failure Behavior (Established Contract)

- If HF provider calls fail (including `402` credit exhaustion), execution continues.
- The runner logs failure and routes each affected task to deterministic fallback.
- The script still writes a valid `outputs/generalization_results.json`.
- No manual intervention is required to complete a run.

Output format:

```json
{
  "per_task_scores": {
    "adverse_event_triage": 1.0,
    "protocol_deviation_audit": 0.9983,
    "safety_narrative_generation": 0.912
  },
  "mean_score": 0.9701
}
```

Interpretation:
- `per_task_scores` reports robustness by task on unseen cases.
- `mean_score` is the arithmetic mean across the 3 task-level scores.

Optional diagnostic command for forced fallback-only run:

```powershell
$env:API_BASE_URL='http://127.0.0.1:9/v1'; python scripts/test_generalization.py
```

---

## 20) License

MIT
