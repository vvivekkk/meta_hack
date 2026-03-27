# Clinical Trial Triage - OpenEnv

A production-style OpenEnv environment for clinical operations automation.

This repository simulates three high-impact trial operations tasks and provides:
- A typed FastAPI/OpenEnv runtime
- Deterministic graders and dense rewards
- A policy-search baseline (`inference.py`)
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

As of the latest local run:
- `inference.py` baseline mean score: **0.8856**
- Per-task:
  - `adverse_event_triage`: **0.8400**
  - `protocol_deviation_audit`: **0.9567**
  - `safety_narrative_generation`: **0.8600**
- Output file: `outputs/baseline_results.json`

Validator status:
- `python scripts/validate_submission.py` passes end-to-end checks.
- Validator internally runs `scripts/baseline_inference.py`, which may produce a different output schema from `inference.py` (details below).

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

## 6.3 Inference Baseline Search (`inference.py`)

`inference.py` runs **best-of policy rollouts** per task:
- `heuristic`
- `hybrid` (LLM + guardrails)
- `llm` (LLM with guardrails/fallback)

Then selects the highest scoring policy per task and writes combined results.

Guardrail logic:
- Enum/domain correction
- field clipping/sanitization
- payload fallback to heuristics when parse/provider fails
- AE-specific GI/liver correction to avoid common mis-coding

This is why `inference.py` can stay strong even when external LLM calls fail.

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
- policy-search baseline (`llm` and `hybrid` policies)
- generating candidate actions from observation text

Environment variables:
- `API_BASE_URL` (default `https://router.huggingface.co/v1`)
- `HF_TOKEN` (or `API_KEY` fallback)
- `MODEL_NAME` (default `meta-llama/Llama-3.3-70B-Instruct`)

Client:
- `openai.OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)`

### Path B: `scripts/baseline_inference.py` (Groq SDK)

Used for:
- Groq baseline script
- also used indirectly by validator

Environment variables:
- `GROQ_API_KEY` or `GROQ_API_KEYS`
- `BASELINE_MODEL` (default `llama-3.3-70b-versatile`)

Client:
- `groq.Groq(...)` via `scripts/groq_key_pool.py`

### Summary Table

| Script | Provider path | Default model | Key env var(s) | Purpose |
|---|---|---|---|---|
| `inference.py` | HF router (OpenAI SDK) | `meta-llama/Llama-3.3-70B-Instruct` | `HF_TOKEN` | Best-of policy-search baseline |
| `scripts/baseline_inference.py` | Groq SDK | `llama-3.3-70b-versatile` | `GROQ_API_KEY(S)` | Groq baseline and validator baseline check |

---

## 8) Submission Requirements: What Is Checked and Which Script Matters

The local submission gate is `scripts/validate_submission.py`.

It checks:
1. Core endpoints respond (`/`, `/health`, `/tasks`)
2. OpenEnv endpoints respond (`/openenv/metadata`, `/openenv/schema`, reset/step/state/health)
3. Each task can complete and score in `[0.0, 1.0]`
4. Baseline script runs and writes `outputs/baseline_results.json`

Important detail:
- Validator executes `scripts/baseline_inference.py`.
- It expects `overall_mean_reward` in output JSON.

This means there are **two valid baseline outputs** depending on what you run last:

1. If you run `inference.py` last:
- output schema includes `mean_score`, `search_mode`, `selected_policy`

2. If you run validator (or `scripts/baseline_inference.py`) last:
- output schema includes `overall_mean_reward`

Recommended submission run order:
1. Start server
2. Run validator
3. If competition expects your policy-search artifact format, run `inference.py` after validator

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

---

## 10) Full File-by-File Map (What Each File Does)

### Root

- `models.py`
- all typed action/observation/reward/state models

- `client.py`
- async and sync client wrappers
- now includes `session_id` support and `X-Session-ID` header propagation

- `inference.py`
- primary baseline for this repo's optimized scoring path
- best-of rollout search across policies
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
3. Re-run `inference.py`; it already does policy search and picks best task policy.
4. If `402` appears in `inference.py`, add HF provider credits or use a valid paid route.
5. Validate with `scripts/validate_submission.py` to ensure compliance.

---

## 16) Known Caveats

- `inference.py` and `scripts/baseline_inference.py` produce different JSON top-level score keys (`mean_score` vs `overall_mean_reward`).
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

## 18) License

MIT
