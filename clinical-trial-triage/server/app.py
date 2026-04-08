"""
Clinical Trial Triage - FastAPI Application
===========================================
Exposes OpenEnv-compliant HTTP endpoints:
  GET  /
  POST /reset
  POST /step
  GET  /state
  GET  /tasks
  GET  /grader
  POST /baseline
  GET  /leaderboard
  GET  /health
  GET  /web
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server import ConcurrencyConfig, create_fastapi_app
from pydantic import BaseModel

from models import TaskID, TriageAction
from server.environment import ClinicalTrialEnvironment, clear_session, get_or_create_session
from server.openenv_env import (
    ClinicalTrialOpenEnv,
    OpenEnvTriageAction,
    OpenEnvTriageObservation,
)


logger = logging.getLogger("uvicorn.error")



@asynccontextmanager
async def lifespan(app: FastAPI):
    clear_session("default")
    get_or_create_session("default")
    yield


app = FastAPI(
    title="Clinical Trial Triage - OpenEnv",
    description=(
        "An OpenEnv-compatible RL environment simulating clinical trial "
        "adverse event triage, protocol deviation auditing, and safety "
        "narrative generation for pharmaceutical AI training."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
if UI_DIR.exists():
  app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

_openenv_shared_env = ClinicalTrialOpenEnv()


def _openenv_env_factory() -> ClinicalTrialOpenEnv:
  # Keep a shared adapter instance so OpenEnv reset/step continuity is preserved.
  return _openenv_shared_env


openenv_app = create_fastapi_app(
    env=_openenv_env_factory,
    action_cls=OpenEnvTriageAction,
    observation_cls=OpenEnvTriageObservation,
  concurrency_config=ConcurrencyConfig(max_concurrent_envs=1),
)
app.mount("/openenv", openenv_app)


class ResetRequest(BaseModel):
    task_id: str = TaskID.ADVERSE_EVENT_TRIAGE


class BaselineRequest(BaseModel):
    task_id: Optional[str] = None


class InferenceStepRequest(BaseModel):
  task_id: str = TaskID.ADVERSE_EVENT_TRIAGE
  force_reset: bool = True


_leaderboard: list[Dict[str, Any]] = []


def _safe_session_id(raw_session_id: Optional[str]) -> str:
    session_id = (raw_session_id or "default").strip()
    return session_id or "default"


def _record_episode(session_id: str, task_id: str, normalized_score: float) -> None:
    score = max(0.0, min(1.0, float(normalized_score)))
    item = {
        "session_id": session_id,
        "mean_score": round(score, 4),
        "task_scores": {task_id: round(score, 4)},
        "timestamp": time.time(),
    }
    _leaderboard.append(item)


def _run_single_task_baseline(task_id: str) -> Dict[str, Any]:
    from scripts.heuristic_baseline import (
        _heuristic_ae_triage,
        _heuristic_deviation_audit,
        _heuristic_narrative,
    )
    from tasks.case_bank import AE_CASES, DEVIATION_CASES, NARRATIVE_CASES

    env = ClinicalTrialEnvironment()
    env.reset(task_id=task_id)

    if task_id == TaskID.ADVERSE_EVENT_TRIAGE:
        cases = AE_CASES
        action_builder = _heuristic_ae_triage
    elif task_id == TaskID.PROTOCOL_DEVIATION_AUDIT:
        cases = DEVIATION_CASES
        action_builder = _heuristic_deviation_audit
    elif task_id == TaskID.SAFETY_NARRATIVE_GENERATION:
        cases = NARRATIVE_CASES
        action_builder = _heuristic_narrative
    else:
        raise HTTPException(status_code=422, detail=f"Unsupported task_id: {task_id}")

    rewards: list[float] = []
    for case in cases:
        result = env.step(action_builder(case))
        rewards.append(result.reward)
        if result.done:
            break

    mean_reward = round(sum(rewards) / len(rewards), 4) if rewards else 0.0
    return {
        "baseline_type": "heuristic",
        "task_id": task_id,
        "per_step_rewards": rewards,
        "mean_reward": mean_reward,
        "n_steps": len(rewards),
    }


@app.post("/reset")
async def reset(
    request: ResetRequest,
    x_session_id: Optional[str] = Header(default="default"),
) -> Dict[str, Any]:
    session_id = _safe_session_id(x_session_id)
    env = get_or_create_session(session_id)
    logger.info("reset request: session_id=%s task_id=%s", session_id, request.task_id)
    try:
        obs = env.reset(task_id=request.task_id)
        logger.info("reset complete: session_id=%s task_id=%s", session_id, request.task_id)
        return {"observation": obs.model_dump(), "status": "ok"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("reset failed: session_id=%s task_id=%s", session_id, request.task_id)
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
async def step(
    action: TriageAction,
    x_session_id: Optional[str] = Header(default="default"),
) -> Dict[str, Any]:
    session_id = _safe_session_id(x_session_id)
    env = get_or_create_session(session_id)
    logger.info("step request: session_id=%s task_id=%s", session_id, action.task_id)
    try:
        result = env.step(action)
        logger.info(
            "step result: session_id=%s task_id=%s reward=%.4f done=%s",
            session_id,
            action.task_id,
            float(result.reward),
            bool(result.done),
        )
        if result.done:
            state = env.state()
            normalized = state.cumulative_reward / max(state.step_count, 1)
            _record_episode(session_id=session_id, task_id=str(state.task_id), normalized_score=normalized)
            logger.info(
                "episode complete: session_id=%s task_id=%s normalized_score=%.4f",
                session_id,
                state.task_id,
                float(normalized),
            )
        return result.model_dump()
    except RuntimeError as exc:
        logger.warning("step runtime error: session_id=%s detail=%s", session_id, str(exc))
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        logger.warning("step validation error: session_id=%s detail=%s", session_id, str(exc))
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/state")
async def state(x_session_id: Optional[str] = Header(default="default")) -> Dict[str, Any]:
    env = get_or_create_session(_safe_session_id(x_session_id))
    try:
        s = env.state()
        return s.model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/tasks")
async def tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "id": TaskID.ADVERSE_EVENT_TRIAGE,
                "name": "Adverse Event Triage",
                "difficulty": "easy",
                "description": (
                    "Classify incoming adverse event reports by severity and reporting timeline. "
                    "Determine MedDRA SOC and Preferred Term."
                ),
                "max_steps": 3,
                "action_schema": {
                    "task_id": "string (adverse_event_triage)",
                    "ae_triage": {
                        "severity_classification": "enum: mild|moderate|severe|life_threatening|fatal",
                        "reporting_timeline": "enum: 7-day|15-day|routine",
                        "meddra_soc": "string (e.g., 'Cardiac disorders')",
                        "meddra_preferred_term": "string (e.g., 'Myocardial infarction')",
                        "is_serious": "boolean",
                        "rationale": "string (max 500 chars)",
                    },
                },
            },
            {
                "id": TaskID.PROTOCOL_DEVIATION_AUDIT,
                "name": "Protocol Deviation Audit",
                "difficulty": "medium",
                "description": (
                    "Audit site monitoring findings. Classify major/minor deviations, "
                    "flag GCP violations, and assess site risk."
                ),
                "max_steps": 3,
                "action_schema": {
                    "task_id": "string (protocol_deviation_audit)",
                    "deviation_audit": {
                        "deviation_type": "enum: major|minor|protocol_amendment",
                        "capa_required": "boolean",
                        "site_risk_score": "float 0.0-10.0",
                        "flagged_finding_ids": "list of strings (finding IDs)",
                        "recommended_action": "string (max 300 chars)",
                    },
                },
            },
            {
                "id": TaskID.SAFETY_NARRATIVE_GENERATION,
                "name": "Safety Narrative Generation",
                "difficulty": "hard",
                "description": (
                    "Generate an ICH E2B-compliant Individual Case Safety Report (ICSR) "
                    "narrative synthesizing patient data, AE details, causality, and outcome."
                ),
                "max_steps": 1,
                "action_schema": {
                    "task_id": "string (safety_narrative_generation)",
                    "safety_narrative": {
                        "narrative_text": "string (100-4000 chars, ICH E2B compliant)",
                        "causality_assessment": "enum: definitely_related|probably_related|possibly_related|unlikely_related|not_related|unassessable",
                        "key_temporal_flags": "list of strings",
                        "dechallenge_positive": "boolean or null",
                        "rechallenge_positive": "boolean or null",
                    },
                },
            },
        ]
    }


@app.get("/grader")
async def grader(x_session_id: Optional[str] = Header(default="default")) -> Dict[str, Any]:
    env = get_or_create_session(_safe_session_id(x_session_id))
    try:
        s = env.state()
        if not s.done:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Episode is still in progress. Complete all steps before calling /grader. "
                    f"Current progress: {s.step_count}/{s.max_steps}."
                ),
            )

        normalized_score = s.cumulative_reward / s.step_count if s.step_count > 0 else 0.0
        return {
            "episode_id": s.episode_id,
            "task_id": s.task_id,
            "done": s.done,
            "cumulative_reward": s.cumulative_reward,
            "step_count": s.step_count,
            "max_steps": s.max_steps,
            "normalized_score": normalized_score,
            "actions": s.actions_taken,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/baseline")
async def baseline(request: Optional[BaselineRequest] = None) -> Dict[str, Any]:
    if request and request.task_id:
        return _run_single_task_baseline(task_id=request.task_id)

    from scripts.heuristic_baseline import run_heuristic_baseline

    results = run_heuristic_baseline()
    results["inference_script"] = "scripts/heuristic_baseline.py"
    return results


@app.post("/infer/step")
async def infer_step(
    request: InferenceStepRequest,
    x_session_id: Optional[str] = Header(default="default"),
) -> Dict[str, Any]:
    session_id = _safe_session_id(x_session_id)
    env = get_or_create_session(session_id)

    try:
        from inference import CLIENT as INFERENCE_CLIENT
        from inference import choose_action
    except Exception as exc:  # noqa: BLE001
        logger.exception("inference import failed")
        raise HTTPException(status_code=500, detail=f"Inference module unavailable: {exc}")

    try:
        if request.force_reset:
            obs = env.reset(task_id=request.task_id)
        else:
            state = env.state()
            if state.done or str(state.task_id) != request.task_id:
                obs = env.reset(task_id=request.task_id)
            else:
                obs = env._build_observation()  # noqa: SLF001

        obs_payload = obs.model_dump()
        action_payload = choose_action(request.task_id, obs_payload)
        action = TriageAction.model_validate(action_payload)

        result = env.step(action)
        if result.done:
            state = env.state()
            normalized = state.cumulative_reward / max(state.step_count, 1)
            _record_episode(session_id=session_id, task_id=str(state.task_id), normalized_score=normalized)

        llm_enabled = INFERENCE_CLIENT is not None
        action_source = "llm_or_fallback" if llm_enabled else "heuristic_fallback"

        logger.info(
            "infer step: session_id=%s task_id=%s source=%s reward=%.4f done=%s",
            session_id,
            request.task_id,
            action_source,
            float(result.reward),
            bool(result.done),
        )

        return {
            "status": "ok",
            "session_id": session_id,
            "task_id": request.task_id,
            "llm_enabled": llm_enabled,
            "action_source": action_source,
            "action": action_payload,
            "step": result.model_dump(),
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/leaderboard")
async def leaderboard() -> Dict[str, Any]:
    top = sorted(_leaderboard, key=lambda item: item.get("mean_score", 0.0), reverse=True)[:10]
    return {
        "leaderboard": top,
        "total_episodes": len(_leaderboard),
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "env": "clinical-trial-triage",
        "version": "2.0.0",
        "openenv": "/openenv",
    }


@app.get("/")
async def root(request: Request) -> Any:
  accept = (request.headers.get("accept") or "").lower()
  if "text/html" in accept and UI_DIR.exists():
    return RedirectResponse(url="/ui/")

    return {
        "status": "ok",
        "message": "Clinical Trial Triage OpenEnv is running.",
        "endpoints": [
            "/reset",
            "/step",
            "/infer/step",
            "/state",
            "/tasks",
            "/grader",
            "/baseline",
            "/leaderboard",
            "/health",
            "/ui/",
            "/triage",
            "/openenv/reset",
            "/openenv/step",
            "/openenv/state",
            "/openenv/schema",
            "/openenv/metadata",
            "/openenv/health",
        ],
    }


@app.get("/triage")
async def triage_ui() -> RedirectResponse:
  if not UI_DIR.exists():
    raise HTTPException(status_code=404, detail="UI folder not found")
  return RedirectResponse(url="/ui/triage.html")


WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ClinicalTrialTriage - OpenEnv</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f8f9fb; color: #1a1a2e; }

  header {
    background: #0f3460; color: white;
    padding: 16px 32px;
    display: flex; align-items: center; gap: 16px;
  }
  .logo { font-size: 20px; font-weight: 700; letter-spacing: -0.5px; }
  .badge {
    background: #16213e; color: #e94560;
    font-size: 11px; padding: 2px 8px; border-radius: 4px;
    font-weight: 600; letter-spacing: 0.5px;
  }

  main { max-width: 1100px; margin: 32px auto; padding: 0 24px; }

  .stats-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
    margin-bottom: 32px;
  }
  .stat-card {
    background: white; border-radius: 12px; padding: 20px;
    border: 1px solid #e8eaef; text-align: center;
  }
  .stat-label { font-size: 12px; color: #6b7280; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-value { font-size: 32px; font-weight: 700; margin-top: 4px; }
  .stat-value.green { color: #059669; }
  .stat-value.blue { color: #2563eb; }
  .stat-value.amber { color: #d97706; }

  .panel { background: white; border-radius: 12px; border: 1px solid #e8eaef; margin-bottom: 24px; }
  .panel-header { padding: 16px 20px; border-bottom: 1px solid #e8eaef; font-weight: 600; font-size: 15px; }
  .panel-body { padding: 20px; }

  .task-row {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 0; border-bottom: 1px solid #f3f4f6;
  }
  .task-row:last-child { border-bottom: none; }
  .task-badge {
    font-size: 11px; padding: 3px 10px; border-radius: 20px; font-weight: 600;
    white-space: nowrap;
  }
  .easy { background: #d1fae5; color: #065f46; }
  .medium { background: #fef3c7; color: #92400e; }
  .hard { background: #fee2e2; color: #991b1b; }
  .task-name { flex: 1; font-size: 14px; font-weight: 500; }
  .task-score { font-size: 14px; font-weight: 700; color: #059669; }

  .run-btn {
    background: #0f3460; color: white;
    border: none; border-radius: 8px;
    padding: 12px 28px; font-size: 14px; font-weight: 600;
    cursor: pointer; transition: background 0.2s;
  }
  .run-btn:hover { background: #1a4a7a; }
  .run-btn:disabled { background: #9ca3af; cursor: not-allowed; }

  pre {
    background: #0f172a; color: #e2e8f0;
    padding: 16px; border-radius: 8px;
    font-size: 13px; overflow-x: auto;
    max-height: 300px; overflow-y: auto;
  }

  select, input {
    border: 1px solid #d1d5db; border-radius: 6px; padding: 8px 12px;
    font-size: 14px; width: 100%;
  }
  label { font-size: 13px; color: #6b7280; font-weight: 500; display: block; margin-bottom: 6px; }
  .form-row { margin-bottom: 16px; }
</style>
</head>
<body>
<header>
  <div>
    <div class="logo">Clinical Trial Triage</div>
  </div>
  <span class="badge">OpenEnv</span>
</header>

<main>
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">Tasks</div>
      <div class="stat-value blue">3</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">LLM baseline</div>
      <div class="stat-value green">0.86</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Domain</div>
      <div class="stat-value amber" style="font-size:18px;padding-top:8px">Pharma</div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-header">Tasks</div>
    <div class="panel-body">
      <div class="task-row">
        <span class="task-badge easy">Easy</span>
        <span class="task-name">Adverse Event Triage - CTCAE v5.0 severity, MedDRA coding, SAE flag</span>
        <span class="task-score">~0.88</span>
      </div>
      <div class="task-row">
        <span class="task-badge medium">Medium</span>
        <span class="task-name">Protocol Deviation Audit - GCP classification, CAPA, site risk scoring</span>
        <span class="task-score">~0.79</span>
      </div>
      <div class="task-row">
        <span class="task-badge hard">Hard</span>
        <span class="task-name">Safety Narrative Generation - ICH E2B(R3) ICSR, causality, 9 mandatory sections</span>
        <span class="task-score">~0.91</span>
      </div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-header">Try an episode</div>
    <div class="panel-body">
      <div class="form-row">
        <label>Task</label>
        <select id="taskSelect">
          <option value="adverse_event_triage">Adverse Event Triage</option>
          <option value="protocol_deviation_audit">Protocol Deviation Audit</option>
          <option value="safety_narrative_generation">Safety Narrative Generation</option>
        </select>
      </div>
      <button class="run-btn" onclick="runEpisode()">Run heuristic baseline</button>
      <div id="output" style="margin-top:16px"></div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-header">API quick reference</div>
    <div class="panel-body">
      <pre>POST /reset     {"task_id": "adverse_event_triage"}
GET  /state
POST /step      {TriageAction JSON}
GET  /grader    -> component scores
POST /baseline  -> run all 3 heuristic baselines
GET  /tasks     -> full action schemas
GET  /leaderboard

# OpenEnv native
POST /openenv/reset
POST /openenv/step
GET  /openenv/state
GET  /openenv/schema</pre>
    </div>
  </div>
</main>

<script>
async function runEpisode() {
  const task = document.getElementById('taskSelect').value;
  const btn = document.querySelector('.run-btn');
  const out = document.getElementById('output');
  btn.disabled = true;
  btn.textContent = 'Running...';
  out.innerHTML = '<p style="color:#6b7280;font-size:14px">Running baseline episode...</p>';

  try {
    const response = await fetch('/baseline', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({task_id: task})
    });
    const data = await response.json();
    out.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
  } catch (error) {
    out.innerHTML = '<p style="color:#dc2626">Error: ' + error.message + '</p>';
  }
  btn.disabled = false;
  btn.textContent = 'Run heuristic baseline';
}
</script>
</body>
</html>
"""


@app.get("/web", response_class=HTMLResponse)
async def web_interface() -> HTMLResponse:
    return HTMLResponse(content=WEB_UI_HTML)