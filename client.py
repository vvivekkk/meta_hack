"""
Clinical Trial Triage — EnvClient
===================================
OpenEnv-compatible client for interacting with the environment server.
Supports both async (recommended) and sync usage patterns.

Usage (async):
    from client import ClinicalTrialEnv, TriageAction

    async with ClinicalTrialEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset(task_id="adverse_event_triage")
        result = await env.step(action)

Usage (sync):
    with ClinicalTrialEnv(base_url="http://localhost:8000").sync() as env:
        obs = env.reset(task_id="adverse_event_triage")
        result = env.step(action)

Usage (HF Spaces):
    async with ClinicalTrialEnv(base_url="https://YOUR_HF_SPACE.hf.space") as env:
        obs = await env.reset()
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx

from models import (
    StepResult,
    TaskID,
    TriageAction,
    TriageObservation,
    TriageState,
)


class ClinicalTrialEnv:
    """
    Async environment client implementing OpenEnv EnvClient pattern.
    Communicates with the FastAPI server over HTTP.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        session_id: str = "default",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id = session_id.strip() or "default"
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "ClinicalTrialEnv":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"X-Session-ID": self.session_id},
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "Client not initialized. Use 'async with ClinicalTrialEnv(...) as env:'"
            )
        return self._client

    async def reset(
        self, task_id: str = TaskID.ADVERSE_EVENT_TRIAGE
    ) -> TriageObservation:
        """Initialize a new episode. Returns initial observation."""
        client = self._ensure_client()
        response = await client.post("/reset", json={"task_id": task_id})
        response.raise_for_status()
        data = response.json()
        return TriageObservation(**data["observation"])

    async def step(self, action: TriageAction) -> StepResult:
        """Execute one action. Returns StepResult(observation, reward, done, info)."""
        client = self._ensure_client()
        response = await client.post("/step", json=action.model_dump())
        response.raise_for_status()
        return StepResult(**response.json())

    async def state(self) -> TriageState:
        """Return current episode state metadata."""
        client = self._ensure_client()
        response = await client.get("/state")
        response.raise_for_status()
        return TriageState(**response.json())

    async def tasks(self) -> Dict[str, Any]:
        """Return available tasks and action schemas."""
        client = self._ensure_client()
        response = await client.get("/tasks")
        response.raise_for_status()
        return response.json()

    async def grader(self) -> Dict[str, Any]:
        """Return grader scores for the last completed episode."""
        client = self._ensure_client()
        response = await client.get("/grader")
        response.raise_for_status()
        return response.json()

    def sync(self) -> "SyncClinicalTrialEnv":
        """Return a synchronous wrapper around this client."""
        return SyncClinicalTrialEnv(self.base_url, self.timeout, self.session_id)


class SyncClinicalTrialEnv:
    """Synchronous wrapper for environments that don't use async."""

    def __init__(self, base_url: str, timeout: float = 30.0, session_id: str = "default") -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id = session_id.strip() or "default"
        self._client: Optional[httpx.Client] = None

    def __enter__(self) -> "SyncClinicalTrialEnv":
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"X-Session-ID": self.session_id},
        )
        return self

    def __exit__(self, *args) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def reset(self, task_id: str = TaskID.ADVERSE_EVENT_TRIAGE) -> TriageObservation:
        assert self._client
        r = self._client.post("/reset", json={"task_id": task_id})
        r.raise_for_status()
        return TriageObservation(**r.json()["observation"])

    def step(self, action: TriageAction) -> StepResult:
        assert self._client
        r = self._client.post("/step", json=action.model_dump())
        r.raise_for_status()
        return StepResult(**r.json())

    def state(self) -> TriageState:
        assert self._client
        r = self._client.get("/state")
        r.raise_for_status()
        return TriageState(**r.json())


# ─────────────────────────────────────────
# QUICK DEMO
# ─────────────────────────────────────────

async def _demo():
    """Quick async demo of the environment client."""
    from models import AdverseEventTriageAction, TriageAction

    print("Clinical Trial Triage — Client Demo")
    print("=" * 50)

    async with ClinicalTrialEnv(base_url="http://localhost:8000") as env:
        # Get available tasks
        tasks = await env.tasks()
        print(f"Available tasks: {[t['id'] for t in tasks['tasks']]}\n")

        # Reset AE triage episode
        obs = await env.reset(task_id="adverse_event_triage")
        print(f"Task: {obs.task_id}")
        if obs.ae_observation:
            print(f"Case: {obs.ae_observation.case_id}")
            print(f"AE: {obs.ae_observation.ae_description}")

        # Send a test action
        action = TriageAction(
            task_id="adverse_event_triage",
            ae_triage=AdverseEventTriageAction(
                severity_classification="life_threatening",
                reporting_timeline="7-day",
                meddra_soc="Cardiac disorders",
                meddra_preferred_term="Myocardial infarction",
                is_serious=True,
                rationale="STEMI with troponin elevation — life-threatening SAE requiring 7-day reporting.",
            ),
        )
        result = await env.step(action)
        print(f"\nReward: {result.reward:.4f}")
        print(f"Done: {result.done}")
        print(f"Reward detail: {result.reward_detail.model_dump()}")


if __name__ == "__main__":
    asyncio.run(_demo())
