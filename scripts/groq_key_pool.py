"""
Groq API key pool manager with dynamic routing and persisted usage stats.

This module supports:
- GROQ_API_KEYS (comma/semicolon/newline-separated values)
- GROQ_API_KEY (single key fallback)
- Dynamic key selection based on prior usage and failures
- Failure cooldowns with exponential backoff
- Persistent state for cross-run adaptation
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from groq import Groq


def _split_keys(raw: str) -> List[str]:
    normalized = raw.replace(";", ",").replace("\n", ",")
    keys = [token.strip() for token in normalized.split(",") if token.strip()]
    # Deduplicate while preserving order
    seen = set()
    unique_keys: List[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique_keys.append(key)
    return unique_keys


def parse_groq_keys(api_key: str = "", api_keys_csv: str = "") -> List[str]:
    """Parse key inputs from env values or direct args."""
    values: List[str] = []
    if api_keys_csv.strip():
        values.extend(_split_keys(api_keys_csv))
    if api_key.strip() and api_key.strip() not in values:
        values.append(api_key.strip())
    return values


class GroqKeyPool:
    """Usage-aware key pool with cooldowns and persistent stats."""

    def __init__(
        self,
        api_keys: List[str],
        base_url: str,
        state_file: Path,
        cooldown_base_seconds: int = 30,
    ) -> None:
        self._base_url = base_url
        self._state_file = state_file
        self._cooldown_base_seconds = cooldown_base_seconds
        self._keys_by_id: Dict[str, str] = {}
        self._clients_by_id: Dict[str, Groq] = {}
        self._stats: Dict[str, Dict[str, Any]] = {}

        for key in api_keys:
            key_id = self._fingerprint(key)
            self._keys_by_id[key_id] = key
            self._clients_by_id[key_id] = Groq(api_key=key, base_url=base_url)

        self._load_state()
        self._ensure_stat_entries()

    @staticmethod
    def _fingerprint(key: str) -> str:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        return f"k_{digest}"

    def _default_stats(self) -> Dict[str, Any]:
        return {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "consecutive_failures": 0,
            "last_used_at": 0.0,
            "last_success_at": 0.0,
            "cooldown_until": 0.0,
            "last_error": "",
        }

    def _load_state(self) -> None:
        if not self._state_file.exists():
            self._stats = {}
            return
        try:
            with open(self._state_file, "r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                self._stats = data
            else:
                self._stats = {}
        except Exception:
            self._stats = {}

    def _save_state(self) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_file, "w", encoding="utf-8") as file:
            json.dump(self._stats, file, indent=2)

    def _ensure_stat_entries(self) -> None:
        key_ids = set(self._keys_by_id.keys())
        # Remove stale keys from state
        self._stats = {key_id: stats for key_id, stats in self._stats.items() if key_id in key_ids}
        # Add missing keys
        for key_id in key_ids:
            if key_id not in self._stats:
                self._stats[key_id] = self._default_stats()
        self._save_state()

    def _now(self) -> float:
        return time.time()

    def _is_cooling(self, key_id: str, now: float) -> bool:
        return float(self._stats[key_id].get("cooldown_until", 0.0)) > now

    def _selection_tuple(self, key_id: str, now: float) -> Tuple[int, float, int, int, float]:
        stats = self._stats[key_id]
        cooling_rank = 1 if self._is_cooling(key_id, now) else 0
        cooldown_until = float(stats.get("cooldown_until", 0.0))
        consecutive_failures = int(stats.get("consecutive_failures", 0))
        requests = int(stats.get("requests", 0))
        last_used_at = float(stats.get("last_used_at", 0.0))
        return (cooling_rank, cooldown_until, consecutive_failures, requests, last_used_at)

    def acquire_key(self) -> Optional[str]:
        if not self._keys_by_id:
            return None
        now = self._now()
        ordered = sorted(self._keys_by_id.keys(), key=lambda key_id: self._selection_tuple(key_id, now))
        return ordered[0] if ordered else None

    def get_client(self, key_id: str) -> Groq:
        return self._clients_by_id[key_id]

    def mark_request(self, key_id: str) -> None:
        stats = self._stats[key_id]
        stats["requests"] = int(stats.get("requests", 0)) + 1
        stats["last_used_at"] = self._now()
        self._save_state()

    def mark_success(self, key_id: str) -> None:
        stats = self._stats[key_id]
        stats["successes"] = int(stats.get("successes", 0)) + 1
        stats["consecutive_failures"] = 0
        stats["last_success_at"] = self._now()
        stats["last_error"] = ""
        stats["cooldown_until"] = 0.0
        self._save_state()

    def mark_failure(self, key_id: str, error_text: str) -> None:
        stats = self._stats[key_id]
        stats["failures"] = int(stats.get("failures", 0)) + 1
        stats["consecutive_failures"] = int(stats.get("consecutive_failures", 0)) + 1
        stats["last_error"] = error_text[:300]

        error_lower = error_text.lower()
        should_cooldown = any(token in error_lower for token in [
            "rate limit",
            "429",
            "quota",
            "temporarily unavailable",
            "timeout",
            "connection",
            "auth",
            "unauthorized",
            "forbidden",
        ])

        if should_cooldown:
            multiplier = min(int(stats["consecutive_failures"]), 5)
            cooldown_seconds = self._cooldown_base_seconds * (2 ** (multiplier - 1))
            stats["cooldown_until"] = self._now() + cooldown_seconds

        self._save_state()

    def snapshot(self) -> Dict[str, Any]:
        now = self._now()
        redacted = {}
        for key_id, stats in self._stats.items():
            redacted[key_id] = {
                "requests": int(stats.get("requests", 0)),
                "successes": int(stats.get("successes", 0)),
                "failures": int(stats.get("failures", 0)),
                "consecutive_failures": int(stats.get("consecutive_failures", 0)),
                "cooling": float(stats.get("cooldown_until", 0.0)) > now,
                "last_error": stats.get("last_error", ""),
            }
        return {
            "total_keys": len(self._keys_by_id),
            "keys": redacted,
        }
