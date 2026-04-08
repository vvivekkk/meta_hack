"""
Generate a compact quality report for the clinical case bank.

Usage:
    python scripts/data_quality_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.case_bank import AE_CASES, DEVIATION_CASES, NARRATIVE_CASES


REQUIRED_KEYS = {
    "ae": [
        "case_id",
        "narrative",
        "ground_truth",
        "patient_age",
        "study_drug",
        "lab_values",
    ],
    "deviation": [
        "site_id",
        "site_name",
        "findings",
        "ground_truth",
    ],
    "narrative": [
        "case_id",
        "patient_demographics",
        "adverse_event",
        "ground_truth",
    ],
}


def _missing_keys(case: Dict[str, Any], required_keys: List[str]) -> List[str]:
    return [key for key in required_keys if key not in case or case[key] in (None, "", [])]


def _analyze_cases(cases: List[Dict[str, Any]], required: List[str], label: str) -> Dict[str, Any]:
    missing = []
    for case in cases:
        gaps = _missing_keys(case, required)
        if gaps:
            missing.append({"id": case.get("case_id") or case.get("site_id"), "missing": gaps})

    return {
        "task": label,
        "count": len(cases),
        "missing_case_count": len(missing),
        "missing_examples": missing[:5],
    }


def main() -> None:
    report = {
        "summary": {
            "ae_cases": len(AE_CASES),
            "deviation_cases": len(DEVIATION_CASES),
            "narrative_cases": len(NARRATIVE_CASES),
            "total_cases": len(AE_CASES) + len(DEVIATION_CASES) + len(NARRATIVE_CASES),
        },
        "quality": [
            _analyze_cases(AE_CASES, REQUIRED_KEYS["ae"], "adverse_event_triage"),
            _analyze_cases(DEVIATION_CASES, REQUIRED_KEYS["deviation"], "protocol_deviation_audit"),
            _analyze_cases(NARRATIVE_CASES, REQUIRED_KEYS["narrative"], "safety_narrative_generation"),
        ],
    }

    output_path = ROOT / "outputs" / "data_quality_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
