"""
Clinical Trial Triage — Task Graders
=====================================
Deterministic, reproducible graders for all 3 tasks.
Each grader returns a TriageReward with sub-component scores.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from models import (
    AdverseEventTriageAction,
    ProtocolDeviationAction,
    SafetyNarrativeAction,
    TriageReward,
)


# ─────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────

def _normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _fuzzy_match(candidate: str, accepted: List[str]) -> bool:
    """Case-insensitive substring match against accepted values."""
    cand = _normalize(candidate)
    for acc in accepted:
        acc_n = _normalize(acc)
        if cand == acc_n or acc_n in cand or cand in acc_n:
            return True
    return False


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ─────────────────────────────────────────
# TASK 1 GRADER: ADVERSE EVENT TRIAGE
# ─────────────────────────────────────────

def grade_ae_triage(
    action: AdverseEventTriageAction,
    case: Dict[str, Any],
) -> TriageReward:
    """
    Grades adverse event triage action against ground truth.

    Scoring breakdown:
    - severity_accuracy:  0.30 — correct CTCAE severity level
    - timeline_accuracy:  0.25 — correct expedited reporting timeline
    - is_serious:         0.10 — correct SAE flag
    - soc_accuracy:       0.175 — MedDRA SOC (fuzzy match)
    - pt_accuracy:        0.175 — MedDRA PT (fuzzy match)

    Partial credit for adjacent severity (1 level off = 0.5 score).
    """
    gt = case["ground_truth"]
    weights = case["scoring_weights"]

    severity_order = ["mild", "moderate", "severe", "life_threatening", "fatal"]

    # 1. Severity — partial credit for adjacent
    agent_sev = _normalize(action.severity_classification)
    gt_sev = _normalize(gt["severity"])
    if agent_sev == gt_sev:
        sev_score = 1.0
    else:
        try:
            agent_idx = severity_order.index(agent_sev)
            gt_idx = severity_order.index(gt_sev)
            diff = abs(agent_idx - gt_idx)
            sev_score = max(0.0, 1.0 - diff * 0.5)
        except ValueError:
            sev_score = 0.0

    # 2. Reporting timeline — binary (safety-critical, no partial)
    timeline_score = 1.0 if _normalize(action.reporting_timeline) == _normalize(gt["reporting_timeline"]) else 0.0

    # 3. Is serious — binary
    is_serious_score = 1.0 if action.is_serious == gt["is_serious"] else 0.0

    # 4. SOC — fuzzy
    soc_score = 1.0 if _fuzzy_match(action.meddra_soc, case["accepted_soc_variations"]) else 0.0

    # 5. PT — fuzzy
    pt_score = 1.0 if _fuzzy_match(action.meddra_preferred_term, case["accepted_pt_variations"]) else 0.0

    total = (
        sev_score * weights["severity"]
        + timeline_score * weights["timeline"]
        + is_serious_score * weights["is_serious"]
        + soc_score * weights["soc"]
        + pt_score * weights["pt"]
    )

    return TriageReward(
        total=_clamp(total),
        severity_accuracy=sev_score,
        timeline_accuracy=timeline_score,
        soc_accuracy=soc_score,
        pt_accuracy=pt_score,
    )


# ─────────────────────────────────────────
# TASK 2 GRADER: PROTOCOL DEVIATION AUDIT
# ─────────────────────────────────────────

def grade_protocol_deviation(
    action: ProtocolDeviationAction,
    case: Dict[str, Any],
) -> TriageReward:
    """
    Grades protocol deviation audit.

    Scoring breakdown:
    - deviation_type:     0.20
    - capa_required:      0.20
    - risk_score:         0.20 — proximity-based (within ±1 = full, ±2 = 0.5, etc.)
    - violation_recall:   0.20 — fraction of true GCP violations correctly flagged
    - violation_precision: 0.20 — fraction of flagged items that are true violations
    """
    gt = case["ground_truth"]

    # 1. Deviation type
    dev_score = 1.0 if _normalize(action.deviation_type) == _normalize(gt["deviation_type"]) else 0.0

    # 2. CAPA required
    capa_score = 1.0 if action.capa_required == gt["capa_required"] else 0.0

    # 3. Risk score proximity — gaussian-like decay
    gt_risk = gt["site_risk_score"]
    diff = abs(action.site_risk_score - gt_risk)
    risk_score = _clamp(1.0 - (diff / 3.0))  # ±3 points = 0, ±0 = 1.0

    # 4. Recall: fraction of true violations the agent flagged
    true_violations = set(gt["gcp_violation_ids"])
    flagged = set(action.flagged_finding_ids)

    if len(true_violations) == 0:
        recall = 1.0 if len(flagged) == 0 else 0.5  # penalize false positives when there are none
        precision = 1.0 if len(flagged) == 0 else 0.0
    else:
        tp = len(true_violations & flagged)
        recall = tp / len(true_violations)
        precision = tp / len(flagged) if flagged else 0.0

    total = (
        dev_score * 0.20
        + capa_score * 0.20
        + risk_score * 0.20
        + recall * 0.20
        + precision * 0.20
    )

    return TriageReward(
        total=_clamp(total),
        deviation_type_accuracy=dev_score,
        capa_accuracy=capa_score,
        risk_score_proximity=risk_score,
        violation_recall=recall,
        violation_precision=precision,
    )


# ─────────────────────────────────────────
# TASK 3 GRADER: SAFETY NARRATIVE
# ─────────────────────────────────────────

_MANDATORY_NARRATIVE_KEYWORDS = {
    "demographics": ["year", "yo", "years old", "female", "male"],
    "study_drug": ["zl-550", "compound zl-550", "150mg", "150 mg"],
    "concomitant": ["warfarin", "atorvastatin", "metoprolol"],
    "ae_description": ["gastrointestinal", "haemorrhage", "hemorrhage", "bleeding", "ulcer"],
    "lab_values": ["inr", "hemoglobin", "hgb", "haemoglobin"],
    "temporal": ["onset", "day", "after", "prior", "before", "weeks", "days"],
    "action_taken": ["discontinued", "suspended", "transfus", "held", "stopped"],
    "outcome": ["recovered", "resolv", "sequelae", "improved", "discharged"],
    "causality": ["related", "probable", "possible", "likely", "causal", "causality"],
}

_SERIOUSNESS_KEYWORDS = [
    "hospitali", "serious", "seriousness", "medically significant",
    "transfus", "blood", "grade 3", "grade3",
]

_REGULATORY_FLAGS = {
    "INR_mentioned": ["inr"],
    "warfarin_interaction_noted": ["warfarin", "anticoagul", "interaction"],
    "seriousness_criteria_stated": _SERIOUSNESS_KEYWORDS,
    "dechallenge_documented": ["dechallenge", "after discontinu", "resolved after"],
    "causality_assessment_provided": ["causal", "causality", "related", "probably related", "assessment"],
}

_CAUSALITY_MAPPING = {
    "definitely_related": ["definitely", "certain", "confirmed"],
    "probably_related": ["probabl", "likely", "almost certain"],
    "possibly_related": ["possibl", "uncertain", "could be"],
    "unlikely_related": ["unlikely", "doubtful"],
    "not_related": ["not related", "unrelated", "coincident"],
    "unassessable": ["unassessable", "cannot assess", "insufficient"],
}


def _tokenize_text(value: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", value.lower()) if len(tok) >= 3]


def _extract_case_section_keywords(case: Dict[str, Any]) -> Dict[str, List[str]]:
    study_drug = str(case.get("study_drug", ""))
    suspect_drugs = [str(x) for x in case.get("suspect_drugs", [])]
    concomitant = case.get("concomitant_medications", [])
    adverse_event = case.get("adverse_event", {})
    medical_history = case.get("medical_history", [])
    timeline = case.get("lab_values_timeline", [])

    med_names: List[str] = []
    for med in concomitant:
        if isinstance(med, dict):
            med_names.append(str(med.get("name", "")))
        else:
            med_names.append(str(med))

    lab_markers: List[str] = []
    for row in timeline:
        if not isinstance(row, dict):
            continue
        for key in row.keys():
            if key.lower() != "date":
                lab_markers.extend(_tokenize_text(str(key)))

    adverse_tokens = _tokenize_text(str(adverse_event.get("term", "")))
    adverse_tokens.extend(_tokenize_text(str(adverse_event.get("meddra_pt", ""))))

    study_drug_tokens = _tokenize_text(study_drug)
    for suspect in suspect_drugs:
        study_drug_tokens.extend(_tokenize_text(suspect))

    history_tokens: List[str] = []
    for item in medical_history:
        history_tokens.extend(_tokenize_text(str(item)))

    medication_tokens: List[str] = []
    for name in med_names:
        medication_tokens.extend(_tokenize_text(name))

    return {
        "study_drug": sorted(set(study_drug_tokens)),
        "concomitant": sorted(set(medication_tokens)),
        "ae_description": sorted(set(adverse_tokens)),
        "lab_values": sorted(set(lab_markers)),
        "history": sorted(set(history_tokens)),
    }


def _flag_keywords(flag_name: str, case: Dict[str, Any]) -> List[str]:
    explicit_map = {
        "INR_mentioned": ["inr"],
        "warfarin_interaction_noted": ["warfarin", "anticoagul", "interaction"],
        "seriousness_criteria_stated": _SERIOUSNESS_KEYWORDS,
        "dechallenge_documented": [
            "dechallenge",
            "after discontinu",
            "resolved after",
            "improved after",
            "drug discontinued",
            "drug stopped",
        ],
        "causality_assessment_provided": ["causal", "causality", "related", "assessment"],
        "lipase_trend_documented": ["lipase", "trend", "elevat", "down-trending"],
        "suspect_drug_named": [str(case.get("study_drug", "")).lower()],
    }

    if flag_name in explicit_map:
        return [kw for kw in explicit_map[flag_name] if kw]

    # Fallback for unknown flags: tokenize flag name and require those concepts.
    return [tok for tok in flag_name.lower().split("_") if len(tok) >= 3]


def grade_safety_narrative(
    action: SafetyNarrativeAction,
    case: Dict[str, Any],
) -> TriageReward:
    """
    Grades a safety narrative for completeness, temporal coverage,
    causality accuracy, and regulatory compliance.

    Scoring breakdown:
    - narrative_completeness:  0.35 — mandatory section coverage (0-1 per section)
    - temporal_coverage:       0.25 — temporal elements identified
    - causality_accuracy:      0.20 — causality assessment correctness
    - regulatory_compliance:   0.20 — regulatory-specific flags present

    Penalty: -0.10 for clearly missing critical elements (SAE criteria, patient demographics).
    """
    gt = case["ground_truth"]
    narrative_lower = action.narrative_text.lower()
    section_keywords = _extract_case_section_keywords(case)

    mandatory_keywords = dict(_MANDATORY_NARRATIVE_KEYWORDS)
    mandatory_keywords["study_drug"] = sorted(
        set(mandatory_keywords["study_drug"] + section_keywords["study_drug"])
    )
    mandatory_keywords["concomitant"] = sorted(
        set(mandatory_keywords["concomitant"] + section_keywords["concomitant"])
    )
    mandatory_keywords["ae_description"] = sorted(
        set(mandatory_keywords["ae_description"] + section_keywords["ae_description"])
    )
    mandatory_keywords["lab_values"] = sorted(
        set(mandatory_keywords["lab_values"] + section_keywords["lab_values"])
    )

    # 1. Narrative completeness — count mandatory section coverage
    section_scores = []
    for section, keywords in mandatory_keywords.items():
        present = any(kw in narrative_lower for kw in keywords)
        section_scores.append(1.0 if present else 0.0)
    completeness_score = sum(section_scores) / len(section_scores)

    # 2. Temporal coverage
    required_temporal = gt.get("required_temporal_elements", [])
    temporal_hits = 0
    for req in required_temporal:
        req_kws = req.lower().split()
        # All keywords in the requirement must appear somewhere in narrative
        if all(kw in narrative_lower for kw in req_kws[:2]):  # match first 2 words minimum
            temporal_hits += 1

    # Agent's temporal flags also give credit
    agent_temporal_hits = 0
    for flag in action.key_temporal_flags:
        if any(
            kw in flag.lower()
            for kw in ["onset", "day", "after", "before", "prior", "date", "timeline", "report"]
        ):
            agent_temporal_hits += 1

    temporal_score = _clamp(
        (temporal_hits / max(len(required_temporal), 1)) * 0.7
        + min(agent_temporal_hits / 3.0, 1.0) * 0.3
    )

    # 3. Causality accuracy
    gt_causality = gt["causality"]  # "probably_related"
    agent_causality = _normalize(str(action.causality_assessment))
    gt_causality_n = _normalize(gt_causality)

    if agent_causality == gt_causality_n:
        causality_score = 1.0
    else:
        # Check narrative text for the right causality keywords
        gt_keywords = _CAUSALITY_MAPPING.get(gt_causality, [])
        if any(kw in narrative_lower for kw in gt_keywords):
            causality_score = 0.7  # right answer in narrative, wrong dropdown
        else:
            # Adjacent causality categories get partial credit
            causality_order = list(_CAUSALITY_MAPPING.keys())
            try:
                agent_idx = causality_order.index(agent_causality.replace(" ", "_"))
                gt_idx = causality_order.index(gt_causality)
                diff = abs(agent_idx - gt_idx)
                causality_score = max(0.0, 1.0 - diff * 0.3)
            except ValueError:
                causality_score = 0.1

    # 4. Regulatory compliance flags
    flag_scores = []
    required_flags = gt.get("regulatory_compliance_flags") or list(_REGULATORY_FLAGS.keys())
    for flag_name in required_flags:
        flag_keywords = _flag_keywords(str(flag_name), case)
        present = any(kw in narrative_lower for kw in flag_keywords)
        flag_scores.append(1.0 if present else 0.0)
    compliance_score = sum(flag_scores) / max(len(flag_scores), 1)

    # Penalty check: missing absolute essentials
    penalty_applied = False
    penalty_reason = None
    if completeness_score < 0.3:
        penalty_applied = True
        penalty_reason = "Narrative missing >70% of required sections — quality below minimum threshold"
    if len(action.narrative_text.strip()) < 150:
        penalty_applied = True
        penalty_reason = "Narrative too short to constitute a valid regulatory submission"

    existing_score = (
        completeness_score * 0.35
        + temporal_score * 0.25
        + causality_score * 0.20
        + compliance_score * 0.20
    )

    # Hard-task traps: temporal ordering, drug-name specificity, causality-text consistency.
    chronology_markers = ["on", "day", "after", "before", "prior", "timeline", "date", "follow-up"]
    chronology_hits = sum(1 for marker in chronology_markers if marker in narrative_lower)
    temporal_order_score = _clamp(chronology_hits / 5.0)

    study_drug = str(case.get("study_drug", "")).lower()
    suspect_drugs = [str(x).lower() for x in case.get("suspect_drugs", [])]
    drug_name_present = (bool(study_drug) and study_drug in narrative_lower) or any(
        drug and drug in narrative_lower for drug in suspect_drugs
    )

    causality_map = {
        "definitely_related": ["clearly caused", "definitively", "direct causal"],
        "probably_related": ["probably", "likely caused", "strong temporal"],
        "possibly_related": ["possible", "cannot exclude", "temporal association"],
        "unlikely_related": ["unlikely", "coincidental", "unrelated"],
        "not_related": ["not related", "no causal", "alternative cause"],
        "unassessable": ["cannot assess", "insufficient data", "unassessable"],
    }
    action_causality = str(getattr(action.causality_assessment, "value", action.causality_assessment))
    expected_phrases = causality_map.get(action_causality, [])
    causality_text_aligned = any(phrase in narrative_lower for phrase in expected_phrases)

    hard_task_bonus = (
        temporal_order_score * 0.30
        + (0.35 if drug_name_present else 0.0)
        + (0.35 if causality_text_aligned else 0.0)
    )

    total = existing_score * 0.7 + hard_task_bonus * 0.3

    if penalty_applied:
        total = total * 0.6  # 40% penalty

    return TriageReward(
        total=_clamp(total),
        narrative_completeness=completeness_score,
        temporal_coverage=temporal_score,
        causality_accuracy=causality_score,
        regulatory_compliance=compliance_score,
        penalty_applied=penalty_applied,
        penalty_reason=penalty_reason,
    )
