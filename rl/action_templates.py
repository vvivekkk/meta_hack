from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from models import (
    AESeverity,
    AdverseEventTriageAction,
    CausalityAssessment,
    DeviationType,
    ProtocolDeviationAction,
    ReportingTimeline,
    SafetyNarrativeAction,
    TaskID,
    TriageAction,
)

_AE_SEVERITIES: List[AESeverity] = [
    AESeverity.MILD,
    AESeverity.MODERATE,
    AESeverity.SEVERE,
    AESeverity.LIFE_THREATENING,
    AESeverity.FATAL,
]

_AE_TIMELINES: List[ReportingTimeline] = [
    ReportingTimeline.ROUTINE,
    ReportingTimeline.FIFTEEN_DAY,
    ReportingTimeline.SEVEN_DAY,
]

_AE_CODEBOOK: List[Dict[str, str]] = [
    {"soc": "Cardiac disorders", "pt": "Myocardial infarction"},
    {"soc": "Nervous system disorders", "pt": "Encephalopathy"},
    {"soc": "Gastrointestinal disorders", "pt": "Nausea"},
    {"soc": "Hepatobiliary disorders", "pt": "Drug-induced liver injury"},
    {"soc": "Immune system disorders", "pt": "Anaphylactic reaction"},
    {"soc": "Skin and subcutaneous tissue disorders", "pt": "Rash"},
    {"soc": "Respiratory disorders", "pt": "Dyspnea"},
    {"soc": "General disorders", "pt": "Fatigue"},
    {"soc": "Infections and infestations", "pt": "Sepsis"},
    {"soc": "Blood and lymphatic system disorders", "pt": "Neutropenia"},
]

_DEV_TYPES: List[DeviationType] = [
    DeviationType.MINOR,
    DeviationType.PROTOCOL_AMENDMENT,
    DeviationType.MAJOR,
]

_RISK_BUCKETS: List[float] = [1.5, 2.5, 3.5, 5.0, 6.5, 7.5, 8.5, 9.5]

_FINDING_SELECT_MODES = 5

_NARRATIVE_STYLES = [
    "regulatory_summary",
    "temporal_first",
    "causality_first",
    "compliance_first",
]

_CAUSALITY_PLAN: List[CausalityAssessment] = [
    CausalityAssessment.DEFINITELY_RELATED,
    CausalityAssessment.PROBABLY_RELATED,
    CausalityAssessment.POSSIBLY_RELATED,
    CausalityAssessment.DEFINITELY_RELATED,
    CausalityAssessment.UNLIKELY_RELATED,
    CausalityAssessment.NOT_RELATED,
    CausalityAssessment.UNASSESSABLE,
]


def _clip_index(value: int, size: int) -> int:
    if size <= 0:
        return 0
    return max(0, min(size - 1, int(value)))


def action_space_nvec() -> List[int]:
    """
    MultiDiscrete action dimensions used by the PPO policy.

    Layout:
    0 severity index
    1 reporting timeline index
    2 seriousness flag
    3 MedDRA codebook index
    4 deviation type index
    5 CAPA flag
    6 risk bucket index
    7 finding selection mode
    8 narrative style index
    9 causality index
    10 dechallenge tri-state (None/False/True)
    11 rechallenge tri-state (None/False/True)
    12 temporal flag strategy
    """
    return [
        len(_AE_SEVERITIES),
        len(_AE_TIMELINES),
        2,
        len(_AE_CODEBOOK),
        len(_DEV_TYPES),
        2,
        len(_RISK_BUCKETS),
        _FINDING_SELECT_MODES,
        len(_NARRATIVE_STYLES),
        len(_CAUSALITY_PLAN),
        3,
        3,
        4,
    ]


def _coerce_action_vector(action_vector: Sequence[int] | int) -> List[int]:
    nvec = action_space_nvec()
    if isinstance(action_vector, int):
        seed = max(0, int(action_vector))
        values = []
        for i, base in enumerate(nvec):
            values.append((seed + (i * 3)) % base)
            seed = seed // base if seed > 0 else 0
        return values

    raw = [int(x) for x in action_vector]
    if len(raw) < len(nvec):
        raw.extend([0] * (len(nvec) - len(raw)))
    return [_clip_index(raw[i], nvec[i]) for i in range(len(nvec))]


def _select_flagged_ids(findings: Iterable[Dict[str, Any]], mode: int) -> List[str]:
    finding_list = [f for f in findings if isinstance(f, dict) and f.get("id")]
    if mode == 0:
        return []
    if mode == 1:
        return [str(f["id"]) for f in finding_list]
    if mode == 2:
        high_risk_tokens = {
            "eligibility",
            "blinding",
            "sae",
            "integrity",
            "consent",
            "endpoint",
            "accountability",
        }
        selected = []
        for finding in finding_list:
            text = f"{finding.get('category', '')} {finding.get('description', '')}".lower()
            if any(token in text for token in high_risk_tokens):
                selected.append(str(finding["id"]))
        return selected
    if mode == 3:
        return [str(f["id"]) for i, f in enumerate(finding_list) if i % 2 == 0]
    return [str(f["id"]) for f in finding_list[: max(1, len(finding_list) // 2)]]


def _tri_bool(value: int) -> bool | None:
    if value == 0:
        return None
    if value == 1:
        return False
    return True


def _temporal_flags(observation: Dict[str, Any], mode: int) -> List[str]:
    adverse_event = observation.get("adverse_event", {})
    onset_date = adverse_event.get("onset_date", "unknown")
    report_date = adverse_event.get("report_date", "unknown")
    lab_values = observation.get("lab_values_timeline", [])

    if mode == 0:
        return [
            f"event onset recorded at {onset_date}",
            "temporal association assessed against treatment exposure",
        ]
    if mode == 1:
        return [
            f"onset on {onset_date}",
            f"initial safety report on {report_date}",
            "sequence supports a plausible exposure-event relationship",
        ]
    if mode == 2:
        if lab_values:
            first_date = str(lab_values[0].get("date", "unknown"))
            last_date = str(lab_values[-1].get("date", "unknown"))
            return [
                f"lab trend monitored from {first_date} to {last_date}",
                "event occurred after notable laboratory trajectory change",
            ]
        return ["lab trend context unavailable", "onset chronology still documented"]
    return [
        f"onset documented at {onset_date}",
        "exposure chronology reviewed with dechallenge/rechallenge context",
        "timeline supports regulatory causality assessment",
    ]


def _build_narrative(
    observation: Dict[str, Any],
    causality: CausalityAssessment,
    style_idx: int,
    dechallenge: bool | None,
    rechallenge: bool | None,
) -> str:
    case_id = observation.get("case_id", "unknown")
    study_drug = observation.get("study_drug", "investigational product")
    demographics = observation.get("patient_demographics", {})
    adverse_event = observation.get("adverse_event", {})

    age = demographics.get("age", "unknown")
    sex = demographics.get("sex", "unknown")
    ae_term = adverse_event.get("term", "adverse event")
    onset = adverse_event.get("onset_date", "unknown onset")
    action_taken = observation.get("action_taken", "study drug management not specified")
    outcome = observation.get("outcome_at_last_followup", "outcome pending")

    style = _NARRATIVE_STYLES[_clip_index(style_idx, len(_NARRATIVE_STYLES))]

    if style == "temporal_first":
        text = (
            f"Chronology: {case_id} developed {ae_term} on {onset} after exposure to {study_drug}. "
            f"Patient profile: {age}-year-old {sex}. "
            f"Action taken: {action_taken}. Outcome: {outcome}. "
            f"Causality is assessed as {causality.value.replace('_', ' ')} with consideration of temporal sequence, "
            f"dechallenge={dechallenge}, rechallenge={rechallenge}."
        )
    elif style == "causality_first":
        text = (
            f"Causality assessment: {causality.value.replace('_', ' ')} for event {ae_term} in case {case_id}. "
            f"Subject: {age}-year-old {sex} receiving {study_drug}. "
            f"Onset occurred on {onset}. Management included {action_taken}. "
            f"Current outcome: {outcome}. Dechallenge={dechallenge}; rechallenge={rechallenge}."
        )
    elif style == "compliance_first":
        text = (
            f"Regulatory narrative for case {case_id}: subject ({age}-year-old {sex}) exposed to {study_drug} "
            f"experienced {ae_term} with onset on {onset}. "
            f"Clinical action taken: {action_taken}. Outcome at follow-up: {outcome}. "
            f"Causality: {causality.value.replace('_', ' ')}."
        )
    else:
        text = (
            f"Case {case_id} involves a {age}-year-old {sex} participant receiving {study_drug}. "
            f"The subject developed {ae_term} with onset on {onset}. "
            f"Action taken included: {action_taken}. Outcome at last follow-up: {outcome}. "
            f"Overall causality is assessed as {causality.value.replace('_', ' ')} with dechallenge={dechallenge} and rechallenge={rechallenge}."
        )

    compliance_tail = (
        " Narrative includes demographics, event chronology, management, outcome, "
        "and structured causality statement for ICH E2B readiness."
    )
    if len(text) < 120:
        text += compliance_tail
    return text


def action_from_vector(
    task_id: str,
    action_vector: Sequence[int] | int,
    observation_payload: Dict[str, Any],
) -> TriageAction:
    values = _coerce_action_vector(action_vector)

    severity_idx = values[0]
    timeline_idx = values[1]
    serious_flag = bool(values[2])
    codebook_idx = values[3]
    dev_type_idx = values[4]
    capa_flag = bool(values[5])
    risk_idx = values[6]
    finding_mode = values[7]
    narrative_style_idx = values[8]
    causality_idx = values[9]
    dechallenge_idx = values[10]
    rechallenge_idx = values[11]
    temporal_mode = values[12]

    task_value = task_id.value if hasattr(task_id, "value") else str(task_id)

    if task_value == TaskID.ADVERSE_EVENT_TRIAGE.value:
        coded = _AE_CODEBOOK[codebook_idx]
        severity = _AE_SEVERITIES[severity_idx]
        timeline = _AE_TIMELINES[timeline_idx]
        return TriageAction(
            task_id=TaskID.ADVERSE_EVENT_TRIAGE,
            ae_triage=AdverseEventTriageAction(
                severity_classification=severity,
                reporting_timeline=timeline,
                meddra_soc=coded["soc"],
                meddra_preferred_term=coded["pt"],
                is_serious=serious_flag,
                rationale=(
                    "Parameterized RL policy selected severity="
                    f"{severity.value}, timeline={timeline.value}, coded_term={coded['pt']}."
                ),
            ),
        )

    if task_value == TaskID.PROTOCOL_DEVIATION_AUDIT.value:
        findings = observation_payload.get("findings", [])
        deviation_type = _DEV_TYPES[dev_type_idx]
        flagged_ids = _select_flagged_ids(findings, finding_mode)
        if deviation_type == DeviationType.MINOR:
            flagged_ids = []

        risk_score = _RISK_BUCKETS[risk_idx]
        if deviation_type == DeviationType.MAJOR:
            risk_score = max(risk_score, 6.0)
        if deviation_type == DeviationType.MINOR:
            risk_score = min(risk_score, 4.5)

        if deviation_type == DeviationType.MAJOR:
            action_text = "Immediate CAPA and sponsor escalation with focused re-monitoring."
        elif deviation_type == DeviationType.PROTOCOL_AMENDMENT:
            action_text = "Manage under controlled amendment workflow and verify retraining."
        else:
            action_text = "Document as minor deviation and track in routine monitoring log."

        if capa_flag:
            action_text += " CAPA effectiveness check required in 30 days."

        return TriageAction(
            task_id=TaskID.PROTOCOL_DEVIATION_AUDIT,
            deviation_audit=ProtocolDeviationAction(
                deviation_type=deviation_type,
                capa_required=capa_flag,
                site_risk_score=float(risk_score),
                flagged_finding_ids=flagged_ids,
                recommended_action=action_text[:300],
            ),
        )

    causality = _CAUSALITY_PLAN[causality_idx]
    dechallenge = _tri_bool(dechallenge_idx)
    rechallenge = _tri_bool(rechallenge_idx)
    narrative = _build_narrative(
        observation_payload,
        causality,
        narrative_style_idx,
        dechallenge,
        rechallenge,
    )
    return TriageAction(
        task_id=TaskID.SAFETY_NARRATIVE_GENERATION,
        safety_narrative=SafetyNarrativeAction(
            narrative_text=narrative,
            causality_assessment=causality,
            key_temporal_flags=_temporal_flags(observation_payload, temporal_mode),
            dechallenge_positive=dechallenge,
            rechallenge_positive=rechallenge,
        ),
    )


def action_space_size() -> int:
    # Kept for compatibility with older callers that still expect a scalar size.
    return int(sum(action_space_nvec()))
