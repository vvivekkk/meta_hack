"""
Clinical Trial Triage — Typed Models
=====================================
Pydantic models for Actions, Observations, Rewards, and State.
All models are fully typed and OpenEnv-spec compliant.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────

class AESeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    LIFE_THREATENING = "life_threatening"
    FATAL = "fatal"


class ReportingTimeline(str, Enum):
    SEVEN_DAY = "7-day"       # SAE unexpected fatal/life-threatening
    FIFTEEN_DAY = "15-day"    # SUSAR (Suspected Unexpected Serious Adverse Reaction)
    ROUTINE = "routine"       # Annual safety report


class DeviationType(str, Enum):
    MAJOR = "major"           # Affects subject safety or data integrity
    MINOR = "minor"           # Administrative, no subject safety impact
    PROTOCOL_AMENDMENT = "protocol_amendment"


class CausalityAssessment(str, Enum):
    DEFINITELY_RELATED = "definitely_related"
    PROBABLY_RELATED = "probably_related"
    POSSIBLY_RELATED = "possibly_related"
    UNLIKELY_RELATED = "unlikely_related"
    NOT_RELATED = "not_related"
    UNASSESSABLE = "unassessable"


class TaskID(str, Enum):
    ADVERSE_EVENT_TRIAGE = "adverse_event_triage"
    PROTOCOL_DEVIATION_AUDIT = "protocol_deviation_audit"
    SAFETY_NARRATIVE_GENERATION = "safety_narrative_generation"


# ─────────────────────────────────────────
# ACTIONS
# ─────────────────────────────────────────

class AdverseEventTriageAction(BaseModel):
    """Action for Task 1: Adverse Event Triage."""

    severity_classification: AESeverity = Field(
        ...,
        description="Agent's severity classification of the adverse event.",
    )
    reporting_timeline: ReportingTimeline = Field(
        ...,
        description="Required regulatory reporting timeline.",
    )
    meddra_soc: str = Field(
        ...,
        description="MedDRA System Organ Class (e.g., 'Cardiac disorders').",
        max_length=120,
    )
    meddra_preferred_term: str = Field(
        ...,
        description="MedDRA Preferred Term (e.g., 'Myocardial infarction').",
        max_length=120,
    )
    is_serious: bool = Field(
        ...,
        description="Whether this qualifies as a Serious Adverse Event (SAE).",
    )
    rationale: str = Field(
        ...,
        description="Agent's reasoning (max 500 chars).",
        max_length=500,
    )


class ProtocolDeviationAction(BaseModel):
    """Action for Task 2: Protocol Deviation Audit."""

    deviation_type: DeviationType = Field(
        ...,
        description="Classification of each deviation found.",
    )
    capa_required: bool = Field(
        ...,
        description="Whether a Corrective and Preventive Action plan is required.",
    )
    site_risk_score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Risk score for the site (0=low, 10=critical).",
    )
    flagged_finding_ids: List[str] = Field(
        default_factory=list,
        description="List of finding IDs the agent considers GCP violations.",
    )
    recommended_action: str = Field(
        ...,
        description="Agent's recommended next step (e.g., 'Immediate re-monitoring').",
        max_length=300,
    )


class SafetyNarrativeAction(BaseModel):
    """Action for Task 3: Safety Narrative Generation."""

    narrative_text: str = Field(
        ...,
        description="Full ICH E2B-compliant ICSR safety narrative.",
        min_length=100,
        max_length=4000,
    )
    causality_assessment: CausalityAssessment = Field(
        ...,
        description="Causality assessment for the primary suspect drug.",
    )
    key_temporal_flags: List[str] = Field(
        default_factory=list,
        description="Temporal markers identified (e.g., 'onset 3 days after dose increase').",
    )
    dechallenge_positive: Optional[bool] = Field(
        None,
        description="Whether the AE resolved on drug discontinuation (None if unknown).",
    )
    rechallenge_positive: Optional[bool] = Field(
        None,
        description="Whether the AE recurred on re-administration (None if not done).",
    )


# Union action type — the agent sends one of these per step
class TriageAction(BaseModel):
    """Top-level Action model wrapping task-specific actions."""

    task_id: TaskID = Field(..., description="Which task this action targets.")
    ae_triage: Optional[AdverseEventTriageAction] = Field(
        None, description="Populated for adverse_event_triage task."
    )
    deviation_audit: Optional[ProtocolDeviationAction] = Field(
        None, description="Populated for protocol_deviation_audit task."
    )
    safety_narrative: Optional[SafetyNarrativeAction] = Field(
        None, description="Populated for safety_narrative_generation task."
    )

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────
# OBSERVATIONS
# ─────────────────────────────────────────

class AdverseEventObservation(BaseModel):
    """Observation returned for AE Triage task."""

    case_id: str
    narrative: str = Field(..., description="Raw AE narrative from site.")
    patient_age: int
    patient_sex: str
    study_drug: str
    dose_mg: float
    days_on_drug: int
    relevant_medical_history: List[str]
    concomitant_medications: List[str]
    lab_values: Dict[str, Any]
    ae_onset_date: str
    ae_description: str
    outcome: str
    step_count: int
    max_steps: int
    scoring_hints: Optional[Dict[str, Any]] = None


class ProtocolDeviationObservation(BaseModel):
    """Observation returned for Protocol Deviation Audit task."""

    site_id: str
    site_name: str
    visit_type: str
    findings: List[Dict[str, Any]]
    prior_deviations: int
    active_subjects: int
    study_phase: str
    last_monitoring_visit: str
    step_count: int
    max_steps: int


class SafetyNarrativeObservation(BaseModel):
    """Observation returned for Safety Narrative Generation task."""

    case_id: str
    patient_demographics: Dict[str, Any]
    study_drug: str
    suspect_drugs: List[str]
    concomitant_medications: List[Dict[str, Any]]
    adverse_event: Dict[str, Any]
    lab_values_timeline: List[Dict[str, Any]]
    medical_history: List[str]
    action_taken: str
    outcome_at_last_followup: str
    reference_documents: List[str]
    step_count: int
    max_steps: int


class TriageObservation(BaseModel):
    """Top-level Observation returned from step() / reset()."""

    task_id: TaskID
    ae_observation: Optional[AdverseEventObservation] = None
    deviation_observation: Optional[ProtocolDeviationObservation] = None
    narrative_observation: Optional[SafetyNarrativeObservation] = None
    message: str = ""

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────
# REWARD
# ─────────────────────────────────────────

class TriageReward(BaseModel):
    """
    Structured reward with partial credit signals.
    All sub-scores normalized to [0, 1].
    """

    total: float = Field(..., ge=0.0, le=1.0, description="Weighted total reward.")

    # Task-1 sub-scores
    severity_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    timeline_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    soc_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    pt_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Task-2 sub-scores
    deviation_type_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    capa_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    risk_score_proximity: Optional[float] = Field(None, ge=0.0, le=1.0)
    violation_recall: Optional[float] = Field(None, ge=0.0, le=1.0)
    violation_precision: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Task-3 sub-scores
    temporal_coverage: Optional[float] = Field(None, ge=0.0, le=1.0)
    causality_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    narrative_completeness: Optional[float] = Field(None, ge=0.0, le=1.0)
    regulatory_compliance: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Penalty flags
    penalty_applied: bool = False
    penalty_reason: Optional[str] = None


# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────

class TriageState(BaseModel):
    """Episode state metadata returned from state()."""

    episode_id: str
    task_id: TaskID
    step_count: int
    max_steps: int
    done: bool
    cumulative_reward: float
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    current_case_id: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None

    model_config = ConfigDict(use_enum_values=True)


class StepResult(BaseModel):
    """Result returned from step()."""

    observation: TriageObservation
    reward: float
    reward_detail: TriageReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)