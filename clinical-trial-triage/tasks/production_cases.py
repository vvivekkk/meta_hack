"""
Expanded production-grade scenarios for realism and policy generalization.
"""
from __future__ import annotations

from typing import Any, Dict, List

EXTRA_AE_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "CASE-AE-004",
        "narrative": (
            "Subject 0204 (Female, 61 years) in Study CARDIO-612 receiving CL-901 "
            "experienced acute dyspnea, wheeze, urticaria, and hypotension 25 minutes "
            "after infusion start on Cycle 2 Day 1. BP dropped to 78/44 mmHg. "
            "Epinephrine and IV steroids administered with rapid stabilization. "
            "Subject observed overnight and discharged next day."
        ),
        "patient_age": 61,
        "patient_sex": "Female",
        "study_drug": "Compound CL-901",
        "dose_mg": 250.0,
        "days_on_drug": 29,
        "relevant_medical_history": [
            "Hypertension",
            "Seasonal allergic rhinitis",
        ],
        "concomitant_medications": ["Amlodipine 5mg QD", "Cetirizine 10mg PRN"],
        "lab_values": {
            "Lactate_mmol_L": 2.6,
            "Eosinophils_percent": 8.2,
            "Tryptase_ng_mL": 21.0,
        },
        "ae_onset_date": "2024-08-19",
        "ae_description": "Acute infusion-related anaphylactic reaction with hypotension",
        "outcome": "recovered",
        "ground_truth": {
            "severity": "life_threatening",
            "reporting_timeline": "7-day",
            "is_serious": True,
            "meddra_soc": "Immune system disorders",
            "meddra_preferred_term": "Anaphylactic reaction",
        },
        "scoring_weights": {
            "severity": 0.30,
            "timeline": 0.25,
            "is_serious": 0.10,
            "soc": 0.175,
            "pt": 0.175,
        },
        "accepted_soc_variations": ["immune system disorders"],
        "accepted_pt_variations": ["anaphylactic reaction", "anaphylaxis"],
    },
    {
        "case_id": "CASE-AE-005",
        "narrative": (
            "Subject 0312 (Male, 45 years) in Study HEPA-210 on HN-144 developed "
            "malaise and jaundice after 6 weeks. ALT 912 U/L (ULN 45), AST 744 U/L, "
            "bilirubin 3.4 mg/dL with INR 1.7. No viral hepatitis markers detected. "
            "Drug stopped immediately; hospitalized for intensive liver monitoring."
        ),
        "patient_age": 45,
        "patient_sex": "Male",
        "study_drug": "Compound HN-144",
        "dose_mg": 120.0,
        "days_on_drug": 42,
        "relevant_medical_history": ["Nonalcoholic fatty liver disease"],
        "concomitant_medications": ["Vitamin D"],
        "lab_values": {
            "ALT_U_L": 912,
            "AST_U_L": 744,
            "Bilirubin_mg_dL": 3.4,
            "INR": 1.7,
        },
        "ae_onset_date": "2024-09-02",
        "ae_description": "Severe hepatocellular drug-induced liver injury",
        "outcome": "recovering",
        "ground_truth": {
            "severity": "severe",
            "reporting_timeline": "15-day",
            "is_serious": True,
            "meddra_soc": "Hepatobiliary disorders",
            "meddra_preferred_term": "Drug-induced liver injury",
        },
        "scoring_weights": {
            "severity": 0.30,
            "timeline": 0.25,
            "is_serious": 0.10,
            "soc": 0.175,
            "pt": 0.175,
        },
        "accepted_soc_variations": ["hepatobiliary disorders", "hepatic disorders"],
        "accepted_pt_variations": ["drug-induced liver injury", "liver injury"],
    },
]

EXTRA_DEVIATION_CASES: List[Dict[str, Any]] = [
    {
        "site_id": "SITE-D-004",
        "site_name": "North Valley Research Institute",
        "visit_type": "Triggered Data Quality Visit",
        "study_phase": "Phase 2",
        "active_subjects": 29,
        "prior_deviations": 10,
        "last_monitoring_visit": "2024-02-21",
        "findings": [
            {
                "id": "F030",
                "category": "Delegation Log",
                "description": "Sub-investigator performing AE assessment not delegated for clinical evaluations.",
                "timestamp": "2024-05-08",
            },
            {
                "id": "F031",
                "category": "SAE Follow-up",
                "description": "Follow-up SAE documents submitted 19 days after sponsor query despite 5-day requirement.",
                "timestamp": "2024-05-08",
            },
            {
                "id": "F032",
                "category": "Temperature Log",
                "description": "No daily IP temperature entries for nine consecutive days.",
                "timestamp": "2024-05-09",
            },
        ],
        "ground_truth": {
            "deviation_type": "major",
            "capa_required": True,
            "site_risk_score": 7.6,
            "gcp_violation_ids": ["F030", "F031", "F032"],
        },
    },
    {
        "site_id": "SITE-E-005",
        "site_name": "Precision Clinical Partners",
        "visit_type": "Routine Monitoring Visit",
        "study_phase": "Phase 4",
        "active_subjects": 16,
        "prior_deviations": 1,
        "last_monitoring_visit": "2024-06-03",
        "findings": [
            {
                "id": "F040",
                "category": "Visit Window",
                "description": "One subject visit occurred one day outside target window but within protocol grace period.",
                "timestamp": "2024-07-01",
            },
            {
                "id": "F041",
                "category": "Source Notes",
                "description": "Single missing investigator signature later completed with dated correction.",
                "timestamp": "2024-07-01",
            },
        ],
        "ground_truth": {
            "deviation_type": "minor",
            "capa_required": False,
            "site_risk_score": 2.2,
            "gcp_violation_ids": [],
        },
    },
]

EXTRA_NARRATIVE_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "CASE-NR-002",
        "patient_demographics": {
            "age": 49,
            "sex": "Male",
            "weight_kg": 82,
            "height_cm": 178,
            "country": "Germany",
        },
        "study_drug": "Compound QX-118 (300mg QD)",
        "suspect_drugs": ["Compound QX-118"],
        "concomitant_medications": [
            {"name": "Levothyroxine", "dose": "50mcg QD", "indication": "Hypothyroidism"},
            {"name": "Ibuprofen", "dose": "400mg PRN", "indication": "Back pain"},
        ],
        "adverse_event": {
            "term": "Acute pancreatitis",
            "onset_date": "2024-10-14",
            "report_date": "2024-10-15",
            "meddra_soc": "Gastrointestinal disorders",
            "meddra_pt": "Acute pancreatitis",
            "ctcae_grade": 3,
            "seriousness_criteria": ["Hospitalization", "Medically significant"],
            "action_taken": "Investigational drug discontinued; IV fluids and analgesia initiated",
            "outcome": "recovering",
            "dechallenge_positive": True,
            "rechallenge_done": False,
        },
        "lab_values_timeline": [
            {"date": "2024-10-10", "Lipase_U_L": 68, "Amylase_U_L": 70},
            {"date": "2024-10-14", "Lipase_U_L": 702, "Amylase_U_L": 530},
            {"date": "2024-10-17", "Lipase_U_L": 210, "Amylase_U_L": 180},
        ],
        "medical_history": [
            "Hypothyroidism",
            "No prior pancreatitis",
            "No history of alcohol misuse",
        ],
        "action_taken": "QX-118 discontinued permanently, inpatient care initiated, and sponsor informed within 24h.",
        "outcome_at_last_followup": "Improving clinically; enzymes down-trending; discharged on Day 6.",
        "reference_documents": [
            "ICH E2B(R3)",
            "EMA GVP Module VI",
            "Protocol QX-118 v5.1",
        ],
        "ground_truth": {
            "causality": "possibly_related",
            "required_temporal_elements": [
                "lipase elevation before event",
                "onset after exposure",
                "dechallenge positive",
                "hospitalization timing",
            ],
            "required_narrative_sections": [
                "patient demographics",
                "relevant medical history",
                "study drug and dose",
                "concomitant medications",
                "AE description and onset",
                "lab values",
                "action taken",
                "outcome",
                "causality",
            ],
            "regulatory_compliance_flags": [
                "lipase_trend_documented",
                "suspect_drug_named",
                "seriousness_criteria_stated",
                "dechallenge_documented",
                "causality_assessment_provided",
            ],
        },
    }
]
