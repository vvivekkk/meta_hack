"""
Clinical Trial Case Bank
=========================
Realistic, de-identified clinical trial scenarios for each task.
Each case has ground-truth answers and scoring rubrics embedded.
"""
from typing import Any, Dict, List

# ─────────────────────────────────────────────────────────────
# TASK 1: ADVERSE EVENT TRIAGE CASES
# Easy → Medium → Hard difficulty
# ─────────────────────────────────────────────────────────────

AE_CASES: List[Dict[str, Any]] = [
    # CASE-AE-001 — Easy: obvious SAE, life-threatening
    {
        "case_id": "CASE-AE-001",
        "narrative": (
            "Subject 0042 (Male, 67 years) enrolled in Study ONCO-301 received "
            "Compound XR-442 400mg QD. On Day 14, subject presented to ER with "
            "acute chest pain and diaphoresis. ECG showed ST-elevation in leads "
            "V1-V4. Troponin I: 8.2 ng/mL (ULN: 0.04). Subject underwent emergency "
            "PCI. Diagnosed as anterior STEMI. Study drug discontinued. Subject "
            "hospitalized for 7 days, recovering."
        ),
        "patient_age": 67,
        "patient_sex": "Male",
        "study_drug": "Compound XR-442",
        "dose_mg": 400.0,
        "days_on_drug": 14,
        "relevant_medical_history": [
            "Hypertension (controlled)",
            "Type 2 diabetes mellitus",
            "Former smoker (20 pack-years, quit 2015)",
        ],
        "concomitant_medications": ["Metformin 500mg BID", "Lisinopril 10mg QD"],
        "lab_values": {
            "Troponin_I_ng_mL": 8.2,
            "Troponin_I_ULN": 0.04,
            "CK_MB_ng_mL": 42.0,
            "WBC": 11.2,
            "Hgb": 13.1,
        },
        "ae_onset_date": "2024-03-14",
        "ae_description": "Acute anterior ST-elevation myocardial infarction (STEMI)",
        "outcome": "recovering",
        "ground_truth": {
            "severity": "life_threatening",
            "reporting_timeline": "7-day",
            "is_serious": True,
            "meddra_soc": "Cardiac disorders",
            "meddra_preferred_term": "Myocardial infarction",
        },
        "scoring_weights": {
            "severity": 0.30,
            "timeline": 0.25,
            "is_serious": 0.10,
            "soc": 0.175,
            "pt": 0.175,
        },
        "accepted_soc_variations": [
            "cardiac disorders",
            "cardiac disorder",
        ],
        "accepted_pt_variations": [
            "myocardial infarction",
            "myocardial infarction acute",
            "acute myocardial infarction",
            "stemi",
            "st-elevation myocardial infarction",
        ],
    },

    # CASE-AE-002 — Medium: moderate severity, ambiguous causality
    {
        "case_id": "CASE-AE-002",
        "narrative": (
            "Subject 0078 (Female, 52 years) in Study DERM-105 receiving Compound "
            "BX-229 200mg BID for plaque psoriasis. On Day 45, subject reported "
            "nausea, vomiting (3-4x/day) and fatigue lasting 5 days. ALT: 68 U/L "
            "(ULN 35), AST: 55 U/L (ULN 40) — consistent with Grade 1 hepatic "
            "enzyme elevation per CTCAE v5. Subject managed with antiemetics, dose "
            "not modified. Symptoms resolved Day 52. No hospitalization required."
        ),
        "patient_age": 52,
        "patient_sex": "Female",
        "study_drug": "Compound BX-229",
        "dose_mg": 200.0,
        "days_on_drug": 45,
        "relevant_medical_history": [
            "Plaque psoriasis (moderate-to-severe)",
            "GERD",
        ],
        "concomitant_medications": ["Omeprazole 20mg QD"],
        "lab_values": {
            "ALT_U_L": 68,
            "ALT_ULN": 35,
            "AST_U_L": 55,
            "AST_ULN": 40,
            "Bilirubin_mg_dL": 0.8,
            "ALP_U_L": 72,
        },
        "ae_onset_date": "2024-05-20",
        "ae_description": "Nausea, vomiting, fatigue with mild transaminase elevation",
        "outcome": "resolved",
        "ground_truth": {
            "severity": "moderate",
            "reporting_timeline": "routine",
            "is_serious": False,
            "meddra_soc": "Gastrointestinal disorders",
            "meddra_preferred_term": "Nausea",
        },
        "scoring_weights": {
            "severity": 0.30,
            "timeline": 0.25,
            "is_serious": 0.10,
            "soc": 0.175,
            "pt": 0.175,
        },
        "accepted_soc_variations": [
            "gastrointestinal disorders",
            "gastrointestinal disorder",
        ],
        "accepted_pt_variations": [
            "nausea",
            "nausea and vomiting",
        ],
    },

    # CASE-AE-003 — Hard: rare SAE, unexpected, requires SUSAR classification
    {
        "case_id": "CASE-AE-003",
        "narrative": (
            "Subject 0113 (Male, 34 years) in Phase 2 Study NEURO-88 (blinded, "
            "first-in-human study) receiving Compound NX-770 50mg QD for "
            "treatment-resistant depression. On Day 22, subject's caregiver called "
            "site reporting confusion, visual hallucinations, and ataxia. Subject "
            "evaluated by neurologist; MRI brain showed T2 hyperintensities in "
            "bilateral basal ganglia. CSF analysis: mild pleocytosis (15 WBC/μL). "
            "Lumbar puncture HSV PCR: negative. Autoimmune encephalitis panel: "
            "negative. Impression: possible drug-induced encephalopathy. Event not "
            "listed in Investigator's Brochure for this compound class. Subject "
            "hospitalized. Drug discontinued. Improving on Day 30."
        ),
        "patient_age": 34,
        "patient_sex": "Male",
        "study_drug": "Compound NX-770",
        "dose_mg": 50.0,
        "days_on_drug": 22,
        "relevant_medical_history": [
            "Treatment-resistant major depressive disorder",
            "No known neurological history",
        ],
        "concomitant_medications": ["None"],
        "lab_values": {
            "CSF_WBC_per_uL": 15,
            "CSF_protein_mg_dL": 48,
            "CSF_glucose_mg_dL": 62,
            "Serum_Na_mEq_L": 138,
            "Serum_K_mEq_L": 4.0,
        },
        "ae_onset_date": "2024-07-08",
        "ae_description": "Drug-induced encephalopathy with basal ganglia involvement",
        "outcome": "recovering",
        "ground_truth": {
            "severity": "severe",
            "reporting_timeline": "15-day",
            "is_serious": True,
            "meddra_soc": "Nervous system disorders",
            "meddra_preferred_term": "Encephalopathy",
        },
        "scoring_weights": {
            "severity": 0.30,
            "timeline": 0.30,
            "is_serious": 0.10,
            "soc": 0.15,
            "pt": 0.15,
        },
        "accepted_soc_variations": [
            "nervous system disorders",
            "nervous system disorder",
        ],
        "accepted_pt_variations": [
            "encephalopathy",
            "drug-induced encephalopathy",
            "toxic encephalopathy",
            "encephalitis",
        ],
    },
]

# ─────────────────────────────────────────────────────────────
# TASK 2: PROTOCOL DEVIATION AUDIT CASES
# ─────────────────────────────────────────────────────────────

DEVIATION_CASES: List[Dict[str, Any]] = [
    # SITE-A: Easy — straightforward minor deviations
    {
        "site_id": "SITE-A-001",
        "site_name": "Central University Hospital",
        "visit_type": "Routine Monitoring Visit",
        "study_phase": "Phase 3",
        "active_subjects": 24,
        "prior_deviations": 2,
        "last_monitoring_visit": "2024-01-15",
        "findings": [
            {
                "id": "F001",
                "category": "Informed Consent",
                "description": "Subject 0012 re-consented 2 days after protocol amendment v3 effective date (required within 1 day).",
                "timestamp": "2024-02-10",
            },
            {
                "id": "F002",
                "category": "Visit Window",
                "description": "Subject 0018 Week-12 visit occurred on Day 88 (protocol window: Day 77-91). Within window.",
                "timestamp": "2024-02-12",
            },
            {
                "id": "F003",
                "category": "Lab Specimen",
                "description": "PK sample for Subject 0020 stored at -18°C instead of required -20°C for 6 hours before transfer. Sample shipped and accepted by central lab.",
                "timestamp": "2024-02-14",
            },
        ],
        "ground_truth": {
            "deviation_type": "minor",
            "capa_required": False,
            "site_risk_score": 2.5,
            "gcp_violation_ids": [],
        },
    },

    # SITE-B: Medium — mix of major/minor, CAPA required
    {
        "site_id": "SITE-B-002",
        "site_name": "Metro Clinical Research Center",
        "visit_type": "For-Cause Monitoring Visit",
        "study_phase": "Phase 2b",
        "active_subjects": 18,
        "prior_deviations": 8,
        "last_monitoring_visit": "2023-11-01",
        "findings": [
            {
                "id": "F010",
                "category": "Eligibility Criteria",
                "description": "Subject 0031 enrolled despite baseline eGFR of 42 mL/min/1.73m² (protocol exclusion: eGFR <45). Subject received 3 doses before discovered.",
                "timestamp": "2024-01-20",
            },
            {
                "id": "F011",
                "category": "Blinding",
                "description": "Unblinding log shows Site PI accessed treatment assignment for Subject 0035 prior to endpoint assessment — without emergency justification documented.",
                "timestamp": "2024-01-22",
            },
            {
                "id": "F012",
                "category": "SAE Reporting",
                "description": "SAE for Subject 0029 (hospitalization) reported to sponsor on Day 8 post-event. Protocol and GCP require expedited report within 24h.",
                "timestamp": "2024-01-25",
            },
            {
                "id": "F013",
                "category": "Drug Accountability",
                "description": "4 units of IP unaccounted for in drug log. No dispensing records found.",
                "timestamp": "2024-01-25",
            },
        ],
        "ground_truth": {
            "deviation_type": "major",
            "capa_required": True,
            "site_risk_score": 8.2,
            "gcp_violation_ids": ["F010", "F011", "F012", "F013"],
        },
    },

    # SITE-C: Hard — complex multi-category audit, data integrity
    {
        "site_id": "SITE-C-003",
        "site_name": "Regional Oncology Associates",
        "visit_type": "Data Integrity Audit",
        "study_phase": "Phase 3 Pivotal",
        "active_subjects": 41,
        "prior_deviations": 15,
        "last_monitoring_visit": "2023-09-15",
        "findings": [
            {
                "id": "F020",
                "category": "Source Data Verification",
                "description": "EDC entry for Subject 0058 tumor response (Week 12): 'Partial Response'. Source document (radiology report): 'Stable Disease'. Discordant without documented rationale.",
                "timestamp": "2024-03-01",
            },
            {
                "id": "F021",
                "category": "Endpoint Assessment",
                "description": "Independent radiologist reading for 6 subjects has not been completed despite protocol requirement. These subjects are included in interim analysis dataset.",
                "timestamp": "2024-03-01",
            },
            {
                "id": "F022",
                "category": "Informed Consent",
                "description": "4 subjects consented by a research coordinator who is not listed on the site delegation log. Coordinator has no GCP training certificate on file.",
                "timestamp": "2024-03-02",
            },
            {
                "id": "F023",
                "category": "Adverse Event",
                "description": "Subject 0062 reported worsening fatigue (Grade 3 per investigator assessment) but no AE was entered in EDC. Not identified until SDV.",
                "timestamp": "2024-03-03",
            },
            {
                "id": "F024",
                "category": "Drug Storage",
                "description": "Temperature excursion for IP fridge: recorded 8.2°C for 14 hours on Feb 28 (required: 2-8°C). 12 subject doses from affected batch administered before excursion detected.",
                "timestamp": "2024-03-04",
            },
        ],
        "ground_truth": {
            "deviation_type": "major",
            "capa_required": True,
            "site_risk_score": 9.4,
            "gcp_violation_ids": ["F020", "F021", "F022", "F023", "F024"],
        },
    },
]

# ─────────────────────────────────────────────────────────────
# TASK 3: SAFETY NARRATIVE CASES
# ─────────────────────────────────────────────────────────────

NARRATIVE_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "CASE-NR-001",
        "patient_demographics": {
            "age": 58,
            "sex": "Female",
            "weight_kg": 68,
            "height_cm": 165,
            "ethnicity": "White",
            "country": "United States",
        },
        "study_drug": "Compound ZL-550 (150mg BID)",
        "suspect_drugs": ["Compound ZL-550"],
        "concomitant_medications": [
            {"name": "Warfarin", "dose": "5mg QD", "indication": "Atrial fibrillation"},
            {"name": "Atorvastatin", "dose": "40mg QD", "indication": "Hyperlipidemia"},
            {"name": "Metoprolol", "dose": "25mg BID", "indication": "Atrial fibrillation"},
        ],
        "adverse_event": {
            "term": "Gastrointestinal haemorrhage",
            "onset_date": "2024-04-10",
            "report_date": "2024-04-11",
            "meddra_soc": "Gastrointestinal disorders",
            "meddra_pt": "Gastrointestinal haemorrhage",
            "ctcae_grade": 3,
            "seriousness_criteria": ["Hospitalization", "Medically significant"],
            "action_taken": "Study drug suspended; transfused 2 units pRBC",
            "outcome": "Recovered with sequelae",
            "dechallenge_positive": True,
            "rechallenge_done": False,
        },
        "lab_values_timeline": [
            {"date": "2024-03-01", "INR": 2.1, "Hgb_g_dL": 12.8, "Platelets_K_uL": 210},
            {"date": "2024-04-08", "INR": 4.7, "Hgb_g_dL": 11.2, "Platelets_K_uL": 195},
            {"date": "2024-04-10", "INR": 5.9, "Hgb_g_dL": 7.1, "Platelets_K_uL": 188},
            {"date": "2024-04-12", "INR": 2.4, "Hgb_g_dL": 9.8, "Platelets_K_uL": 202},
        ],
        "medical_history": [
            "Atrial fibrillation (diagnosed 2019)",
            "Hyperlipidemia",
            "No prior GI bleeding",
            "No H. pylori infection",
        ],
        "action_taken": "ZL-550 suspended on Day of event. Warfarin held. GI endoscopy performed: bleeding duodenal ulcer identified and treated. 2 units pRBC transfused.",
        "outcome_at_last_followup": "Subject recovered with residual mild anemia. ZL-550 not restarted.",
        "reference_documents": [
            "ICH E2B(R3) Implementation Guide",
            "ZL-550 Investigator's Brochure v4.0 Section 8.2 (GI adverse events)",
            "21 CFR Part 312.32 — IND safety reporting",
            "CTCAE v5.0 — Gastrointestinal disorders",
        ],
        "ground_truth": {
            "causality": "probably_related",
            "required_temporal_elements": [
                "INR elevation before event",
                "warfarin interaction",
                "onset 3 days after INR spike",
                "positive dechallenge",
                "dose information",
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
                "INR_mentioned",
                "warfarin_interaction_noted",
                "seriousness_criteria_stated",
                "dechallenge_documented",
                "causality_assessment_provided",
            ],
        },
    }
]

# Optional high-fidelity expansion set used for RL training and stress evaluation.
try:
    from tasks.production_cases import (  # type: ignore
        EXTRA_AE_CASES,
        EXTRA_DEVIATION_CASES,
        EXTRA_NARRATIVE_CASES,
    )

    AE_CASES.extend(EXTRA_AE_CASES)
    DEVIATION_CASES.extend(EXTRA_DEVIATION_CASES)
    NARRATIVE_CASES.extend(EXTRA_NARRATIVE_CASES)
except Exception:
    # Keep baseline case bank available even if optional expansion file is absent.
    pass
