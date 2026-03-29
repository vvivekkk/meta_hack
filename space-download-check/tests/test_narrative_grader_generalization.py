from __future__ import annotations

from models import CausalityAssessment, SafetyNarrativeAction
from tasks.graders import grade_safety_narrative
from tasks.production_cases import EXTRA_NARRATIVE_CASES


def test_narrative_grader_scores_non_warfarin_case_well() -> None:
    case = EXTRA_NARRATIVE_CASES[0]

    action = SafetyNarrativeAction(
        narrative_text=(
            "A 49-year-old male receiving Compound QX-118 developed acute pancreatitis with onset on 2024-10-14 "
            "and report on 2024-10-15. The patient had no prior pancreatitis and was on levothyroxine and ibuprofen. "
            "Laboratory timeline showed lipase and amylase elevation on the event date followed by down-trending values "
            "during follow-up. The event met seriousness criteria due to hospitalization and medical significance. "
            "QX-118 was discontinued, inpatient care was initiated, and dechallenge was positive with clinical improvement. "
            "Outcome at last follow-up was recovery trend with discharge on Day 6. Causality assessment: possibly related "
            "based on temporal association after exposure and improvement after discontinuation."
        ),
        causality_assessment=CausalityAssessment.POSSIBLY_RELATED,
        key_temporal_flags=[
            "lipase elevation before clinical worsening",
            "event onset on 2024-10-14",
            "dechallenge positive after discontinuation",
        ],
        dechallenge_positive=True,
        rechallenge_positive=None,
    )

    reward = grade_safety_narrative(action, case)
    assert reward.total >= 0.75
