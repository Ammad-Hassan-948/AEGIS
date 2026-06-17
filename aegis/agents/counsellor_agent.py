from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from config import get_llm_client
from state import AegisState

logger = logging.getLogger(__name__)

# System prompt for a fully-validated report
SYSTEM_PROMPT_NORMAL = """You are a compassionate clinical counsellor. You receive a validated clinical findings report about an antibiotic request.
Your job is to explain the findings to a patient in plain, clear language — not medical jargon.
Structure your response as:
1. A brief summary of what was found
2. The key risks or flags (if any), explained simply
3. Whether this antibiotic appears appropriate for their situation
4. What they should do next (e.g., "Please consult a licensed physician before taking this")

You must NEVER make a final prescription decision. You inform, flag, and refer.
End every response with: "AEGIS does not replace a licensed medical professional. Please consult a qualified physician or pharmacist."
Tone: warm, clear, non-alarmist unless risk is CRITICAL."""

# System prompt when Safety Gate exhausted retries without full validation
SYSTEM_PROMPT_DEGRADED = """You are a compassionate clinical counsellor. A clinical analysis was attempted for this patient's antibiotic request,
but the automated validation system could not fully verify the findings after multiple attempts.
You must clearly communicate this to the patient in plain, warm language.
Your response must:
1. Acknowledge that an analysis was attempted but could not be fully validated
2. NOT present any specific findings as confirmed conclusions
3. Clearly recommend that the patient consult a licensed physician or pharmacist in person before proceeding
4. Reassure the patient that this is a precautionary measure, not a cause for alarm
End with: "AEGIS does not replace a licensed medical professional. Please consult a qualified physician or pharmacist."
Tone: calm, warm, and honest."""


async def run_counsellor_agent(state: AegisState) -> Dict[str, Any]:
    """Translate the validated ClinicalFindingsReport into patient-friendly language.

    If gate_degraded is True, the counsellor switches to a degraded-mode prompt
    that recommends in-person consultation without presenting partial findings as final.
    """
    report = state.compiled_report
    gate_degraded = False

    if report is not None:
        gate_degraded = (
            report.gate_degraded
            if hasattr(report, "gate_degraded")
            else report.get("gate_degraded", False) if isinstance(report, dict)
            else False
        )

    if gate_degraded or report is None:
        system_prompt = SYSTEM_PROMPT_DEGRADED
        user_prompt = (
            "The automated clinical analysis could not be fully validated. "
            "Please generate a warm, honest message explaining this to the patient "
            "and directing them to seek in-person clinical advice."
        )
        logger.warning("[counsellor_agent] gate_degraded=True — using degraded-mode prompt.")
    else:
        system_prompt = SYSTEM_PROMPT_NORMAL

        # Pull risk level and key findings from the report
        risk_level = getattr(report, "overall_risk_level", "UNKNOWN")
        antibiotic_warranted = getattr(report, "antibiotic_warranted", None)
        confidence = getattr(report, "appropriateness_confidence", None)
        interactions = getattr(report, "interactions", [])
        dose = getattr(report, "recommended_dose_mg", None)
        frequency = getattr(report, "frequency", None)
        duration = getattr(report, "duration_days", None)
        dosing_flag = getattr(report, "dosing_flag", False)
        evidence_summary = getattr(report, "rag_evidence_summary", [])

        interaction_lines = ""
        for ix in interactions:
            if hasattr(ix, "drug_a"):
                interaction_lines += f"  - {ix.drug_a} + {ix.drug_b} ({ix.severity}): {ix.description}\n"
            elif isinstance(ix, dict):
                interaction_lines += (
                    f"  - {ix.get('drug_a', '?')} + {ix.get('drug_b', '?')} "
                    f"({ix.get('severity', '?')}): {ix.get('description', '')}\n"
                )

        profile = state.patient_profile
        patient_desc = ""
        if profile:
            patient_desc = (
                f"Patient: {profile.age}yo, {profile.weight_kg}kg\n"
                f"Complaint: {profile.chief_complaint}\n"
                f"Antibiotic requested: {profile.antibiotic_in_question}\n"
                f"Current medications: {', '.join(profile.current_medications) or 'none'}\n"
                f"Known allergies: {', '.join(profile.known_allergies) or 'none'}"
            )

        user_prompt = f"""{patient_desc}

VALIDATED CLINICAL FINDINGS:
- Overall risk level: {risk_level}
- Antibiotic warranted: {antibiotic_warranted} (confidence: {confidence}%)
- Contraindications found: {bool(interactions)}
- Interactions:
{interaction_lines or '  None identified'}
- Recommended dose: {dose} mg | Frequency: {frequency} | Duration: {duration} days
- Dosing flag: {dosing_flag}
- Evidence summary: {'; '.join(evidence_summary[:2]) if evidence_summary else 'see report'}

Write a patient-friendly explanation of these findings."""

    llm = get_llm_client()
    counsellor_output = await asyncio.to_thread(llm["call"], system_prompt, user_prompt)

    logger.info(
        "[counsellor_agent] Output generated (%d chars, degraded=%s)",
        len(counsellor_output),
        gate_degraded,
    )
    return {"counsellor_output": counsellor_output}
