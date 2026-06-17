from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from config import get_llm_client
from state import AegisState, ClinicalFindingsReport, DrugInteraction
from agents._utils import extract_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior clinical reviewer. You receive structured findings from three specialist agents:
an appropriateness assessor, a contraindication checker, and a dosing specialist.
Your job is to compile these into a single, coherent ClinicalFindingsReport JSON object.
Do NOT add new clinical claims. Only synthesize what the agents have found.
Assign an overall_risk_level: LOW / MODERATE / HIGH / CRITICAL based on aggregate findings:
  - CRITICAL: severe contraindication or clearly inappropriate antibiotic with dangerous interaction
  - HIGH: moderate interaction risk or inappropriateness with some uncertainty
  - MODERATE: minor interactions, dosing concerns, or partial appropriateness uncertainty
  - LOW: antibiotic appears appropriate, no interactions, dosing within range
Ensure every claim in rag_evidence_summary is copied verbatim from agent rag_evidence fields — do not paraphrase.
Return ONLY valid JSON with no markdown:
{
  "antibiotic_warranted": <bool>,
  "appropriateness_confidence": <0-100>,
  "contraindications_found": <bool>,
  "interactions": [
    {
      "drug_a": "<string>",
      "drug_b": "<string>",
      "severity": "<mild|moderate|severe>",
      "description": "<string>",
      "rag_evidence": "<verbatim substring>"
    }
  ],
  "recommended_dose_mg": <number or null>,
  "frequency": "<string or null>",
  "duration_days": <integer or null>,
  "dosing_flag": <bool>,
  "overall_risk_level": "<LOW|MODERATE|HIGH|CRITICAL>",
  "rag_evidence_summary": ["<verbatim substring>", ...]
}"""


async def run_supervisor_agent(state: AegisState) -> Dict[str, Any]:
    """Compile appropriateness, contraindication, and dosing findings into one report.

    If validation_errors are present from a prior Safety Gate retry, they are
    passed back to the LLM so it can correct the specific issues.
    """
    profile = state.patient_profile

    retry_note = ""
    if state.validation_errors:
        retry_note = (
            "\n\nPREVIOUS COMPILATION WAS REJECTED BY THE SAFETY GATE. "
            "Fix these specific issues in your new output:\n"
            + "\n".join(f"  - {e}" for e in state.validation_errors)
        )

    user_prompt = f"""Compile the following agent findings into a ClinicalFindingsReport.

PATIENT:
- Chief complaint: {profile.chief_complaint if profile else 'unknown'}
- Antibiotic: {profile.antibiotic_in_question if profile else 'unknown'}
- Age: {profile.age if profile else 'unknown'}, Weight: {profile.weight_kg if profile else 'unknown'} kg

APPROPRIATENESS AGENT FINDINGS:
{state.appropriateness_findings}

CONTRAINDICATION AGENT FINDINGS:
{state.contraindication_findings}

DOSING AGENT FINDINGS:
{state.dosing_findings}
{retry_note}

Return ONLY the compiled ClinicalFindingsReport JSON."""

    llm = get_llm_client()
    raw_response = await asyncio.to_thread(llm["call"], SYSTEM_PROMPT, user_prompt)

    try:
        parsed = extract_json(raw_response)

        # Coerce nested interactions list to DrugInteraction objects
        raw_interactions = parsed.get("interactions", [])
        coerced_interactions = []
        for item in raw_interactions:
            if isinstance(item, dict):
                try:
                    coerced_interactions.append(DrugInteraction(**item))
                except Exception as ie:
                    logger.warning("[supervisor_agent] Skipping malformed interaction: %s — %s", item, ie)
            elif isinstance(item, DrugInteraction):
                coerced_interactions.append(item)
        parsed["interactions"] = coerced_interactions

        # Ensure gate_degraded is not overwritten during retry
        parsed.setdefault("gate_degraded", False)

        report = ClinicalFindingsReport(**parsed)
        logger.info(
            "[supervisor_agent] Compiled report: risk=%s contraindications=%s interactions=%d",
            report.overall_risk_level,
            report.contraindications_found,
            len(report.interactions),
        )
        return {"compiled_report": report}

    except Exception as e:
        logger.error("[supervisor_agent] Failed to compile ClinicalFindingsReport: %s", e)
        # Return None so the Safety Gate catches the missing report
        return {"compiled_report": None}
