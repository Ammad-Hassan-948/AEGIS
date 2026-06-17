from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from config import get_llm_client
from state import AegisState, PatientProfile
from agents._utils import extract_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical intake assistant. Your only job is to extract structured patient data from free-text input.
Extract: chief_complaint, symptom_duration, symptom_severity (1-10 integer), current_medications (list of strings),
known_allergies (list of strings), age (integer), weight_kg (float), antibiotic_in_question (string).
Return ONLY a valid JSON object — no markdown, no explanation:
{
  "chief_complaint": "<string>",
  "symptom_duration": "<string>",
  "symptom_severity": <1-10>,
  "current_medications": ["<name dose>", ...],
  "known_allergies": ["<allergy>", ...],
  "age": <integer>,
  "weight_kg": <float>,
  "antibiotic_in_question": "<string>"
}
If a field cannot be determined from the input, use sensible defaults:
  symptom_severity → 5, current_medications → [], known_allergies → [], weight_kg → 70.0"""


async def run_intake_agent(state: AegisState) -> Dict[str, Any]:
    """Extract a structured PatientProfile from raw free-text input.

    If the state already has a patient_profile (e.g. loaded from a demo JSON),
    this node is a no-op and passes through unchanged.
    """
    if state.patient_profile is not None:
        logger.info("[intake_agent] patient_profile already set — skipping extraction.")
        return {}

    raw = state.raw_input_text
    if not raw.strip():
        logger.warning("[intake_agent] raw_input_text is empty.")
        return {}

    llm = get_llm_client()
    user_prompt = f"Extract structured patient data from the following input:\n\n{raw}"
    raw_response = await asyncio.to_thread(llm["call"], SYSTEM_PROMPT, user_prompt)

    try:
        parsed = extract_json(raw_response)
        profile = PatientProfile(**parsed)
        logger.info("[intake_agent] Extracted profile: %s", profile.chief_complaint)
        return {"patient_profile": profile}
    except Exception as e:
        logger.error("[intake_agent] Failed to parse PatientProfile: %s", e)
        return {}
