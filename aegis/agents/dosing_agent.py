from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from config import get_llm_client
from rag.vector_store import retrieve_with_sources
from state import AegisState
from agents._utils import build_context_block, extract_json, verify_citations

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical dosing specialist. You calculate safe, evidence-based antibiotic dosing based on patient parameters.
You are given: patient age, weight (kg), and the antibiotic in question.
Use standard formularies (e.g., BNF, WHO Essential Medicines dosing guidelines) from the provided context only.
You MUST base every claim on the provided reference excerpts. Do not use outside knowledge.
Every rag_evidence entry must be a verbatim substring copied exactly from the excerpts — do not paraphrase.
Calculate: recommended dose, frequency, duration, and flag if the patient parameters suggest dose adjustment is needed.
Return ONLY valid JSON with no markdown:
{
  "recommended_dose_mg": <number or null>,
  "frequency": "<e.g. 'twice daily', 'every 12 hours'>",
  "duration_days": <integer or null>,
  "dosing_flag": <true if dose adjustment needed or data insufficient, else false>,
  "deviation_note": "<note if dosing flag raised, else empty string>",
  "rag_evidence": "<verbatim substring from excerpts>"
}"""


async def run_dosing_agent(state: AegisState) -> Dict[str, Any]:
    """Calculate appropriate dosing range based on patient-specific parameters.

    Steps:
    1. Retrieve top-4 dosing guideline chunks for the antibiotic
    2. Call LLM with dosing specialist system prompt
    3. Parse JSON
    4. Strip fabricated rag_evidence entries
    5. Return validated dosing findings
    """
    profile = state.patient_profile
    if profile is None:
        return {
            "dosing_findings": {
                "recommended_dose_mg": None,
                "frequency": None,
                "duration_days": None,
                "dosing_flag": True,
                "deviation_note": "No patient profile available.",
                "rag_evidence": "",
            }
        }

    query = f"{profile.antibiotic_in_question} dosing dose adult weight age formulary BNF"
    rag_results = retrieve_with_sources(query, k=4)
    retrieved_chunks = [r["text"] for r in rag_results]
    context_block = build_context_block(rag_results)

    user_prompt = f"""{context_block}

PATIENT CASE:
- Antibiotic: {profile.antibiotic_in_question}
- Age: {profile.age} years
- Weight: {profile.weight_kg} kg
- Chief complaint / indication: {profile.chief_complaint}

Calculate the recommended dose, frequency, and duration for this patient. Return ONLY JSON."""

    llm = get_llm_client()
    raw_response = await asyncio.to_thread(llm["call"], SYSTEM_PROMPT, user_prompt)

    try:
        parsed = extract_json(raw_response)
    except ValueError as e:
        logger.error("[dosing_agent] JSON parse failed: %s", e)
        parsed = {
            "recommended_dose_mg": None,
            "frequency": None,
            "duration_days": None,
            "dosing_flag": True,
            "deviation_note": f"Agent parse error: {e}",
            "rag_evidence": "",
        }

    parsed.setdefault("recommended_dose_mg", None)
    parsed.setdefault("frequency", None)
    parsed.setdefault("duration_days", None)
    parsed.setdefault("dosing_flag", False)
    parsed.setdefault("deviation_note", "")
    parsed.setdefault("rag_evidence", "")

    # Normalise dose to float or None
    if parsed["recommended_dose_mg"] is not None:
        try:
            parsed["recommended_dose_mg"] = float(parsed["recommended_dose_mg"])
        except (TypeError, ValueError):
            parsed["recommended_dose_mg"] = None
            parsed["dosing_flag"] = True
            parsed["deviation_note"] = "Could not parse recommended dose from LLM response."

    # Per-agent citation guard
    evidence = parsed.get("rag_evidence", "")
    verified = verify_citations(
        [evidence] if evidence else [], retrieved_chunks, "dosing_agent"
    )
    parsed["rag_evidence"] = verified[0] if verified else ""

    if not parsed["rag_evidence"]:
        parsed["dosing_flag"] = True
        if not parsed["deviation_note"]:
            parsed["deviation_note"] = "No grounded dosing evidence found in retrieved context."

    logger.info(
        "[dosing_agent] dose=%s freq=%s duration=%s flag=%s",
        parsed["recommended_dose_mg"],
        parsed["frequency"],
        parsed["duration_days"],
        parsed["dosing_flag"],
    )

    return {
        "dosing_findings": parsed,
        "retrieved_chunks": state.retrieved_chunks + retrieved_chunks,
    }
