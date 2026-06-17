from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from config import get_llm_client
from rag.vector_store import retrieve_with_sources
from state import AegisState
from agents._utils import build_context_block, extract_json, verify_citations

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an adversarial drug safety specialist. Your only job is to find contraindications, dangerous drug-drug interactions, and allergy risks.
You are given the patient's current medication list, known allergies, and the antibiotic under consideration.
Cross-reference this against the pharmacological database provided in context.
Be aggressive — if there is any known interaction risk, flag it. Do not minimize.
You MUST base every claim on the provided reference excerpts. Do not use outside knowledge.
Every rag_evidence entry must be a verbatim substring copied exactly from the excerpts — do not paraphrase.
Return ONLY valid JSON with no markdown:
{
  "contraindications_found": <true or false>,
  "interactions": [
    {
      "drug_a": "<medication name>",
      "drug_b": "<antibiotic name>",
      "severity": "<mild|moderate|severe>",
      "description": "<clinical explanation>",
      "rag_evidence": "<verbatim substring from excerpts>"
    }
  ]
}
If no interactions are found, return an empty interactions list."""


async def run_contraindication_agent(state: AegisState) -> Dict[str, Any]:
    """Identify drug-drug interactions, contraindications, and allergy risks.

    Steps:
    1. Retrieve top-4 interaction/safety chunks for the antibiotic + current meds
    2. Call LLM with adversarial safety system prompt
    3. Parse JSON
    4. Strip fabricated rag_evidence entries
    5. Return validated contraindication findings
    """
    profile = state.patient_profile
    if profile is None:
        return {
            "contraindication_findings": {
                "contraindications_found": False,
                "interactions": [],
            }
        }

    meds_str = ", ".join(profile.current_medications) if profile.current_medications else "none"
    allergies_str = ", ".join(profile.known_allergies) if profile.known_allergies else "none"

    query = (
        f"{profile.antibiotic_in_question} drug interaction {meds_str} "
        f"contraindication allergy {allergies_str}"
    )
    rag_results = retrieve_with_sources(query, k=4)
    retrieved_chunks = [r["text"] for r in rag_results]
    context_block = build_context_block(rag_results)

    user_prompt = f"""{context_block}

PATIENT CASE:
- Antibiotic under consideration: {profile.antibiotic_in_question}
- Current medications: {meds_str}
- Known allergies: {allergies_str}
- Age: {profile.age} years

Identify all contraindications, drug-drug interactions, and allergy risks. Return ONLY JSON."""

    llm = get_llm_client()
    raw_response = await asyncio.to_thread(llm["call"], SYSTEM_PROMPT, user_prompt)

    try:
        parsed = extract_json(raw_response)
    except ValueError as e:
        logger.error("[contraindication_agent] JSON parse failed: %s", e)
        parsed = {"contraindications_found": False, "interactions": []}

    parsed.setdefault("contraindications_found", False)
    parsed.setdefault("interactions", [])

    # Per-interaction citation guard
    clean_interactions = []
    for interaction in parsed["interactions"]:
        evidence = interaction.get("rag_evidence", "")
        verified = verify_citations([evidence], retrieved_chunks, "contraindication_agent")
        interaction["rag_evidence"] = verified[0] if verified else ""
        clean_interactions.append(interaction)

    parsed["interactions"] = clean_interactions

    # Ensure contraindications_found is consistent with the interactions list
    if clean_interactions:
        parsed["contraindications_found"] = True

    logger.info(
        "[contraindication_agent] contraindications=%s interactions=%d",
        parsed["contraindications_found"],
        len(clean_interactions),
    )

    return {
        "contraindication_findings": parsed,
        "retrieved_chunks": state.retrieved_chunks + retrieved_chunks,
    }
