from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from config import get_llm_client
from rag.vector_store import retrieve_with_sources
from state import AegisState
from agents._utils import build_context_block, extract_json, verify_citations

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical pharmacologist. Assess whether an antibiotic is appropriate for these symptoms.
You MUST base your reasoning only on the provided reference excerpts below — if the excerpts do not
address this specific case, say so explicitly in your reasoning rather than relying on outside knowledge.
Every claim in rag_evidence must be a substring that actually appears in the provided excerpts —
do not paraphrase into rag_evidence, copy the relevant fragment exactly.
Return ONLY valid JSON with no markdown, no explanation, just the JSON object:
{
  "antibiotic_warranted": <true or false>,
  "reasoning": "<concise clinical reasoning based strictly on the excerpts>",
  "confidence_score": <integer 0-100>,
  "rag_evidence": ["<verbatim substring from excerpts>", ...]
}"""


async def run_appropriateness_agent(state: AegisState) -> Dict[str, Any]:
    """Assess whether an antibiotic is clinically warranted for this patient.

    Steps:
    1. Retrieve top-4 clinical guideline chunks relevant to the chief complaint
    2. Call LLM with grounded system prompt
    3. Parse JSON response
    4. Strip any fabricated citations not found verbatim in retrieved text
    5. Return validated findings dict
    """
    profile = state.patient_profile
    if profile is None:
        return {
            "appropriateness_findings": {
                "antibiotic_warranted": False,
                "reasoning": "No patient profile available.",
                "confidence_score": 0,
                "rag_evidence": [],
            }
        }

    logger.info("[appropriateness_agent] Starting RAG retrieval...")
    query = f"{profile.chief_complaint} {profile.antibiotic_in_question} antibiotic appropriate"
    rag_results = retrieve_with_sources(query, k=4)
    logger.info("[appropriateness_agent] RAG retrieval complete (found %d chunks).", len(rag_results))
    retrieved_chunks = [r["text"] for r in rag_results]
    context_block = build_context_block(rag_results)

    user_prompt = f"""{context_block}

PATIENT CASE:
- Chief complaint: {profile.chief_complaint}
- Symptom duration: {profile.symptom_duration}
- Symptom severity (1-10): {profile.symptom_severity}
- Age: {profile.age} years
- Antibiotic requested: {profile.antibiotic_in_question}

Assess whether {profile.antibiotic_in_question} is clinically warranted. Return ONLY JSON."""

    llm = get_llm_client()
    raw_response = await asyncio.to_thread(llm["call"], SYSTEM_PROMPT, user_prompt)

    try:
        parsed = extract_json(raw_response)
    except ValueError as e:
        logger.error("[appropriateness_agent] JSON parse failed: %s", e)
        parsed = {
            "antibiotic_warranted": False,
            "reasoning": f"Agent parse error: {e}",
            "confidence_score": 0,
            "rag_evidence": [],
        }

    # Normalise types
    parsed.setdefault("antibiotic_warranted", False)
    parsed.setdefault("reasoning", "")
    parsed.setdefault("confidence_score", 0)
    parsed.setdefault("rag_evidence", [])
    if isinstance(parsed["confidence_score"], float):
        parsed["confidence_score"] = int(parsed["confidence_score"])

    # Per-agent citation guard — drop anything not verbatim in retrieved text
    parsed["rag_evidence"] = verify_citations(
        parsed["rag_evidence"], retrieved_chunks, "appropriateness_agent"
    )

    logger.info(
        "[appropriateness_agent] warranted=%s confidence=%s citations=%d",
        parsed["antibiotic_warranted"],
        parsed["confidence_score"],
        len(parsed["rag_evidence"]),
    )

    # Merge retrieved chunks into state so Safety Gate can verify them later
    return {
        "appropriateness_findings": parsed,
        "retrieved_chunks": state.retrieved_chunks + retrieved_chunks,
    }
