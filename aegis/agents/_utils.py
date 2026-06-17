"""Shared helpers used by all clinical agents."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def extract_json(raw: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from an LLM response string.

    LLMs sometimes wrap JSON in markdown fences — this strips those first,
    then falls back to a regex search for the outermost { ... } block.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: find the outermost { ... } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from LLM response:\n{raw[:500]}")


def verify_citations(
    citations: List[str],
    retrieved_chunks: List[str],
    agent_name: str,
) -> List[str]:
    """Drop any citation that is not a verbatim substring of the retrieved chunks.

    This is the per-agent, pre-Safety-Gate hallucination guard.
    Returns only verified citations.
    """
    joined = " ".join(retrieved_chunks)
    verified: List[str] = []
    for c in citations:
        if c and c.strip() in joined:
            verified.append(c)
        else:
            logger.warning(
                "[%s] Dropping fabricated citation (not found in retrieved context): %r",
                agent_name,
                c[:120],
            )
    return verified


def build_context_block(results: List[dict]) -> str:
    """Format retrieved RAG results into a numbered reference block for the LLM."""
    lines = ["REFERENCE EXCERPTS (base ALL claims on these — nothing else):"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] Source: {r['source']}\n    {r['text']}")
    return "\n".join(lines)
