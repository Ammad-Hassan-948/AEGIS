"""Safety Gate — pure Python, zero LLM calls.

This is the deterministic validation layer that sits between the Supervisor
and the Counsellor.  Nothing reaches the patient unless it passes here.

Routing:
  - All checks pass                  → next_step = "counsellor"
  - Any check fails, iteration < 3   → next_step = "retry_compilation"
  - Any check fails, iteration >= 3  → gate_degraded = True, next_step = "counsellor"
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from state import AegisState

logger = logging.getLogger(__name__)

_VALID_RISK_LEVELS = {"LOW", "MODERATE", "HIGH", "CRITICAL"}
_HIGH_RISK_LEVELS = {"HIGH", "CRITICAL"}
_MAX_RETRIES = 3


def _get_field(obj: Any, field: str, default: Any = None) -> Any:
    """Access a field from either a Pydantic model or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


def _collect_all_evidence(state: AegisState) -> List[str]:
    """Pull every rag_evidence string out of the compiled report."""
    report = state.compiled_report
    if report is None:
        return []

    evidence: List[str] = []

    # Top-level summary list
    evidence.extend(_get_field(report, "rag_evidence_summary", []))

    # Per-interaction evidence fields
    for interaction in _get_field(report, "interactions", []):
        ev = _get_field(interaction, "rag_evidence", "")
        if ev:
            evidence.append(ev)

    return [e for e in evidence if e and e.strip()]


def _check_substring_grounding(evidence_entries: List[str], retrieved_chunks: List[str]) -> List[str]:
    """Return a list of evidence entries that cannot be found in retrieved_chunks."""
    if not retrieved_chunks:
        # No chunks means we cannot verify — treat as ungrounded
        return evidence_entries

    joined = " ".join(retrieved_chunks)
    return [e for e in evidence_entries if e.strip() not in joined]


def safety_gate_node(state: AegisState) -> Dict[str, Any]:
    """Validate the compiled ClinicalFindingsReport.  Pure Python — no LLM calls."""

    errors: List[str] = []
    report = state.compiled_report

    # ── Guard: report must exist ──────────────────────────────────────────────
    if report is None:
        errors.append("compiled_report is None — Supervisor did not produce output.")
        return _route(state, errors)

    # ── Rule 1: overall_risk_level must be a recognised value ─────────────────
    risk_level = _get_field(report, "overall_risk_level", "")
    if risk_level not in _VALID_RISK_LEVELS:
        errors.append(
            f"Rule 1 FAIL — invalid overall_risk_level: '{risk_level}'. "
            f"Must be one of {sorted(_VALID_RISK_LEVELS)}."
        )

    # ── Rule 2: rag_evidence_summary must not be empty ───────────────────────
    if not _get_field(report, "rag_evidence_summary", []):
        errors.append(
            "Rule 2 FAIL — rag_evidence_summary is empty. "
            "Every report must carry at least one grounded citation."
        )

    # ── Rule 3: appropriateness_confidence must be in [0, 100] ───────────────
    confidence = _get_field(report, "appropriateness_confidence", -1)
    if not (0 <= confidence <= 100):
        errors.append(
            f"Rule 3 FAIL — appropriateness_confidence={confidence} "
            "is out of bounds [0, 100]."
        )

    # ── Rule 4: if contraindications_found, interactions must not be empty ────
    if _get_field(report, "contraindications_found", False) and not _get_field(report, "interactions", []):
        errors.append(
            "Rule 4 FAIL — contraindications_found=True but interactions list is empty. "
            "Each flagged contraindication must be described."
        )

    # ── Rule 5: HIGH/CRITICAL interactions must each have rag_evidence ────────
    if _get_field(report, "overall_risk_level") in _HIGH_RISK_LEVELS:
        for interaction in _get_field(report, "interactions", []):
            ev = _get_field(interaction, "rag_evidence", "")
            if not ev or not ev.strip():
                drug_a = _get_field(interaction, "drug_a", "?")
                drug_b = _get_field(interaction, "drug_b", "?")
                errors.append(
                    f"Rule 5 FAIL — interaction '{drug_a} + {drug_b}' "
                    f"has no rag_evidence but report is {_get_field(report, 'overall_risk_level')}."
                )

    # ── Rule 6: all evidence entries must be verbatim substrings of retrieved chunks ──
    all_evidence = _collect_all_evidence(state)
    ungrounded = _check_substring_grounding(all_evidence, state.retrieved_chunks)
    for entry in ungrounded:
        errors.append(
            f"Rule 6 FAIL — evidence entry not found verbatim in retrieved context: "
            f"'{entry[:120]}'"
        )

    if errors:
        for err in errors:
            logger.warning("[safety_gate] %s", err)

    return _route(state, errors)


def _route(state: AegisState, errors: List[str]) -> Dict[str, Any]:
    """Decide next step based on error list and retry count."""

    if not errors:
        logger.info(
            "[safety_gate] All checks passed (iteration=%d). Routing to counsellor.",
            state.iteration_count,
        )
        return {
            "validation_errors": [],
            "next_step": "counsellor",
        }

    if state.iteration_count < _MAX_RETRIES:
        logger.warning(
            "[safety_gate] %d error(s) found — retry %d/%d. Routing to supervisor for recompilation.",
            len(errors),
            state.iteration_count + 1,
            _MAX_RETRIES,
        )
        return {
            "validation_errors": errors,
            "iteration_count": state.iteration_count + 1,
            "next_step": "retry_compilation",
        }

    # Exhausted retries — degrade gracefully
    logger.error(
        "[safety_gate] Exhausted %d retries with %d error(s). Setting gate_degraded=True.",
        _MAX_RETRIES,
        len(errors),
    )
    if state.compiled_report is not None:
        report = state.compiled_report
        if isinstance(report, dict):
            report["gate_degraded"] = True
            updated_report = report
        else:
            updated_report = report.model_copy(update={"gate_degraded": True})
        return {
            "compiled_report": updated_report,
            "validation_errors": [],   # clear so UI shows the degraded banner, not the error list
            "next_step": "counsellor",
        }

    return {
        "validation_errors": [],
        "next_step": "counsellor",
    }
