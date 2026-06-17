"""AEGIS FastAPI backend.

Run:  uvicorn main:app --reload --port 8000

Endpoints
---------
GET  /health                       — liveness check
POST /api/v1/intake                — store PatientProfile, return patient_id
POST /api/v1/analyze               — stream SSE pipeline events for a patient_id
POST /api/v1/upload-image          — vision LLM extracts PatientProfile fields from image
GET  /api/v1/report/{patient_id}   — return cached ClinicalFindingsReport from SQLite
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel

import config as cfg
from state import AegisState, ClinicalFindingsReport, PatientProfile

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# SQLite — simple persistent cache
# ─────────────────────────────────────────────────────────────────────────────

_DB_PATH = os.path.join(os.environ.get("TMPDIR", "/tmp"), "aegis_cache.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id          TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                patient_id       TEXT PRIMARY KEY,
                report_json      TEXT,
                counsellor_output TEXT,
                risk_level       TEXT,
                gate_degraded    INTEGER DEFAULT 0,
                created_at       TEXT DEFAULT (datetime('now'))
            )""")
        conn.commit()


def _store_patient(patient_id: str, profile: PatientProfile) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO patients (id, profile_json) VALUES (?, ?)",
            (patient_id, profile.model_dump_json()),
        )
        conn.commit()


def _load_patient(patient_id: str) -> Optional[PatientProfile]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT profile_json FROM patients WHERE id = ?", (patient_id,)
        ).fetchone()
    if row is None:
        return None
    return PatientProfile.model_validate_json(row["profile_json"])


def _store_report(
    patient_id: str,
    report: Optional[ClinicalFindingsReport],
    counsellor_output: str,
) -> None:
    report_json = report.model_dump_json() if report else None
    risk = getattr(report, "overall_risk_level", None) if report else None
    degraded = int(getattr(report, "gate_degraded", False)) if report else 0
    with _get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO reports
               (patient_id, report_json, counsellor_output, risk_level, gate_degraded)
               VALUES (?, ?, ?, ?, ?)""",
            (patient_id, report_json, counsellor_output, risk, degraded),
        )
        conn.commit()


def _load_report(patient_id: str) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM reports WHERE patient_id = ?", (patient_id,)
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    if result.get("report_json"):
        result["report"] = json.loads(result.pop("report_json"))
    else:
        result.pop("report_json", None)
        result["report"] = None
    result["gate_degraded"] = bool(result.get("gate_degraded", 0))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI lifespan — init DB on startup
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    logger.info("[main] SQLite cache initialised at %s", _DB_PATH)
    yield
    logger.info("[main] Shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AEGIS API",
    description="Adversarial Engine for Guardian Intelligence in Safe-prescribing",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class IntakeRequest(BaseModel):
    chief_complaint: str
    symptom_duration: str
    symptom_severity: int
    current_medications: list[str] = []
    known_allergies: list[str] = []
    age: int
    weight_kg: float
    antibiotic_in_question: str


class IntakeResponse(BaseModel):
    patient_id: str
    profile: dict


class AnalyzeRequest(BaseModel):
    patient_id: str


# ─────────────────────────────────────────────────────────────────────────────
# SSE helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    """Format a single SSE message."""
    payload = json.dumps({"event": event, **data})
    return f"data: {payload}\n\n"


def _sse_error(message: str) -> str:
    return _sse("error", {"message": message})


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph pipeline runner (streams per-node SSE events)
# ─────────────────────────────────────────────────────────────────────────────

_NODE_DISPLAY = {
    "intake":           "Intake Node",
    "parallel_agents":  "Clinical Agents (parallel)",
    "supervisor_compiler": "Supervisor Agent",
    "safety_gate":      "Safety Gate",
    "counsellor":       "Counsellor Agent",
    "finalizer":        "Finalizer",
}


async def _stream_pipeline(patient_id: str, profile: PatientProfile) -> AsyncGenerator[str, None]:
    """Run aegis_graph.stream() in a thread pool and yield SSE events."""
    from graph import aegis_graph

    yield _sse("pipeline_start", {"patient_id": patient_id, "total_nodes": len(_NODE_DISPLAY)})

    initial_state = AegisState(patient_profile=profile)
    final_state_holder: list = [None]
    error_holder: list = [None]

    def _run_stream():
        events = []
        try:
            for chunk in aegis_graph.stream(initial_state, stream_mode="updates"):
                events.append(chunk)
            final = aegis_graph.invoke(initial_state)
            final_state_holder[0] = final
        except Exception as exc:
            error_holder[0] = str(exc)
        return events

    loop = asyncio.get_running_loop()
    node_events = await loop.run_in_executor(None, _run_stream)

    if error_holder[0]:
        yield _sse_error(error_holder[0])
        return

    # Emit one event per node that ran
    seen_nodes: set = set()
    for chunk in node_events:
        for node_name, delta in chunk.items():
            if node_name.startswith("__"):
                continue
            display = _NODE_DISPLAY.get(node_name, node_name)
            if node_name not in seen_nodes:
                seen_nodes.add(node_name)
                yield _sse("node_complete", {"node": node_name, "display": display})
                await asyncio.sleep(0)  # yield control to event loop

    # Extract final state
    final = final_state_holder[0]
    if final is None:
        yield _sse_error("Pipeline completed but no final state was captured.")
        return

    # Persist to SQLite  (LangGraph returns a plain dict)
    if isinstance(final, dict):
        report = final.get("compiled_report")
        counsellor_output = final.get("counsellor_output", "") or ""
    else:
        report = getattr(final, "compiled_report", None)
        counsellor_output = getattr(final, "counsellor_output", "") or ""

    await loop.run_in_executor(None, _store_report, patient_id, report, counsellor_output)

    # Serialise report for SSE payload (patient-safe: no raw evidence dumps)
    report_payload: dict = {}
    if report is not None:
        report_payload = {
            "overall_risk_level":       getattr(report, "overall_risk_level", None),
            "gate_degraded":            getattr(report, "gate_degraded", False),
            "antibiotic_warranted":     getattr(report, "antibiotic_warranted", None),
            "appropriateness_confidence": getattr(report, "appropriateness_confidence", None),
            "contraindications_found":  getattr(report, "contraindications_found", False),
            "interactions_count":       len(getattr(report, "interactions", [])),
            "dosing_flag":              getattr(report, "dosing_flag", False),
            "recommended_dose_mg":      getattr(report, "recommended_dose_mg", None),
            "frequency":                getattr(report, "frequency", None),
            "duration_days":            getattr(report, "duration_days", None),
        }

    yield _sse("pipeline_complete", {
        "patient_id":      patient_id,
        "counsellor_output": counsellor_output,
        "report_summary":  report_payload,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return {"status": "ok", "model": cfg.LLM_PROVIDER, "model_name": cfg.LLM_MODEL_NAME}


@app.post("/api/v1/intake", response_model=IntakeResponse)
async def intake(body: IntakeRequest):
    """Store a PatientProfile and return a patient_id for subsequent calls."""
    try:
        profile = PatientProfile(**body.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    patient_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _store_patient, patient_id, profile)

    logger.info("[intake] Stored patient %s — %s", patient_id, profile.chief_complaint[:60])
    return IntakeResponse(patient_id=patient_id, profile=profile.model_dump())


@app.post("/api/v1/analyze")
async def analyze(body: AnalyzeRequest):
    """Trigger the full AEGIS pipeline and stream SSE events.

    SSE event types:
      pipeline_start   — pipeline kicked off
      node_complete    — a graph node finished (node, display)
      pipeline_complete — all done; includes counsellor_output + report_summary
      error            — something went wrong
    """
    loop = asyncio.get_event_loop()
    profile = await loop.run_in_executor(None, _load_patient, body.patient_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"patient_id '{body.patient_id}' not found.")

    return StreamingResponse(
        _stream_pipeline(body.patient_id, profile),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/v1/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Extract PatientProfile fields from a prescription or pill photo via vision LLM."""
    # Validate content type
    allowed = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Upload JPG or PNG.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be under 5 MB.")

    vision_prompt = """You are a medical document parser. Extract all readable information from this prescription or pill image:
drug name, dose, frequency, prescribing diagnosis if visible, and any warnings.
Return ONLY a JSON object with these fields (use null for anything illegible):
{
  "antibiotic_in_question": "<drug name or null>",
  "chief_complaint": "<diagnosis/indication or null>",
  "notes": "<any dosing instructions or warnings visible>"
}"""

    loop = asyncio.get_event_loop()
    try:
        raw = await loop.run_in_executor(None, cfg.call_vision_llm, image_bytes, vision_prompt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Vision LLM error: {exc}")

    # Strip markdown fences and parse
    import re
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        extracted = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        extracted = json.loads(match.group()) if match else {"raw": raw}

    # Generate a patient_id so the caller can continue to /intake or /analyze
    patient_id = str(uuid.uuid4())
    logger.info("[upload-image] Extracted fields for %s: %s", patient_id, extracted)

    return {
        "patient_id": patient_id,
        "extracted_data": extracted,
        "filename": file.filename,
    }


@app.get("/api/v1/report/{patient_id}")
async def get_report(patient_id: str):
    """Return the cached ClinicalFindingsReport for a completed analysis."""
    loop = asyncio.get_event_loop()
    row = await loop.run_in_executor(None, _load_report, patient_id)

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No report found for patient_id '{patient_id}'. "
                   "Run /api/v1/analyze first.",
        )

    return {
        "patient_id":       patient_id,
        "risk_level":       row.get("risk_level"),
        "gate_degraded":    row.get("gate_degraded", False),
        "counsellor_output": row.get("counsellor_output", ""),
        "report":           row.get("report"),
        "created_at":       row.get("created_at"),
    }
