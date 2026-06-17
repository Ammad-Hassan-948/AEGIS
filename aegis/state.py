from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class PatientProfile(BaseModel):
    chief_complaint: str
    symptom_duration: str
    symptom_severity: int = Field(..., ge=1, le=10)
    current_medications: List[str] = Field(default_factory=list)
    known_allergies: List[str] = Field(default_factory=list)
    age: int = Field(..., ge=0, le=130)
    weight_kg: float = Field(..., gt=0)
    antibiotic_in_question: str


class DrugInteraction(BaseModel):
    drug_a: str
    drug_b: str
    severity: Literal["mild", "moderate", "severe"]
    description: str
    rag_evidence: str


class ClinicalFindingsReport(BaseModel):
    antibiotic_warranted: bool
    appropriateness_confidence: int = Field(..., ge=0, le=100)
    contraindications_found: bool
    interactions: List[DrugInteraction] = Field(default_factory=list)
    recommended_dose_mg: Optional[float] = None
    frequency: Optional[str] = None
    duration_days: Optional[int] = None
    dosing_flag: bool = False
    overall_risk_level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    rag_evidence_summary: List[str] = Field(default_factory=list)
    # True when the Safety Gate exhausted all retries without passing full validation.
    # The UI uses this to show a "consult a physician in person" banner instead of
    # presenting the partial findings as a confident conclusion.
    gate_degraded: bool = False

    @field_validator("interactions")
    @classmethod
    def interactions_required_when_contraindications_found(
        cls, v: List[DrugInteraction], info: Any
    ) -> List[DrugInteraction]:
        data = info.data if hasattr(info, "data") else {}
        if data.get("contraindications_found") and not v:
            raise ValueError(
                "interactions must not be empty when contraindications_found is True"
            )
        return v


class AegisState(BaseModel):
    # Raw input
    raw_input_text: str = ""
    uploaded_image_analysis: str = ""

    # Structured patient data (populated by Intake Node)
    patient_profile: Optional[PatientProfile] = None

    # Individual agent outputs (raw dicts before Supervisor compiles them)
    appropriateness_findings: Dict[str, Any] = Field(default_factory=dict)
    contraindication_findings: Dict[str, Any] = Field(default_factory=dict)
    dosing_findings: Dict[str, Any] = Field(default_factory=dict)

    # Retrieved RAG chunks for the current run — passed to Safety Gate for
    # citation grounding verification (substring checks against agent rag_evidence fields)
    retrieved_chunks: List[str] = Field(default_factory=list)

    # Compiled and validated output
    compiled_report: Optional[ClinicalFindingsReport] = None
    counsellor_output: str = ""

    # Safety Gate control flow
    validation_errors: List[str] = Field(default_factory=list)
    iteration_count: int = 0

    # LangGraph routing signal written by safety_gate_node, read by conditional edge
    next_step: str = ""

    class Config:
        # Allow arbitrary types so LangGraph can attach extra metadata if needed
        arbitrary_types_allowed = True
