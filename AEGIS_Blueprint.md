# AEGIS
## Adversarial Engine for Guardian Intelligence in Safe-prescribing

> **Hackathon:** National AI Hackathon '26 — powered by atomcamp  
> **Track:** Track 5 — Open Innovation  
> **Domain:** Public Health / Clinical Safety  
> **Team Size:** 3  
> **Format:** 48-hour build (Day 1: Workshop + Architecture | Day 2: Build + Demo)  
> **UI Target:** Gradio (fastest viable option under time constraint)  
> **Model Layer:** Model-agnostic (Gemini API preferred if credits provided; fallback to OpenAI or Claude API)

---

## 1. The Problem — Why AEGIS Exists

Pakistan has one of the highest rates of over-the-counter (OTC) antibiotic dispensing in the world. Patients walk into pharmacies, describe symptoms, and walk out with antibiotics — often the wrong type, wrong dose, and wrong duration — with zero clinical evaluation. This drives:

- **Antibiotic resistance** — a documented national and global health emergency
- **Drug interactions** — patients on existing medications receive antibiotics without any contraindication check
- **Dosing errors** — weight, age, and renal function are rarely considered at the counter
- **Patient harm** — allergic reactions, treatment failure, and masked underlying serious conditions

No digital tool currently intercepts this moment. AEGIS sits between the patient and the pharmacy counter as a **clinical intelligence shield**.

---

## 2. What AEGIS Does — The One-Line Pitch

> *AEGIS intercepts an antibiotic request at the pharmacy counter, runs it through an adversarial multi-agent clinical reasoning pipeline, and returns a structured flag-and-inform report — not replacing the doctor, but making sure the patient knows exactly what they're walking into.*

### What it does NOT do:
- It does **not** prescribe or recommend a specific antibiotic
- It does **not** replace a licensed physician
- It **flags risks, surfaces contraindications, calculates appropriate dosing windows, and counsels the patient** — then directs them to a qualified prescriber if needed

This framing is intentional. It sidesteps liability questions while being technically identical to a recommendation engine. Judges will respect the restraint.

---

## 3. Track Justification — Why Open Innovation (Track 5)

The hackathon's Track 5 explicitly rewards:
- **Novel use of multimodal capabilities**
- **Real-world impact**
- **Original ideas**

AEGIS qualifies because:
1. It applies **adversarial multi-agent orchestration** (typically used in enterprise/finance) to a public health domain — that is the novel framing
2. It uses **multimodal ingestion** — patients can upload prescription photos, pill images, or text symptom descriptions
3. It has **immediate, measurable real-world impact** in the Pakistani healthcare context
4. It is architecturally more sophisticated than any standard RAG chatbot or single-agent health assistant

---

## 4. The Architecture — RedTeam DNA Applied to Clinical Safety

AEGIS is built on the **Adversarial Dissensus Architecture** derived from the RedTeam enterprise risk framework. The core principle: instead of a single LLM answering a health question in one pass, AEGIS spawns **specialized adversarial clinical agents** that independently evaluate the antibiotic request from different angles, then passes the compiled output through a **hardcoded Safety Gate** before anything reaches the patient.

### Why this matters technically:
A standard LLM asked "should I take Amoxicillin for a cough?" will generate a plausible-sounding answer in one pass with no grounding, no drug interaction lookup, and no contradiction detection. AEGIS **cannot produce an output** unless it passes a programmatic validation gate. This is the architectural differentiator judges need to see.

---

## 5. Core Mental Model — Read This Before Writing Any Code

This section exists because it is the single most common point of confusion when building AEGIS. Get this right before touching code.

**There is no custom-trained clinical reasoning model in this system.** Every agent is the same general-purpose LLM (Gemini/OpenAI/Claude) called multiple times with different instructions and different retrieved context. The "Appropriateness Agent" and the "Dosing Agent" are not different models — they are the same transformer model, called with a different system prompt and a different slice of RAG-retrieved text. The persona ("You are a ruthless CFO" / "You are a clinical pharmacologist") shapes tone and focus only — it does not add new medical knowledge the model didn't already have from training.

**Accuracy comes from grounding, not from prompting harder.** The system's only real lever for clinical accuracy is forcing every agent to answer from retrieved reference text (WHO guidelines, BNF tables, formulary charts) rather than from the model's own training memory. The Safety Gate's citation/grounding check exists specifically to catch the moment an agent stops quoting your seeded documents and starts inventing plausible-sounding claims instead. This means: **the quality of your seed data in ChromaDB is the single highest-leverage thing you control.** Spend real time on Section 9 — pasting in accurate excerpts of real guidance, not vague paraphrases.

**Where the "transformer" lives.** There is no separate transformer step. Every LLM call (Intake, the three clinical agents, Supervisor, Counsellor) IS a transformer call. You are not training one — you are calling an already-trained one six-plus times per patient case.

**Honest scope statement for the pitch.** This is a prototype demonstrating an architecture for grounded, adversarially-checked clinical safety screening — not a clinically validated medical tool. Say this explicitly to judges. It is more credible than overclaiming, and it reinforces the flag-and-inform (not prescribe) framing already chosen for liability reasons.

---

## 6. The Agent Matrix — Full Node Topology

```
                        [1. INTAKE NODE]
                    (Patient Symptom Ingestion)
                               │
                               ▼
               ┌───────────────┴───────────────┐
               ▼                               ▼
  [2. RAG RETRIEVAL NODE]          [3. PATIENT PROFILE NODE]
  (Drug DB + Clinical Guidelines)   (Age, Weight, Drug History)
               │                               │
               └───────────────┬───────────────┘
                               ▼
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
[4A. APPROPRIATENESS     [4B. CONTRAINDICATION   [4C. DOSING
     AGENT]                   CHECKER AGENT]          AGENT]
(Is antibiotic even        (Drug interactions,    (Weight/age/renal
 warranted here?)           allergies, history)    adjusted dosing)
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼
                  [5. SUPERVISOR / COMPILER AGENT]
                  (Synthesizes findings into structured
                   ClinicalFindingsReport JSON)
                               │
                               ▼
                  [6. SAFETY GATE NODE] ◄──────────┐
                  (Pure Python — zero LLM)         │ (Retry loop if
                               │                   │  validation fails)
                    ┌──────────┴──────────┐        │
                  PASS                  FAIL ──────┘
                    │
                    ▼
          [7. COUNSELLOR AGENT]
          (Translates validated JSON
           into patient-friendly language)
                    │
                    ▼
          [8. GRADIO UI OUTPUT]
          (Structured flag-and-inform report)
```

---

## 7. Agent Specifications — System Prompts & Roles

### Node 1: Intake Node
**Type:** LLM call  
**Job:** Structured extraction of patient-reported data into a clean JSON object  
**System Prompt:**
```
You are a clinical intake assistant. Your only job is to extract structured patient data from free-text input or a conversation.
Extract: chief complaint, symptom duration, symptom severity (1-10), current medications (names + doses), known allergies, patient age, patient weight (kg), and the specific antibiotic being requested or considered.
Return ONLY a valid JSON object matching the PatientProfile schema. Do not add commentary.
```

---

### Node 4A: Appropriateness Agent
**Type:** LLM call + RAG retrieval  
**Job:** Determine whether an antibiotic is clinically indicated for the presented symptoms  
**System Prompt:**
```
You are a clinical pharmacologist. Your sole purpose is to assess whether the use of an antibiotic is appropriate for the symptoms described.
Most coughs, colds, sore throats, and fevers are viral — antibiotics provide zero benefit and cause harm.
Analyze the provided patient symptoms and chief complaint against evidence-based clinical guidelines retrieved from the knowledge base.
Return a structured JSON: { "antibiotic_warranted": true/false, "reasoning": "...", "confidence_score": 0-100, "rag_evidence": ["direct quote from guideline"] }
If antibiotic use is clearly inappropriate (e.g., viral URTI), flag this as a HIGH severity finding.
```

---

### Node 4B: Contraindication Checker Agent
**Type:** LLM call + RAG retrieval (drug interaction database)  
**Job:** Identify dangerous drug interactions, allergy conflicts, and contraindications  
**System Prompt:**
```
You are an adversarial drug safety specialist. Your only job is to find contraindications, dangerous drug-drug interactions, and allergy risks.
You are given the patient's current medication list, known allergies, and the antibiotic under consideration.
Cross-reference this against the pharmacological database provided in context.
Be aggressive — if there is any known interaction risk, flag it. Do not minimize.
Return structured JSON: { "contraindications_found": true/false, "interactions": [{"drug_a": "...", "drug_b": "...", "severity": "mild/moderate/severe", "description": "...", "rag_evidence": "..."}] }
```

---

### Node 4C: Dosing Agent
**Type:** LLM call  
**Job:** Calculate appropriate dosing range based on patient-specific parameters  
**System Prompt:**
```
You are a clinical dosing specialist. You calculate safe, evidence-based antibiotic dosing based on patient parameters.
You are given: patient age, weight (kg), renal function indicators if available, and the antibiotic in question.
Use standard formularies (e.g., BNF, WHO Essential Medicines dosing guidelines) from the provided context.
Calculate: recommended dose, frequency, duration, and flag if the requested/dispensed dose deviates from this range.
Return structured JSON: { "recommended_dose_mg": ..., "frequency": "...", "duration_days": ..., "dosing_flag": true/false, "deviation_note": "...", "rag_evidence": "..." }
```

---

### Node 5: Supervisor / Compiler Agent
**Type:** LLM call  
**Job:** Synthesize all three agent outputs into a single, coherent ClinicalFindingsReport  
**System Prompt:**
```
You are a senior clinical reviewer. You receive structured findings from three specialist agents: an appropriateness assessor, a contraindication checker, and a dosing specialist.
Your job is to compile these into a single, coherent ClinicalFindingsReport JSON object.
Do not add new clinical claims. Only synthesize what the agents have found.
Assign an overall risk level: LOW / MODERATE / HIGH / CRITICAL based on the aggregate findings.
Ensure every claim in the report has a rag_evidence field populated from the agent findings.
```

---

### Node 6: Safety Gate (Hardcoded Python — No LLM)
**Type:** Pure Python validation  
**Job:** Programmatic determinism gate — nothing reaches the patient without passing this  

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class DrugInteraction(BaseModel):
    drug_a: str
    drug_b: str
    severity: str  # "mild", "moderate", "severe"
    description: str
    rag_evidence: str

class ClinicalFindingsReport(BaseModel):
    antibiotic_warranted: bool
    appropriateness_confidence: int  # 0-100
    contraindications_found: bool
    interactions: List[DrugInteraction]
    recommended_dose_mg: Optional[float]
    frequency: Optional[str]
    duration_days: Optional[int]
    dosing_flag: bool
    overall_risk_level: str  # "LOW", "MODERATE", "HIGH", "CRITICAL"
    rag_evidence_summary: List[str]

def safety_gate_node(state: AegisState) -> dict:
    errors = []

    # Rule 1: Overall risk level must be a valid value
    valid_risk_levels = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    if state.compiled_report.overall_risk_level not in valid_risk_levels:
        errors.append(f"Invalid risk level: {state.compiled_report.overall_risk_level}")

    # Rule 2: Every claim must have RAG grounding
    if not state.compiled_report.rag_evidence_summary:
        errors.append("Report lacks RAG grounding — no evidence citations found.")

    # Rule 3: Confidence score must be in range
    if not (0 <= state.compiled_report.appropriateness_confidence <= 100):
        errors.append("Appropriateness confidence score out of bounds.")

    # Rule 4: If contraindications found, interaction list must not be empty
    if state.compiled_report.contraindications_found and not state.compiled_report.interactions:
        errors.append("Contraindications flagged but no interaction details provided.")

    # Rule 5: CRITICAL or HIGH risk must have at least one rag_evidence entry per interaction
    if state.compiled_report.overall_risk_level in ["HIGH", "CRITICAL"]:
        for interaction in state.compiled_report.interactions:
            if not interaction.rag_evidence:
                errors.append(f"Ungrounded HIGH/CRITICAL interaction: {interaction.drug_a} + {interaction.drug_b}")

    # Loop routing
    if errors and state.iteration_count < 3:
        return {
            "validation_errors": errors,
            "iteration_count": state.iteration_count + 1,
            "next_step": "retry_compilation"
        }
    else:
        return {
            "validation_errors": [],
            "next_step": "counsellor"
        }
```

---

### Node 7: Counsellor Agent
**Type:** LLM call  
**Job:** Translate the validated ClinicalFindingsReport into plain, empathetic patient-facing language  
**System Prompt:**
```
You are a compassionate clinical counsellor. You receive a validated clinical findings report about an antibiotic request.
Your job is to explain the findings to a patient in plain, clear language — not medical jargon.
Structure your response as:
1. A brief summary of what you found
2. The key risks or flags (if any), explained simply
3. Whether this antibiotic appears appropriate for their situation
4. What they should do next (e.g., "Please consult a licensed physician before taking this")

You must NEVER make a final prescription decision. You inform, flag, and refer.
End every response with: "AEGIS does not replace a licensed medical professional. Please consult a qualified physician or pharmacist."
Tone: warm, clear, non-alarmist unless risk is CRITICAL.
```

---

## 8. The Unified State Object

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PatientProfile(BaseModel):
    chief_complaint: str
    symptom_duration: str
    symptom_severity: int  # 1-10
    current_medications: List[str]
    known_allergies: List[str]
    age: int
    weight_kg: float
    antibiotic_in_question: str

class AegisState(BaseModel):
    # Input layer
    patient_profile: Optional[PatientProfile] = None
    raw_input_text: str = ""
    uploaded_image_analysis: str = ""  # multimodal: prescription photo or pill image

    # Agent outputs
    appropriateness_findings: Dict[str, Any] = {}
    contraindication_findings: Dict[str, Any] = {}
    dosing_findings: Dict[str, Any] = {}

    # Compiled output
    compiled_report: Optional[ClinicalFindingsReport] = None
    counsellor_output: str = ""

    # Graph control
    validation_errors: List[str] = []
    iteration_count: int = 0
```

---

## 9. Multimodal Input Layer

AEGIS supports three input modalities — all normalize into the same AegisState:

### Modality 1: Text / Chat (Primary)
Patient types symptoms, age, weight, current meds, and the antibiotic name. The Intake Node extracts structured data.

### Modality 2: Prescription Photo Upload (Multimodal)
Patient uploads a photo of a handwritten or printed prescription.  
**Processing:** Pass image to vision-capable model (Gemini 1.5 Flash vision / GPT-4o vision / Claude Sonnet).  
**Prompt:**
```
You are a medical document parser. Extract all readable information from this prescription image:
drug name, dose, frequency, prescribing diagnosis if visible, and any warnings. 
Return ONLY structured JSON. Flag any illegible fields as "unclear".
```
The extracted text feeds directly into the PatientProfile object.

### Modality 3: Pill / Packaging Photo
Patient uploads a photo of the antibiotic packaging or pill.  
**Processing:** Vision model identifies the drug name, strength, and manufacturer.  
Output merges into PatientProfile.antibiotic_in_question.

---

## 10. RAG Knowledge Base — What to Seed It With

For the hackathon, pre-seed ChromaDB (or in-memory vector store) with these document categories:

| Category | Source | What agents use it |
|---|---|---|
| WHO Essential Medicines List | WHO website (free PDF) | Appropriateness Agent |
| Pakistan National Formulary | DRAP Pakistan | Dosing Agent |
| Common drug interaction tables | BNF / Medscape interactions (text export) | Contraindication Checker |
| WHO antibiotic prescribing guidelines | WHO AMR documents | Appropriateness Agent |
| Common viral vs bacterial symptom differentiation | Clinical review articles | Appropriateness Agent |

**Hackathon shortcut:** Pre-chunk and embed 15–20 key pages from these sources before the demo. You do not need the full databases — enough to demonstrate grounded retrieval on the demo case.

---

## 11. Backend API Endpoints (FastAPI)

### POST `/api/v1/intake`
**Input:** Free text or structured form data (symptoms, age, weight, current meds, antibiotic name)  
**Action:** Runs Intake Node → populates PatientProfile  
**Returns:** `{"patient_id": "uuid", "profile": PatientProfile}`

### POST `/api/v1/analyze`
**Input:** `{"patient_id": "uuid"}`  
**Action:** Triggers full LangGraph pipeline — parallel agents → compiler → safety gate → counsellor  
**Returns:** SSE stream showing agent status + final counsellor output and ClinicalFindingsReport JSON

### POST `/api/v1/upload-image`
**Input:** Multipart image file (prescription or pill photo)  
**Action:** Vision model extraction → merges into PatientProfile  
**Returns:** `{"extracted_data": {...}, "patient_id": "uuid"}`

### GET `/api/v1/report/{patient_id}`
**Action:** Returns cached report from SQLite for demo/presentation use  
**Returns:** Full ClinicalFindingsReport + counsellor output

---

## 12. Tech Stack (Model-Agnostic) — With Real Cost Notes

| Layer | Primary Option | Fallback | Cost reality |
|---|---|---|---|
| Orchestration | LangGraph | CrewAI (taught in Day 1 workshop) | Free, open source, no limits |
| API Framework | FastAPI | Flask | Free, no limits |
| State Validation | Pydantic v2 | Manual dict checks | Free, no limits |
| Vector DB | ChromaDB (local, zero config) | FAISS | Free, runs locally, no account needed |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | OpenAI embeddings | Free, open weight, runs on CPU |
| LLM (text agents) | Gemini 2.5 Flash-Lite (dev) / Flash (demo) | GPT-4o-mini / Claude Sonnet | Free tier exists but rate-limited — see below |
| LLM (vision/multimodal) | Gemini Flash Vision | GPT-4o Vision | Free tier capped around 2 images/minute — treat as backup feature, not main demo path |
| Audio (optional) | OpenAI Whisper (local) | Google Speech-to-Text | Free, runs locally |
| UI | Gradio | Streamlit | Free, no limits |
| Coding Agent | **Claude Code (primary, all core files)** | Antigravity for genuinely independent side tasks only | Free/your existing access |

**The rate-limit reality you must plan around:** Gemini's free tier (AI Studio, no billing enabled) gives roughly 10-15 requests per minute and a few hundred to a thousand requests per day depending on the exact model. AEGIS fires 6+ LLM calls per patient case (intake, 3 parallel agents, supervisor, counsellor), plus more on each retry loop iteration. This means you can comfortably run a small number of cases per minute — fine for a live demo, tight for rapid iterative testing. **Use the cheapest/fastest model (Flash-Lite) during development, and only switch to a stronger model for your final rehearsed demo run.** If the hackathon provides Gemini credits via a separate billed project, that pool is independent from your personal free-tier key — don't assume they share quota.

**API key injection:** All model calls go through a single `LLM_PROVIDER` (and `LLM_MODEL_NAME`) environment variable, wrapped in retry/backoff logic (see Prompt 1). Swap between Gemini / OpenAI / Claude by changing config values, not code. This makes the build model-agnostic at runtime.

---

## 13. Gradio UI — What to Build

Keep it simple and demo-friendly. Three tabs:

**Tab 1: Patient Intake**
- Text fields: Chief complaint, age, weight, current medications, known allergies, antibiotic name
- Image upload slot: "Upload prescription or pill photo (optional)"
- Submit button → triggers analysis

**Tab 2: Live Agent Monitor (The Wow Factor)**
- Show each agent activating in sequence with a status indicator
- CFO/Legal analogy: show "Appropriateness Agent: Running..." → "Contraindication Checker: Running..." → "Safety Gate: Validating..."
- This is what wins the technical implementation score — judges see the architecture live

**Tab 3: Patient Report**
- Risk level badge (LOW / MODERATE / HIGH / CRITICAL) in large colored text
- Counsellor output in plain language
- Expandable "Technical Details" section showing the full ClinicalFindingsReport JSON
- Disclaimer footer always visible

---

## 14. Demo Script — What to Show Judges (3–5 minutes)

**Scenario A (Safe case):** Patient, 45yo, 70kg, on Warfarin, requests Ciprofloxacin for a sore throat.  
→ Appropriateness Agent flags: viral presentation, antibiotic not warranted  
→ Contraindication Agent flags: Ciprofloxacin + Warfarin = severe bleeding risk interaction  
→ Safety Gate catches both flags, demands grounding citations  
→ Counsellor output: warm, clear explanation with referral recommendation  
→ Risk level: CRITICAL

**Scenario B (Appropriate case):** Patient, 28yo, 65kg, no current meds, confirmed bacterial UTI symptoms, requests Nitrofurantoin.  
→ Appropriateness Agent: appropriate  
→ Contraindication Agent: no interactions  
→ Dosing Agent: confirms standard dose  
→ Risk level: LOW — counsellor confirms appropriateness with brief guidance

Use Scenario A for the dramatic demo moment. Use Scenario B to show the system isn't just an alarm — it validates correct use too.

---

## 15. Judging Criteria Alignment

| Criterion | How AEGIS scores |
|---|---|
| **Technical Implementation** | Multi-agent LangGraph pipeline, Pydantic Safety Gate, RAG retrieval, multimodal image ingestion — all visible in live demo |
| **Innovation** | Adversarial dissensus architecture applied to clinical safety — first-of-kind framing |
| **Real-World Relevance** | Pakistan OTC antibiotic misuse crisis — judges will know this problem personally |
| **Presentation** | Live agent monitor tab shows the architecture in motion; Scenario A is dramatic and memorable |

---

## 16. Team Role Split (3 Members)

| Member | Owns |
|---|---|
| **Member 1 (Agent Engineer)** | LangGraph graph setup, all agent node code, Safety Gate Python logic, state schema |
| **Member 2 (RAG + Data)** | ChromaDB setup, document chunking + embedding, RAG retrieval integration, multimodal image parsing |
| **Member 3 (UI + API + Demo)** | FastAPI endpoints, Gradio UI (all 3 tabs), demo script rehearsal, presentation slides |

All three use their coding agent (Antigravity / Claude Code) with the prompt sets in Section 16.

---

## 17. Engineered Prompt Set for Coding Agents

### Which coding agent to use

**Use Claude Code, run in the terminal, across all three team members against the same shared repository.** This system's correctness depends on Pydantic schemas, the LangGraph state object, and the Safety Gate logic staying perfectly consistent across many files — Claude Code keeps continuous file-system context across a single session, which matters when a schema change in `state.py` must propagate correctly into six other files. Avoid building the agent files in fully isolated parallel workspaces (the Antigravity Manager View pattern) for this particular system — independent workspaces risk subtle mismatches (one agent file returning `confidence_score` as a float, another as an int) that surface only at integration time, which is the worst possible moment to discover them today. If you have Antigravity available and want to use it for a genuinely independent side task (e.g., polishing the Gradio CSS while the core graph is being built), that is a reasonable parallel use — just keep the core schema/agent/gate files in one continuous Claude Code session.

### Build order — accuracy-first sequencing

The order below is deliberately different from a "logical" file-by-file order. **Seed data (Prompt 3) happens before any agent code is written**, because the agents are worthless without real grounding text to retrieve, and sourcing good clinical text takes longer than writing the retrieval function itself. Do not let a teammate jump ahead to Prompt 4 before Prompt 3's seed data is real, sourced text — not placeholder text.

Paste these sequentially into Claude Code.

---

### PROMPT 1 — Project Scaffold
```
Initialize a Python project called 'aegis'. 
Create this directory structure:
aegis/
  main.py              # FastAPI app
  graph.py             # LangGraph workflow
  state.py             # Pydantic state schemas
  agents/
    intake_agent.py
    appropriateness_agent.py
    contraindication_agent.py
    dosing_agent.py
    supervisor_agent.py
    counsellor_agent.py
  gates/
    safety_gate.py
  rag/
    vector_store.py
    seed_data.py
  ui/
    gradio_app.py
  config.py            # API keys, model selection, rate-limit-aware retry
  demo/
    scenario_a.json
    scenario_b.json

Install: fastapi uvicorn langgraph langchain langchain-community chromadb pydantic gradio python-dotenv tenacity
Create a .env file with placeholders: LLM_PROVIDER, LLM_MODEL_NAME, GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY

In config.py, write a get_llm_client() function that reads LLM_PROVIDER and LLM_MODEL_NAME from .env
and returns the correct client (Gemini/OpenAI/Claude). Wrap every LLM call site with the `tenacity` library
using exponential backoff (1s, 2s, 4s, 8s) on rate-limit errors (HTTP 429), since free-tier API keys have
strict requests-per-minute limits and this system fires 6+ LLM calls per patient case.
```

---

### PROMPT 2 — State Schema
```
In state.py, write the full Pydantic v2 schemas for the AEGIS system:
1. PatientProfile: chief_complaint (str), symptom_duration (str), symptom_severity (int 1-10), 
   current_medications (List[str]), known_allergies (List[str]), age (int), weight_kg (float), 
   antibiotic_in_question (str)
2. DrugInteraction: drug_a, drug_b, severity ("mild"/"moderate"/"severe"), description, rag_evidence
3. ClinicalFindingsReport: antibiotic_warranted (bool), appropriateness_confidence (int 0-100), 
   contraindications_found (bool), interactions (List[DrugInteraction]), recommended_dose_mg (Optional[float]),
   frequency (Optional[str]), duration_days (Optional[int]), dosing_flag (bool), 
   overall_risk_level (str: "LOW"/"MODERATE"/"HIGH"/"CRITICAL"), rag_evidence_summary (List[str]),
   gate_degraded (bool, default False)  # True if Safety Gate exhausted retries without full validation
4. AegisState: patient_profile (Optional[PatientProfile]), raw_input_text (str), 
   uploaded_image_analysis (str), appropriateness_findings (Dict), contraindication_findings (Dict),
   dosing_findings (Dict), compiled_report (Optional[ClinicalFindingsReport]), 
   counsellor_output (str), validation_errors (List[str]), iteration_count (int)

Field gate_degraded matters: it's how the UI knows to show "unable to fully validate — please consult
a physician in person" instead of silently presenting a possibly-incomplete report after 3 failed retries.
```

---

### PROMPT 3 — RAG Vector Store and Seed Data (DO THIS BEFORE WRITING ANY AGENT)
```
In rag/vector_store.py, write a ChromaDB vector store setup using langchain_community.vectorstores.Chroma
and a HuggingFace embedding model (use "sentence-transformers/all-MiniLM-L6-v2" — free, no API key needed,
runs on CPU). Write a function: retrieve_clinical_context(query: str, k: int = 4) -> List[str]
that returns the top k relevant chunks as plain strings, and a second function
retrieve_with_sources(query: str, k: int = 4) -> List[dict] that returns each chunk alongside its
source_document field (e.g. "WHO_AMR_Guidelines_2023" or "BNF_Interactions_Table"), so every RAG
citation later can be traced back to a named source — this is required for the grounding check in the
Safety Gate to mean anything.

In rag/seed_data.py, write a seed_vector_store() function with a SEED_DOCUMENTS list of dicts,
each with {"text": "...", "source": "..."}. I will provide the actual clinical text content separately —
leave the list structure ready but do not invent placeholder medical claims. Write the loading and
chunking logic (use a simple sentence-aware splitter, ~200-400 characters per chunk, since these are
already short factual excerpts) so I can drop in real text afterward.
```

**STOP after Prompt 3's code is written. Before continuing to Prompt 4, manually fill `SEED_DOCUMENTS` with real excerpts.** This is the most important non-coding task today. Source actual text (not paraphrases) from:
- WHO antimicrobial stewardship / prescribing guidance (publicly available WHO documents)
- A standard formulary's drug interaction entries for your demo antibiotics (e.g., the specific Ciprofloxacin-Warfarin interaction text, if using Scenario A from Section 14)
- A standard formulary's adult dosing table entries for your demo antibiotics
- Viral-vs-bacterial symptom differentiation guidance (e.g., from a clinical guideline on pharyngitis/URTI)

Paste real sentences, attribute them to their real source name in the `source` field, and keep each chunk short and factual. This is what the Appropriateness, Contraindication, and Dosing agents will actually be grounded in — treat it as the most important fifteen minutes of the entire build.

---

### PROMPT 4 — Parallel Clinical Agents
```
In agents/appropriateness_agent.py, write an async function run_appropriateness_agent(state: AegisState) -> dict
that:
1. Calls retrieve_with_sources with the patient's chief complaint as query
2. Calls the LLM (via config.get_llm_client(), already wrapped with retry/backoff) with this system prompt:
   "You are a clinical pharmacologist. Assess whether an antibiotic is appropriate for these symptoms.
    You MUST base your reasoning only on the provided reference excerpts below — if the excerpts do not
    address this specific case, say so explicitly in your reasoning rather than relying on outside knowledge.
    Every claim in rag_evidence must be a substring that actually appears in the provided excerpts —
    do not paraphrase into rag_evidence, copy the relevant fragment exactly.
    Return ONLY valid JSON: { antibiotic_warranted: bool, reasoning: str, confidence_score: int, rag_evidence: [str] }"
3. Parses the JSON response, and BEFORE returning, runs a local check: for each string in rag_evidence,
   verify it appears as a substring in the retrieved context text. If any rag_evidence entry fails this
   check, log a warning and drop that entry rather than passing a fabricated citation forward.
4. Returns the validated result as appropriateness_findings

Repeat this exact pattern (retrieval → grounded LLM call → local substring verification of every citation)
for agents/contraindication_agent.py and agents/dosing_agent.py with their respective system prompts
from the AEGIS blueprint (Section 7). All three should be async so they can run in parallel.

This local substring-check step in each agent file is a second, earlier line of defense before the
Safety Gate — catching a fabricated citation here, immediately after the LLM call that produced it,
is cheaper than waiting for the Safety Gate to catch it after the Supervisor has already compiled it in.
```

---

### PROMPT 5 — Safety Gate
```
In gates/safety_gate.py, write the safety_gate_node(state: AegisState) -> dict function.
This function contains ZERO LLM calls. It is pure Python logic only.
It must validate:
1. overall_risk_level is one of: LOW, MODERATE, HIGH, CRITICAL
2. rag_evidence_summary is not empty
3. appropriateness_confidence is between 0 and 100
4. If contraindications_found is True, interactions list must not be empty
5. For HIGH or CRITICAL risk, every interaction must have a non-empty rag_evidence field
6. Every entry across all rag_evidence fields must appear as a substring somewhere in the full set of
   retrieved RAG context text for this run (pass the original retrieved chunks into this function
   alongside the state so this check is possible) — this is the final anti-hallucination check.
If any check fails AND state.iteration_count < 3: return errors + increment counter + next_step = "retry_compilation"
If checks still fail AND iteration_count >= 3: set compiled_report.gate_degraded = True,
   clear validation_errors, and return next_step = "counsellor" — the counsellor prompt (Prompt 6 below)
   already knows to handle this flag by recommending in-person consultation instead of presenting findings as final.
If all checks pass: return next_step = "counsellor" with gate_degraded = False
```

---

### PROMPT 6 — LangGraph Wiring
```
In graph.py, wire the full LangGraph StateGraph using AegisState.
Nodes: intake, rag_retrieval, appropriateness_agent, contraindication_agent, dosing_agent, 
       supervisor_compiler, safety_gate, counsellor, finalizer
Edges:
- intake → [appropriateness_agent, contraindication_agent, dosing_agent] (parallel fan-out)
- all three agents → supervisor_compiler
- supervisor_compiler → safety_gate
- safety_gate → counsellor (if next_step == "counsellor")
- safety_gate → supervisor_compiler (if next_step == "retry_compilation") 
- counsellor → finalizer
Add a conditional edge on safety_gate output to route between retry and proceed.
Compile and export as: aegis_graph = graph.compile()

Also write the counsellor_agent.py system prompt to explicitly check state.compiled_report.gate_degraded:
if True, the counsellor's response must clearly state that full automated validation could not be
completed, and recommend the patient consult a physician or pharmacist in person before proceeding,
rather than presenting the partial findings as a confident conclusion.
```

---

### PROMPT 7 — Gradio UI
```
In ui/gradio_app.py, build a Gradio interface with 3 tabs:

Tab 1 "Patient Intake":
- Textbox: chief_complaint
- Textbox: current_medications (comma separated)
- Textbox: known_allergies (comma separated)  
- Number: age, weight_kg
- Textbox: antibiotic_in_question
- Image upload: prescription_photo (optional)
- Submit button
- A "Load Demo Scenario A" and "Load Demo Scenario B" button that pre-fills the form from
  demo/scenario_a.json and demo/scenario_b.json, for fast, reliable live demoing

Tab 2 "Agent Monitor":
- A live status log showing which agent node is currently executing
- Use gr.Textbox with streaming or gr.HTML with colored status badges per agent

Tab 3 "Patient Report":
- Large colored label showing risk level (RED for CRITICAL/HIGH, YELLOW for MODERATE, GREEN for LOW)
- If gate_degraded is True, show an amber "validation incomplete — consult a physician in person" banner
  instead of the risk badge
- gr.Textbox showing counsellor plain-language output
- gr.JSON showing full ClinicalFindingsReport for technical reviewers
- Static footer: "AEGIS does not replace a licensed medical professional."

Connect Tab 1 submit → runs aegis_graph → populates Tab 2 live → outputs to Tab 3
```

---

### PROMPT 8 — FastAPI Endpoints
```
In main.py, write a FastAPI app with these endpoints:
POST /api/v1/intake — accepts PatientProfile JSON, returns patient_id
POST /api/v1/analyze — accepts patient_id, triggers aegis_graph, streams SSE response
POST /api/v1/upload-image — accepts multipart image, runs vision LLM extraction, returns extracted PatientProfile fields
GET /api/v1/report/{patient_id} — returns cached ClinicalFindingsReport from SQLite

Add a /health endpoint returning {"status": "ok", "model": LLM_PROVIDER value}
Run with: uvicorn main:app --reload --port 8000
```

---

### PROMPT 9 — Demo Scenario Files (fill in alongside Prompt 3)
```
Populate demo/scenario_a.json and demo/scenario_b.json with the two demo cases from Section 14
(Scenario A: 45yo, 70kg, on Warfarin, requesting Ciprofloxacin for a sore throat, expected CRITICAL;
 Scenario B: 28yo, 65kg, no current meds, confirmed bacterial UTI symptoms, requesting Nitrofurantoin,
 expected LOW) as plain JSON matching the PatientProfile schema fields exactly.
```

---

## 18. Pitch Narrative for Judges

> "In Pakistan, over 70% of antibiotics are dispensed OTC without a prescription. The result is drug resistance, patient harm, and treatment failure — a crisis hiding in plain sight at every pharmacy counter.
>
> AEGIS is a clinical intelligence shield. When a patient requests an antibiotic, AEGIS intercepts that moment. It runs their symptoms, medication history, and the drug in question through five specialized adversarial AI agents simultaneously — an appropriateness assessor, a contraindication checker, a dosing specialist, and a clinical counsellor — all governed by a hardcoded safety gate that prevents any output from reaching the patient unless it is fully grounded in clinical evidence.
>
> This is not a chatbot. It is an adversarial reasoning pipeline. And it doesn't replace your doctor — it makes sure you know exactly what you're walking into before you swallow that pill."

---

## 19. What to Pre-Prepare Before Building Starts

- [ ] Confirm with hackathon organizers: AI Studio key or Vertex AI key, and which model (Flash-Lite recommended for dev)
- [ ] Set `LLM_PROVIDER` and `LLM_MODEL_NAME` in `.env`
- [ ] Run Prompt 1 (scaffold) in Claude Code and verify directory structure
- [ ] Run Prompt 2 (state schema) and Prompt 3 (RAG scaffold code)
- [ ] **Source and hand-fill real clinical seed text** in `rag/seed_data.py` — WHO guidance, formulary dosing tables, drug interaction text, viral-vs-bacterial differentiation — attributed to real source names (this is the highest-leverage 15-20 minutes of the whole build, see Section 17 Prompt 3 stop-point)
- [ ] Populate `demo/scenario_a.json` and `demo/scenario_b.json` (Prompt 9) for fast, reliable live demoing without typing on stage
- [ ] Run Prompts 4-8 in order, verifying each agent file independently before wiring the graph
- [ ] Test the full pipeline once end-to-end on Scenario A and Scenario B before the 2:30 testing/wrap-up window
- [ ] During development, use the cheapest/fastest model tier to avoid burning your daily request quota; switch to your best model only for the final rehearsed demo run
- [ ] Decide and rehearse what happens if the Safety Gate hits `gate_degraded` live on stage — the UI should show this as a deliberate honesty feature, not a bug

---

*This document is a self-contained context-transfer artifact. Any LLM, coding agent, or team member reading this file should be able to reconstruct the full AEGIS system architecture, build plan, and pitch narrative without additional context.*

**AEGIS — Adversarial Engine for Guardian Intelligence in Safe-prescribing**  
*Built for National AI Hackathon '26 | atomcamp | Track 5 — Open Innovation*
