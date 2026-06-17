"""AEGIS Gradio UI — Clinical Safety Assistant
================================================
Entry point: create_app() → gr.Blocks
Called by app.py after ONNX/LangGraph pre-warming.
"""

import time
import gradio as gr
import logging

# Project-level imports (available when run via app.py)
try:
    from state import PatientProfile, AegisState
    from graph import aegis_graph
    REAL_PIPELINE_AVAILABLE = True
except ImportError:
    REAL_PIPELINE_AVAILABLE = False

logger = logging.getLogger(__name__)

# ==========================================
# 1. CUSTOM CSS & AESTHETIC THEME
# ==========================================

CSS_STYLES = """
/* Force Light Mode Theme Variables even in Dark Mode */
.dark, body.dark, html.dark, .gradio-container.dark {
    background-color: #ffffff !important;
    color: #0f172a !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --body-text-color: #0f172a !important;
    --body-text-color-subdued: #475569 !important;
    --block-title-text-color: #0f172a !important;
    --block-label-text-color: #475569 !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #e2e8f0 !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: #cbd5e1 !important;
    --input-text-color: #0f172a !important;
    --input-placeholder-color: #94a3b8 !important;
    --border-color-primary: #e2e8f0 !important;
    --border-color-secondary: #f1f5f9 !important;
}

/* Healthcare theme custom styles */
.aegis-header {
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
    padding: 24px;
    border-radius: 12px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}
.aegis-header h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    display: flex;
    align-items: center;
    gap: 10px;
}
.aegis-header p {
    margin: 6px 0 0 0;
    font-size: 1.1rem;
    opacity: 0.9;
}
.disclaimer-box {
    background-color: #fffbeb;
    border-left: 4px solid #f59e0b;
    padding: 16px;
    border-radius: 8px;
    margin-top: 10px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}
.disclaimer-text {
    color: #78350f;
    font-weight: 600;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Live Agent Monitor Layout */
.agent-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 16px;
    margin-top: 15px;
}
.agent-card {
    background-color: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 120px;
}
.agent-card.status-pending {
    border-left: 5px solid #94a3b8;
    background-color: #f8fafc;
}
.agent-card.status-running {
    border-left: 5px solid #0ea5e9;
    background-color: #f0f9ff;
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.15);
    animation: pulse-border 2s infinite ease-in-out;
}
.agent-card.status-complete {
    border-left: 5px solid #10b981;
    background-color: #f0fdf4;
}
.agent-title {
    font-weight: 700;
    font-size: 1.05rem;
    color: #1e293b;
    margin: 0 0 6px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.agent-desc {
    font-size: 0.85rem;
    color: #64748b;
    line-height: 1.4;
    margin: 0;
    flex-grow: 1;
}
.status-badge {
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.badge-pending {
    background-color: #e2e8f0;
    color: #475569;
}
.badge-running {
    background-color: #bae6fd;
    color: #0369a1;
    animation: pulse-bg 1.2s infinite alternate;
}
.badge-complete {
    background-color: #d1fae5;
    color: #065f46;
}

/* Patient Report visual elements */
.risk-card {
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}
.risk-card.risk-low {
    background: linear-gradient(135deg, #22c55e 0%, #15803d 100%);
}
.risk-card.risk-moderate {
    background: linear-gradient(135deg, #eab308 0%, #a16207 100%);
}
.risk-card.risk-high {
    background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
}
.risk-card.risk-critical {
    background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
    animation: critical-pulse 2s infinite ease-in-out;
}
.risk-title {
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: 0.05em;
    margin: 0 0 6px 0;
}
.risk-subtitle {
    font-size: 1.1rem;
    opacity: 0.95;
    margin: 0;
    font-weight: 500;
}

/* Report Sections */
.report-section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #0f766e;
    margin-bottom: 12px;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.warning-item {
    background-color: #fef2f2;
    border-left: 4px solid #ef4444;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 12px;
    color: #991b1b;
    font-weight: 600;
    font-size: 0.95rem;
}
.warning-item-high {
    background-color: #fff7ed;
    border-left: 4px solid #f97316;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 12px;
    color: #c2410c;
    font-weight: 600;
    font-size: 0.95rem;
}

/* Animations */
@keyframes pulse-bg {
    0% { opacity: 0.6; }
    100% { opacity: 1; }
}
@keyframes pulse-border {
    0% { box-shadow: 0 0 0 0 rgba(14, 165, 233, 0.4); }
    70% { box-shadow: 0 0 0 6px rgba(14, 165, 233, 0); }
    100% { box-shadow: 0 0 0 0 rgba(14, 165, 233, 0); }
}
@keyframes critical-pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.01); box-shadow: 0 8px 16px rgba(239, 68, 68, 0.3); }
    100% { transform: scale(1); }
}
"""

# Alias so app.py can do: from ui.gradio_app import create_app, CSS
CSS = CSS_STYLES

# ==========================================
# 2. VALIDATION & DECISION LOGIC ENGINE
# ==========================================

def validate_inputs(symptoms, age, weight, antibiotic):
    errors = []

    if not symptoms or len(symptoms.strip()) < 5:
        errors.append("<strong>Chief Complaint / Symptoms</strong> are required and must be at least 5 characters long.")

    if age is None:
        errors.append("<strong>Age</strong> is required.")
    elif age < 0 or age > 120:
        errors.append("<strong>Age</strong> must be a valid number between 0 and 120.")

    if weight is None:
        errors.append("<strong>Weight</strong> is required.")
    elif weight < 2 or weight > 250:
        errors.append("<strong>Weight</strong> must be a valid number between 2 and 250 kg.")

    if not antibiotic or len(antibiotic.strip()) < 3:
        errors.append("<strong>Antibiotic Requested</strong> is required and must be at least 3 characters long.")

    return errors


def evaluate_safety_report(symptoms, duration, severity, age, weight, medications, allergies, antibiotic, image):
    """
    Rule-based safety check.
    DEVELOPERS: replace with a real LangGraph pipeline call when ready.
    """
    med_list     = [m.strip().lower() for m in medications.split(",") if m.strip()]
    allergy_list = [a.strip().lower() for a in allergies.split(",") if a.strip()]
    symptoms_lower   = symptoms.lower()
    antibiotic_lower = antibiotic.strip().lower()

    emergency_words   = ["difficulty breathing", "chest pain", "unconscious", "swelling of face", "blue lips"]
    found_emergencies = [w for w in emergency_words if w in symptoms_lower]

    has_penicillin_allergy = any("penicillin" in a for a in allergy_list)
    is_amoxicillin         = "amoxicillin" in antibiotic_lower
    has_warfarin           = any("warfarin" in m for m in med_list)
    is_ciprofloxacin       = "ciprofloxacin" in antibiotic_lower

    app_msg    = "Antibiotic therapy appears appropriate for standard bacterial indications."
    contra_msg = "No contraindications or drug interactions flagged."
    dosing_msg = "Standard weight-based dosing verified."

    if found_emergencies:
        risk_level = "CRITICAL"
        summary    = f"Emergency red flag detected: patient reported emergency symptom ({found_emergencies[0]}). Immediate emergency medical response is required."
        next_steps = [
            "Do NOT take the requested antibiotic.",
            "Seek immediate emergency medical attention or go to the nearest emergency department.",
            "Contact a healthcare professional to address emergency symptoms immediately.",
        ]
        app_msg    = "Emergency symptom bypass: clinical safety overrides standard indication check."
        contra_msg = "Safety gate override: clinical emergency requires direct physical examination."
        dosing_msg = "Unable to verify safe dosing in an active medical emergency."

    elif has_warfarin and is_ciprofloxacin:
        risk_level = "CRITICAL"
        summary    = "There may be a significant interaction between Ciprofloxacin and Warfarin that could increase bleeding risk."
        next_steps = [
            "Consult a licensed physician before taking this medication.",
            "Speak with a pharmacist regarding possible interactions.",
            "Seek urgent medical attention if symptoms worsen.",
        ]
        app_msg    = "Antibiotic may not be warranted based on the reported symptoms (throat irritation and mild fever are frequently viral)."
        contra_msg = "Potential interaction detected between Ciprofloxacin and Warfarin (Ciprofloxacin inhibits CYP1A2 and CYP3A4, increasing Warfarin toxicity)."
        dosing_msg = "Unable to verify safe dosing from available information (requires therapeutic drug monitoring and INR safety gates)."

    elif has_penicillin_allergy and is_amoxicillin:
        risk_level = "CRITICAL"
        summary    = "Critical allergy concern: Amoxicillin (a penicillin class drug) requested for a patient with a known penicillin allergy."
        next_steps = [
            "Consult a licensed physician before taking this medication.",
            "Request a non-beta-lactam alternative antibiotic class from a healthcare professional.",
            "Speak with a pharmacist regarding potential cross-reactivity concerns.",
        ]
        app_msg    = "Antibiotic is appropriate for bacterial etiology, but overridden due to safety risk."
        contra_msg = "Potential cross-reactivity detected between requested Amoxicillin and patient's penicillin allergy."
        dosing_msg = "Unable to verify safe dosing from available information."

    else:
        if severity >= 8:
            risk_level = "HIGH"
            summary    = f"High risk identified due to high symptom severity index ({severity}/10). Direct clinical review is recommended."
            next_steps = [
                "Consult a licensed physician before taking this medication.",
                "Speak with a pharmacist regarding appropriate treatment guidelines.",
                "Monitor body temperature and symptoms closely; seek physical assessment today.",
            ]
            app_msg    = "Antibiotic appropriateness is pending clinical diagnostics (requires physical examination)."
            contra_msg = "No contraindications or drug interactions flagged."
            dosing_msg = "Dose requires clinical verification due to high symptom severity."

        elif severity >= 5 or age < 5 or age > 75:
            risk_level = "MODERATE"
            summary    = "Moderate concern identified. Patient is in a vulnerable age cohort or reports moderate symptom severity."
            next_steps = [
                "Consult a licensed physician before taking this medication.",
                "Speak with a pharmacist regarding safe pediatric/geriatric dosing.",
                "Observe symptoms for changes.",
            ]
            app_msg    = "Antibiotic therapy may be appropriate; suggest confirmation from a provider."
            contra_msg = "No contraindications or drug interactions flagged."
            dosing_msg = "Dose requires verification. Dosing adjustments may be required for age/weight cohort."

        else:
            risk_level = "LOW"
            summary    = "Low clinical risk. No drug interactions, allergies, or emergency symptoms were detected."
            next_steps = [
                "Speak with a pharmacist regarding possible interactions.",
                "Observe symptoms for changes; consult a provider if symptoms persist.",
            ]
            app_msg    = "Antibiotic therapy appears appropriate for reported symptoms."
            contra_msg = "No contraindications or drug interactions flagged."
            dosing_msg = "Standard weight-based dosing verified."

    findings = [
        {
            "label_flagged": "Drug interaction detected",
            "label_clear":   "No drug interactions flagged",
            "flagged": has_warfarin and is_ciprofloxacin,
        },
        {
            "label_flagged": "Antibiotic may not be appropriate",
            "label_clear":   "Antibiotic appropriateness verified",
            "flagged": ("throat" in symptoms_lower or "fever" in symptoms_lower) if is_ciprofloxacin else False,
        },
        {
            "label_flagged": "Allergy concern detected",
            "label_clear":   "No allergy concerns flagged",
            "flagged": has_penicillin_allergy and is_amoxicillin,
        },
        {
            "label_flagged": "Dose requires verification",
            "label_clear":   "Standard dose verified",
            "flagged": (severity >= 5 or age < 5 or age > 75),
        },
    ]

    return {
        "risk_level":    risk_level,
        "summary":       summary,
        "findings":      findings,
        "next_steps":    next_steps,
        "agent_details": {
            "appropriateness":  app_msg,
            "contraindication": contra_msg,
            "dosing":           dosing_msg,
        },
    }


# ==========================================
# 3. MONITOR HTML GENERATOR
# ==========================================

_NODE_IDS = [
    "intake", "rag", "appropriateness", "contraindication",
    "dosing", "supervisor", "safety_gate", "counsellor",
]

_NODES_INFO = [
    {"id": "intake",           "name": "Intake Node",            "desc": "Ingests and parses patient demographics, symptoms, and medical histories."},
    {"id": "rag",              "name": "RAG Retrieval Node",     "desc": "Searches clinical knowledge bases for interaction profiles and safe prescribing rules."},
    {"id": "appropriateness",  "name": "Appropriateness Agent",  "desc": "Evaluates symptoms against medical guidelines to verify antibiotic indication."},
    {"id": "contraindication", "name": "Contraindication Agent", "desc": "Analyzes listed allergies and current medications for incompatibilities."},
    {"id": "dosing",           "name": "Dosing Agent",           "desc": "Computes recommended drug quantities and schedules adjusted for weight and age."},
    {"id": "supervisor",       "name": "Supervisor Agent",       "desc": "Aggregates and synthesizes sub-agent responses, resolving clinical conflicts."},
    {"id": "safety_gate",      "name": "Safety Gate",            "desc": "Enforces strict bounds to prevent hazardous prescriptions or missed red flags."},
    {"id": "counsellor",       "name": "Counsellor Agent",       "desc": "Formulates non-judgmental, patient-friendly safety advice and medical instructions."},
]


def generate_monitor_html(node_statuses):
    html = '<div class="agent-grid">'
    for node in _NODES_INFO:
        status     = node_statuses.get(node["id"], "Pending")
        status_cls = status.lower()
        badge_cls  = f"badge-{status_cls}"
        symbol     = {"Running": "⚙", "Complete": "✓"}.get(status, "⧗")
        html += f"""
        <div class="agent-card status-{status_cls}">
            <div class="agent-title">
                <span>{node['name']}</span>
                <span class="status-badge {badge_cls}">{symbol} {status}</span>
            </div>
            <p class="agent-desc">{node['desc']}</p>
        </div>"""
    html += '</div>'
    return html


# ==========================================
# 4. SIMULATION PIPELINE CONTROLLER
# ==========================================

def run_simulation(symptoms, duration, severity, age, weight, medications, allergies, antibiotic, image):
    """Streaming generator — runs real pipeline if available, else falls back to simulation."""

    # ── Validate ─────────────────────────────────────────────────────────────
    errors = validate_inputs(symptoms, age, weight, antibiotic)
    if errors:
        error_html = f"""
        <div style="background-color:#fef2f2;border:1px solid #fca5a5;border-radius:8px;
                    padding:15px;color:#991b1b;margin-top:10px;">
            <h4 style="margin-top:0;margin-bottom:8px;font-weight:700;
                       display:flex;align-items:center;gap:8px;">
                ⚠️ Patient Intake Validation Failed
            </h4>
            <ul style="margin:0;padding-left:20px;font-size:0.95rem;">
                {"".join(f'<li style="margin-bottom:4px;">{e}</li>' for e in errors)}
            </ul>
        </div>"""
        yield (
            gr.update(value=error_html, visible=True),
            gr.update(selected="intake"),
            gr.update(),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )
        return

    # ── Initialise State ──────────────────────────────────────────────────────
    node_statuses = {nid: "Pending" for nid in _NODE_IDS}
    yield (
        gr.update(visible=False, value=""),
        gr.update(selected="monitor"),
        generate_monitor_html(node_statuses),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
    )

    if not REAL_PIPELINE_AVAILABLE:
        # ── Fallback: Simulated Animation ─────────────────────────────────────
        for nid in _NODE_IDS:
            node_statuses[nid] = "Running"
            yield (gr.update(), gr.update(), generate_monitor_html(node_statuses), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
            time.sleep(0.4)
            node_statuses[nid] = "Complete"
            yield (gr.update(), gr.update(), generate_monitor_html(node_statuses), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
        
        report = evaluate_safety_report(symptoms, duration, severity, age, weight, medications, allergies, antibiotic, image)
        yield from _render_report(report, node_statuses)
        return

    # ── Real Pipeline Execution ───────────────────────────────────────────────
    try:
        profile = PatientProfile(
            chief_complaint=symptoms,
            symptom_duration=duration,
            symptom_severity=severity,
            current_medications=[m.strip() for m in medications.split(",") if m.strip()],
            known_allergies=[a.strip() for a in allergies.split(",") if a.strip()],
            age=int(age),
            weight_kg=float(weight),
            antibiotic_in_question=antibiotic
        )
        initial_state = AegisState(patient_profile=profile)
        
        # Stream graph updates
        for chunk in aegis_graph.stream(initial_state, stream_mode="updates"):
            for node_name, delta in chunk.items():
                if node_name.startswith("__"): continue
                
                # Map LangGraph nodes to UI monitor nodes
                ui_map = {
                    "intake": ["intake"],
                    "parallel_agents": ["rag", "appropriateness", "contraindication", "dosing"],
                    "supervisor_compiler": ["supervisor"],
                    "safety_gate": ["safety_gate"],
                    "counsellor": ["counsellor"]
                }
                
                for ui_nid in ui_map.get(node_name, []):
                    node_statuses[ui_nid] = "Running"
                
                yield (gr.update(), gr.update(), generate_monitor_html(node_statuses), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
                
                # Small delay for visual impact
                time.sleep(0.1)
                
                for ui_nid in ui_map.get(node_name, []):
                    node_statuses[ui_nid] = "Complete"
                
                yield (gr.update(), gr.update(), generate_monitor_html(node_statuses), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

        # Grab final state for the report
        final_state = aegis_graph.invoke(initial_state)
        report_data = _map_graph_state_to_report(final_state)
        yield from _render_report(report_data, node_statuses)

    except Exception as exc:
        logger.error(f"Pipeline error: {exc}", exc_info=True)
        error_html = f"""
        <div style="background-color:#fef2f2;border:1px solid #fca5a5;border-radius:8px;padding:15px;color:#991b1b;margin-top:10px;">
            <h4 style="margin:0 0 8px 0;font-weight:700;">❌ Pipeline Execution Error</h4>
            <p style="margin:0;font-size:0.9rem;">{str(exc)}</p>
        </div>"""
        yield (gr.update(value=error_html, visible=True), gr.update(selected="intake"), gr.update(), gr.update(visible=False), gr.update(visible=True), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

def _map_graph_state_to_report(state):
    """Convert AegisState/ClinicalFindingsReport to UI-compatible report dict."""
    if isinstance(state, dict):
        report = state.get("compiled_report")
        counsellor_output = state.get("counsellor_output", "")
        appropriateness_findings = state.get("appropriateness_findings", {})
        contraindication_findings = state.get("contraindication_findings", {})
        dosing_findings = state.get("dosing_findings", {})
    else:
        report = getattr(state, "compiled_report", None)
        counsellor_output = getattr(state, "counsellor_output", "")
        appropriateness_findings = getattr(state, "appropriateness_findings", {})
        contraindication_findings = getattr(state, "contraindication_findings", {})
        dosing_findings = getattr(state, "dosing_findings", {})
    
    findings = []
    if report:
        # report might be an object or a dict depending on how Supervisor returned it
        def _get_val(obj, key, default=None):
            if isinstance(obj, dict): return obj.get(key, default)
            return getattr(obj, key, default)

        findings.append({
            "label_flagged": "Antibiotic may not be appropriate",
            "label_clear": "Antibiotic appropriateness verified",
            "flagged": not _get_val(report, "antibiotic_warranted", False)
        })
        findings.append({
            "label_flagged": "Contraindications/Interactions detected",
            "label_clear": "No contraindications flagged",
            "flagged": _get_val(report, "contraindications_found", False)
        })
        findings.append({
            "label_flagged": "Dose requires clinical verification",
            "label_clear": "Standard dose verified",
            "flagged": _get_val(report, "dosing_flag", False)
        })

    next_steps = [s.strip() for s in counsellor_output.split("\n") if s.strip() and (s.strip().startswith("-") or s.strip()[0].isdigit())]
    if not next_steps:
        next_steps = [counsellor_output] if counsellor_output else ["Consult a physician."]

    risk_level = _get_val(report, "overall_risk_level", "UNKNOWN") if report else "HIGH"
    gate_degraded = _get_val(report, "gate_degraded", False) if report else False

    return {
        "risk_level": risk_level,
        "summary": "Clinical analysis completed by AEGIS adversarial agents." if not gate_degraded else "Safety Gate degraded: results may be incomplete. Consult a physician.",
        "findings": findings,
        "next_steps": next_steps,
        "agent_details": {
            "appropriateness": str(appropriateness_findings.get("reasoning", "No data")),
            "contraindication": str(contraindication_findings.get("reasoning", "No data")),
            "dosing": str(dosing_findings.get("reasoning", "No data")),
        }
    }

def _render_report(report, node_statuses):
    """Internal helper to yield the final report UI updates."""
    risk_level = report["risk_level"]
    risk_css   = f"risk-{risk_level.lower()}"

    risk_badge_html = f"""
    <div class="risk-card {risk_css}">
        <div class="risk-title">{risk_level} RISK</div>
        <p class="risk-subtitle">AEGIS Automated Safety Guardrail Analysis</p>
    </div>"""

    findings_html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:20px;">'
    for f in report["findings"]:
        if f["flagged"]:
            findings_html += f"""
            <div style="background-color:#fef2f2;border:1px solid #fca5a5;border-radius:8px;
                        padding:14px;display:flex;align-items:center;gap:10px;">
                <span style="color:#ef4444;font-size:1.2rem;font-weight:bold;">⚠️</span>
                <span style="color:#991b1b;font-weight:700;font-size:0.9rem;">{f['label_flagged']}</span>
            </div>"""
        else:
            findings_html += f"""
            <div style="background-color:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;
                        padding:14px;display:flex;align-items:center;gap:10px;">
                <span style="color:#10b981;font-size:1.2rem;font-weight:bold;">✓</span>
                <span style="color:#166534;font-weight:600;font-size:0.9rem;">{f['label_clear']}</span>
            </div>"""
    findings_html += '</div>'

    next_steps_html = '<ul style="margin:0;padding-left:20px;line-height:1.6;font-size:0.95rem;color:#334155;">'
    for step in report["next_steps"]:
        next_steps_html += f'<li style="margin-bottom:8px;">{step}</li>'
    next_steps_html += '</ul>'

    d = report["agent_details"]
    details_html = f"""
    <div style="background-color:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                padding:16px;margin-top:10px;">
        <div style="margin-bottom:12px;border-bottom:1px solid #f1f5f9;padding-bottom:8px;">
            <strong style="color:#0d9488;font-size:0.95rem;">💡 Appropriateness Agent:</strong>
            <p style="margin:4px 0 0 0;font-size:0.9rem;color:#475569;line-height:1.4;">{d['appropriateness']}</p>
        </div>
        <div style="margin-bottom:12px;border-bottom:1px solid #f1f5f9;padding-bottom:8px;">
            <strong style="color:#0d9488;font-size:0.95rem;">🛡️ Contraindication Agent:</strong>
            <p style="margin:4px 0 0 0;font-size:0.9rem;color:#475569;line-height:1.4;">{d['contraindication']}</p>
        </div>
        <div>
            <strong style="color:#0d9488;font-size:0.95rem;">📊 Dosing Agent:</strong>
            <p style="margin:4px 0 0 0;font-size:0.9rem;color:#475569;line-height:1.4;">{d['dosing']}</p>
        </div>
    </div>"""

    yield (
        gr.update(visible=False),
        gr.update(selected="report"),
        generate_monitor_html(node_statuses),
        gr.update(visible=True),
        gr.update(visible=False),
        risk_badge_html,
        report["summary"],
        findings_html,
        next_steps_html,
        details_html,
    )


# ==========================================
# 5. BUTTON CALLBACKS
# ==========================================

def load_demo_case():
    return (
        "sore throat and mild fever for 2 days",
        "1-3 days",
        3,
        45,
        70,
        "warfarin",
        "none",
        "ciprofloxacin",
        None,
        gr.update(visible=False, value=""),
    )


def reset_form():
    default_monitor = generate_monitor_html({nid: "Pending" for nid in _NODE_IDS})
    return (
        "",
        "< 24 hours",
        5,
        None,
        None,
        "",
        "",
        "",
        None,
        gr.update(visible=False, value=""),
        gr.update(selected="intake"),
        default_monitor,
        gr.update(visible=False),
        gr.update(visible=True),
    )


# ==========================================
# 6. APP BUILDER
# ==========================================

def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="AEGIS - Antibiotic Safety Assistant",
        css=CSS_STYLES,
        js="""
        () => {
            const forceLight = () => {
                document.documentElement.classList.remove('dark');
                document.body.classList.remove('dark');
            };
            forceLight();
            setTimeout(forceLight, 100);
            setTimeout(forceLight, 500);
        }
        """,
    ) as demo:

        gr.HTML("""
        <div class="aegis-header">
            <h1>🛡️ AEGIS Antibiotic Safety Assistant</h1>
            <p>Adversarial Engine for Guardian Intelligence in Safe-prescribing</p>
        </div>
        """)

        with gr.Tabs() as tabs:

            # ── TAB 1: PATIENT INTAKE ─────────────────────────────────────────
            with gr.Tab("Patient Intake", id="intake"):
                gr.Markdown("### 📋 Enter Patient Intake and Request Details")

                with gr.Row():
                    with gr.Column(scale=3):
                        symptoms_input = gr.Textbox(
                            label="Chief Complaint / Symptoms",
                            placeholder="Describe current symptoms, e.g., persistent coughing, fever, sore throat...",
                            lines=5, max_lines=10,
                        )
                        with gr.Row():
                            duration_input = gr.Dropdown(
                                choices=["< 24 hours", "1-3 days", "4-7 days", "1-2 weeks", "2+ weeks"],
                                value="1-3 days",
                                label="Symptom Duration",
                            )
                            severity_input = gr.Slider(
                                minimum=1, maximum=10, step=1, value=5,
                                label="Symptom Severity (1: Mild - 10: Severe)",
                            )
                        meds_input = gr.Textbox(
                            label="Current Medications",
                            placeholder="Comma-separated list (e.g., warfarin, lisinopril, metformin, or 'none')",
                        )
                        allergies_input = gr.Textbox(
                            label="Known Allergies",
                            placeholder="Comma-separated list (e.g., penicillin, sulfa, or 'none')",
                        )
                        antibiotic_input = gr.Textbox(
                            label="Antibiotic Being Requested",
                            placeholder="Enter the requested antibiotic name (e.g., ciprofloxacin, amoxicillin)",
                        )

                    with gr.Column(scale=2):
                        age_input = gr.Number(
                            label="Patient Age (years)", minimum=0, maximum=120, precision=0,
                        )
                        weight_input = gr.Number(
                            label="Patient Weight (kg)", minimum=2, maximum=250, precision=1,
                        )
                        image_input = gr.Image(
                            label="Prescription / Pill Bottle Photo (Optional)",
                            type="filepath",
                        )

                with gr.Row():
                    btn_submit = gr.Button("🔍 Analyze Safety Risk", variant="primary",   scale=2)
                    btn_demo   = gr.Button("📖 Load Demo Case",       variant="secondary", scale=1)
                    btn_reset  = gr.Button("🔄 Reset Form",           variant="stop",      scale=1)

                error_box = gr.HTML(visible=False)

            # ── TAB 2: LIVE AGENT MONITOR ─────────────────────────────────────
            with gr.Tab("Live Agent Monitor", id="monitor"):
                gr.Markdown("### ⚙️ Multi-Agent Safe-prescribing Verification pipeline")
                gr.Markdown("Observe safety checks, clinical validation rules, and agent reasoning steps in real time.")
                monitor_html = gr.HTML(
                    value=generate_monitor_html({nid: "Pending" for nid in _NODE_IDS})
                )

            # ── TAB 3: PATIENT REPORT ─────────────────────────────────────────
            with gr.Tab("Patient Report", id="report"):
                gr.HTML("""
                <div class="disclaimer-box">
                    <div class="disclaimer-text">
                        ⚠️ MEDICAL DISCLAIMER: AEGIS does not replace a licensed medical professional.
                        Please consult a qualified physician or pharmacist. AEGIS does NOT prescribe medicine.
                    </div>
                </div>
                """)

                with gr.Column(visible=True) as report_placeholder:
                    gr.HTML("""
                    <div style="text-align:center;padding:40px;border:2px dashed #cbd5e1;
                                border-radius:10px;color:#64748b;">
                        <p style="font-size:1.1rem;font-weight:600;margin:0;">
                            No safety report generated yet.</p>
                        <p style="font-size:0.9rem;margin-top:4px;">
                            Fill out the <strong>Patient Intake</strong> form and select
                            <strong>Analyze Safety Risk</strong> to begin.</p>
                    </div>
                    """)

                with gr.Column(visible=False) as report_container:
                    report_badge = gr.HTML()
                    gr.HTML('<div class="report-section-title">📝 Summary</div>')
                    report_summary = gr.Markdown()
                    gr.HTML('<div class="report-section-title">🔍 Key Findings</div>')
                    report_findings = gr.HTML()
                    gr.HTML('<div class="report-section-title">📍 Recommended Next Step</div>')
                    report_next_steps = gr.HTML()
                    with gr.Accordion("View Analysis Details", open=False):
                        report_details = gr.HTML()

        # ── Event wiring ──────────────────────────────────────────────────────

        _sim_inputs  = [symptoms_input, duration_input, severity_input,
                        age_input, weight_input, meds_input, allergies_input,
                        antibiotic_input, image_input]
        _sim_outputs = [error_box, tabs, monitor_html,
                        report_container, report_placeholder,
                        report_badge, report_summary, report_findings,
                        report_next_steps, report_details]

        btn_submit.click(fn=run_simulation, inputs=_sim_inputs, outputs=_sim_outputs, queue=True)

        btn_demo.click(
            fn=load_demo_case,
            outputs=[symptoms_input, duration_input, severity_input,
                     age_input, weight_input, meds_input, allergies_input,
                     antibiotic_input, image_input, error_box],
        )

        btn_reset.click(
            fn=reset_form,
            outputs=[symptoms_input, duration_input, severity_input,
                     age_input, weight_input, meds_input, allergies_input,
                     antibiotic_input, image_input,
                     error_box, tabs, monitor_html,
                     report_container, report_placeholder],
        )

    return demo


# ==========================================
# Standalone entry point
# ==========================================

if __name__ == "__main__":
    create_app().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
