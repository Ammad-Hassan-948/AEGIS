"""Seed the ChromaDB vector store with real clinical reference text.

Run directly:  python -m rag.seed_data
Or call:       seed_vector_store() from application startup.

Each entry in SEED_DOCUMENTS must have:
    "text"   — verbatim excerpt from a real clinical source (no paraphrases)
    "source" — the source document name used in rag_evidence citations
"""

from __future__ import annotations

import re
import sys
from typing import List

SEED_DOCUMENTS: List[dict] = [
    # === Category 1: Viral vs Bacterial URTI / Sore Throat ===
    {
        "text": "Acute sore throat, including pharyngitis and tonsillitis, is self-limiting and often triggered by a viral infection of the upper respiratory tract.",
        "source": "NICE_NG84_Sore_Throat_Antimicrobial_Prescribing"
    },
    {
        "text": "Symptoms of acute sore throat can last for around 1 week, but most people will get better within this time without antibiotics, regardless of cause (bacteria or virus).",
        "source": "NICE_NG84_Sore_Throat_Antimicrobial_Prescribing"
    },
    {
        "text": "Use FeverPAIN or Centor criteria to identify people who are more likely to benefit from an antibiotic. Withholding antibiotics is unlikely to lead to complications.",
        "source": "NICE_NG84_Sore_Throat_Antimicrobial_Prescribing"
    },
    {
        "text": "People who are unlikely to benefit from an antibiotic (FeverPAIN score of 0 or 1, or Centor score of 0, 1 or 2) should not be offered an antibiotic prescription.",
        "source": "NICE_NG84_Sore_Throat_Antimicrobial_Prescribing"
    },
    {
        "text": "People who are most likely to benefit from an antibiotic have a FeverPAIN score of 4 or 5, or a Centor score of 3 or 4. Consider an immediate antibiotic prescription for this group.",
        "source": "NICE_NG84_Sore_Throat_Antimicrobial_Prescribing"
    },
    {
        "text": "The Centor criteria include history of fever, tonsillar exudate, tender anterior cervical lymphadenopathy, and the absence of cough. Each scores 1 point if present.",
        "source": "NICE_Guidance_Sore_Throat_Summary_CPD"
    },
    {
        "text": "Group A beta-haemolytic streptococcus is the most common bacterial cause of sore throat and is isolated in approximately 20% of cases, meaning the majority of presentations are viral.",
        "source": "NICE_Guidance_Sore_Throat_Summary_CPD"
    },
    {
        "text": "Complications associated with bacterial sore throat infection, including quinsy (peri-tonsillar abscess), acute otitis media, and acute sinusitis, are uncommon in both adults and children.",
        "source": "NICE_Guidance_Sore_Throat_Summary_CPD"
    },
    {
        "text": "Phenoxymethylpenicillin is recommended as the first-choice antibiotic for acute sore throat due to its narrow spectrum and low risk of driving antimicrobial resistance.",
        "source": "NICE_Guidance_Sore_Throat_Summary_CPD"
    },

    # === Category 2: Ciprofloxacin–Warfarin Drug Interaction ===
    {
        "text": "Fluoroquinolones, including ciprofloxacin, exhibit a major drug-drug interaction with warfarin that significantly enhances the anticoagulant effect and increases hemorrhage risk.",
        "source": "BNF_81_Appendix_1_Interactions"
    },
    {
        "text": "Ciprofloxacin inhibits the hepatic CYP1A2 isoenzyme, a major pathway responsible for the metabolism of the R-enantiomer of warfarin, causing systemic accumulation.",
        "source": "Medscape_Drug_Interactions_Database"
    },
    {
        "text": "The concurrent administration of ciprofloxacin and warfarin can cause severe, unpredictable elevation of the International Normalized Ratio (INR), often rising above 4.0.",
        "source": "BNF_81_Appendix_1_Interactions"
    },
    {
        "text": "If co-prescribing ciprofloxacin with warfarin cannot be avoided, immediate baseline INR monitoring followed by frequent reassessments throughout therapy is mandatory.",
        "source": "WHO_Formulary_Fluoroquinolone_Warnings"
    },
    {
        "text": "Clinical symptoms of a ciprofloxacin-warfarin interaction include epistaxis, hematuria, extensive gastrointestinal bleeding, bruising, or acute hematoma formation.",
        "source": "Medscape_Drug_Interactions_Database"
    },

    # === Category 3: Nitrofurantoin — UTI Appropriateness and Dosing ===
    {
        "text": "Nitrofurantoin is recommended as a first-line agent in primary care for the treatment of uncomplicated lower urinary tract infections (cystitis) in healthy adult women.",
        "source": "BNF_81_Urinary_Tract_Infections"
    },
    {
        "text": "The standard adult dosage for modified-release nitrofurantoin in an uncomplicated lower urinary tract infection is 100 mg orally twice daily for a duration of 3 to 5 days.",
        "source": "BNF_81_Urinary_Tract_Infections"
    },
    {
        "text": "Alternatively, standard immediate-release nitrofurantoin can be dosed at 50 mg four times daily for a treatment duration of 5 to 7 days.",
        "source": "BNF_81_Urinary_Tract_Infections"
    },
    {
        "text": "Nitrofurantoin is strictly contraindicated in patients with an estimated glomerular filtration rate (eGFR) or creatinine clearance below 30 mL/min due to inadequate urinary clearance and risk of toxic accumulation.",
        "source": "Perth_Childrens_Hospital_ChAMP_Monographs"
    },
    {
        "text": "Nitrofurantoin should not be given to pregnant patients at term (38–42 weeks) or during labor, as it may precipitate hemolytic anemia in the newborn due to immature erythrocyte enzyme systems.",
        "source": "WHO_Essential_Medicines_Formulary"
    },
    {
        "text": "Nitrofurantoin is not suitable for upper urinary tract infections or acute pyelonephritis, as it does not achieve therapeutic tissue concentrations in the renal parenchyma.",
        "source": "BNF_81_Urinary_Tract_Infections"
    },
    {
        "text": "Common adverse events of nitrofurantoin therapy include gastrointestinal distress, nausea, and headache. Rare but severe risks include acute or chronic pulmonary toxicity.",
        "source": "Perth_Childrens_Hospital_ChAMP_Monographs"
    },

    # === Category 4: Ciprofloxacin — Dosing and Safety Cautions ===
    {
        "text": "The standard adult oral dosage of ciprofloxacin for uncomplicated lower urinary tract infections ranges from 250 mg to 500 mg every 12 hours for 3 days.",
        "source": "BNF_81_Fluoroquinolones"
    },
    {
        "text": "Fluoroquinolones, including ciprofloxacin, are associated with a risk of tendonitis and tendon rupture, particularly the Achilles tendon, which can occur within 48 hours of treatment initiation.",
        "source": "WHO_Fluoroquinolone_Safety_Alert"
    },
    {
        "text": "Ciprofloxacin can cause QT interval prolongation on an electrocardiogram, increasing the risk of life-threatening ventricular arrhythmias such as Torsades de Pointes.",
        "source": "BNF_81_Fluoroquinolones"
    },
    {
        "text": "Fluoroquinolones should be avoided or used with extreme caution in children and growing adolescents due to the documented risk of arthropathy and cartilage damage in weight-bearing joints.",
        "source": "WHO_Essential_Medicines_Formulary"
    },

    # === Category 5: General Antibiotic Stewardship ===
    {
        "text": "Pakistan is one of the highest consumers of antibiotics globally, facing distinct challenges from unregulated over-the-counter sales and overuse of broad-spectrum classes.",
        "source": "JMIR_Public_Health_Surveillance_AMR_2026"
    },
    {
        "text": "Analysis of pharmaceutical sales data in Pakistan shows a massive, disproportionate increase in the consumption of macrolides and cephalosporins, reflecting systemic overreliance on last-line therapies.",
        "source": "JMIR_Public_Health_Surveillance_AMR_2026"
    },
    {
        "text": "Surveys across urban centers in Pakistan reveal that over 50% of self-medication instances involve the illegal or informal procurement of antibiotics directly from community pharmacies without professional evaluation.",
        "source": "ResearchGate_OTC_Misuse_Pakistan_Trends"
    },
    {
        "text": "Unregulated sales, patient demand for immediate relief, and lack of awareness regarding antimicrobial resistance are the primary factors driving self-medication trends in South Asia.",
        "source": "ResearchGate_OTC_Misuse_Pakistan_Trends"
    },
    {
        "text": "Inappropriate antimicrobial use, driven heavily by over-the-counter dispensing for self-limiting viral respiratory infections, directly accelerates the emergence of multi-drug resistant bacterial strains.",
        "source": "WHO_Global_Action_Plan_AMR"
    },
]

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, min_len: int = 80, max_len: int = 400) -> List[str]:
    """Split text into sentence-aware chunks of 80–400 characters.

    Each seed entry is already a short factual sentence (< 400 chars), so in
    most cases the chunk IS the full sentence.  The splitter handles the rare
    multi-sentence entry by splitting on sentence boundaries first.
    """
    # Split on sentence-ending punctuation followed by whitespace / end-of-string
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_len:
            current = (current + " " + sentence).strip() if current else sentence
        else:
            if current:
                chunks.append(current)
            # If a single sentence exceeds max_len, keep it whole rather than
            # splitting mid-word — the Safety Gate substring check needs intact text.
            current = sentence

    if current:
        chunks.append(current)

    # Drop anything below the minimum length (would be too vague to be useful)
    return [c for c in chunks if len(c) >= min_len]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_vector_store(force_reseed: bool = False) -> int:
    """Chunk and embed all SEED_DOCUMENTS into ChromaDB.

    Args:
        force_reseed: if True, re-seeds even if the store already has documents.

    Returns:
        Number of chunks added.
    """
    # Late import to avoid loading the embedding model at import time
    from rag.vector_store import add_documents, collection_count

    existing = collection_count()
    if existing > 0 and not force_reseed:
        print(f"[seed_data] Store already has {existing} chunks — skipping reseed. "
              "Pass force_reseed=True to overwrite.")
        return 0

    texts: List[str] = []
    metadatas: List[dict] = []

    for entry in SEED_DOCUMENTS:
        raw_text = entry["text"]
        source = entry["source"]
        chunks = _chunk_text(raw_text)
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({"source_document": source})

    if not texts:
        print("[seed_data] No chunks produced — check SEED_DOCUMENTS is populated.")
        return 0

    add_documents(texts, metadatas)
    total = collection_count()
    print(f"[seed_data] Seeded {len(texts)} chunks. Store now contains {total} chunks.")
    return len(texts)


if __name__ == "__main__":
    force = "--force" in sys.argv
    seed_vector_store(force_reseed=force)
