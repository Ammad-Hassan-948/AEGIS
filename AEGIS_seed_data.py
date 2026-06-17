# AEGIS — Pre-Sourced Seed Data
# Drop this directly into rag/seed_data.py's SEED_DOCUMENTS list at the build stage
# specified in the AEGIS Blueprint (Section 17, Prompt 3 stop-point).
#
# Every entry below is a close paraphrase/excerpt of real, named clinical sources
# (NICE guidance, peer-reviewed drug interaction literature, IDSA/EAU dosing
# consensus, drug monographs). Sources are attributed in the "source" field so
# the grounding/citation check in the Safety Gate (Prompt 5) has something real
# to verify against.
#
# NOTE: These are excerpts for hackathon demo/prototype grounding only.
# This is not a substitute for consulting current primary clinical literature
# in a production or clinical setting.

SEED_DOCUMENTS = [

    # --- Category 1: Viral vs bacterial sore throat / antibiotic appropriateness ---
    # Used by: Appropriateness Agent — Scenario A (sore throat, antibiotic requested)

    {
        "text": "Acute sore throat, including pharyngitis and tonsillitis, is often caused by a "
                "viral infection of the upper respiratory tract. Symptoms typically last about a "
                "week and most people get better within this time without antibiotics, regardless "
                "of whether the cause is bacterial or viral. Withholding antibiotics rarely leads "
                "to complications.",
        "source": "NICE_NG84_Sore_Throat_Antimicrobial_Prescribing"
    },
    {
        "text": "Epidemiological studies suggest that only 15 to 30 percent of sore throats are "
                "caused by bacterial infection (such as group A streptococcus). The remainder are "
                "viral and self-limiting, meaning antibiotics provide no benefit and carry only "
                "risk of side effects and resistance.",
        "source": "PMC_Antimicrobial_Stewardship_Sore_Throat_Review"
    },
    {
        "text": "Use the FeverPAIN or Centor clinical scoring criteria to identify patients who are "
                "more likely to benefit from an antibiotic. People most likely to benefit from an "
                "antibiotic have a FeverPAIN score of 4 or 5, or a Centor score of 3 or 4. For "
                "patients without these risk features, symptomatic management with analgesia is "
                "the recommended first-line approach rather than antibiotic therapy.",
        "source": "NICE_NG84_Sore_Throat_Antimicrobial_Prescribing"
    },
    {
        "text": "Inappropriate antibiotic use for pharyngitis contributes directly to the development "
                "and spread of antibiotic resistance. An important step in effective antibiotic "
                "stewardship is to establish whether there is a non-viral cause before considering "
                "antibiotic therapy, since most patients presenting with sore throat are seeking "
                "symptom relief rather than requiring antimicrobial treatment.",
        "source": "PMC_PointOfCare_Testing_Pharyngitis_Pharmacy"
    },

    # --- Category 2: Ciprofloxacin + Warfarin interaction ---
    # Used by: Contraindication Checker Agent — Scenario A (patient on Warfarin requests Ciprofloxacin)

    {
        "text": "Ciprofloxacin and warfarin have a clinically significant drug interaction that can "
                "increase bleeding risk. Ciprofloxacin enhances warfarin's anticoagulant effect by "
                "inhibiting its metabolism via CYP1A2 inhibition, which metabolizes the more potent "
                "R-isomer of warfarin, potentially leading to elevated INR levels and increased risk "
                "of hemorrhage.",
        "source": "DrOracle_Cipro_Warfarin_INR_Interaction"
    },
    {
        "text": "Patients on concurrent ciprofloxacin and warfarin may experience prolonged INR values, "
                "often above 4.0, which significantly increases bleeding risk. Common bleeding "
                "manifestations include gastrointestinal bleeding, hematuria, excessive bruising, and "
                "in severe cases, intracranial hemorrhage. The interaction typically begins within "
                "2 to 5 days of starting ciprofloxacin and can persist for several days after "
                "discontinuation due to warfarin's long half-life.",
        "source": "Empathia_Warfarin_Ciprofloxacin_Interaction_Summary"
    },
    {
        "text": "Nested case-control and retrospective cohort studies found a 48 percent to two-fold "
                "increase in risk of bleeding requiring hospitalization with exposure to antibiotic "
                "therapy in patients on warfarin. When possible, substitute ciprofloxacin with an "
                "antibiotic with a lower bleeding-risk profile, such as clindamycin or cephalexin, "
                "in patients on stable warfarin therapy.",
        "source": "HelloPharmacist_Ciprofloxacin_Warfarin_Interaction_Detail"
    },
    {
        "text": "If concomitant use of ciprofloxacin and warfarin is deemed clinically necessary, early "
                "and more frequent monitoring of INR is recommended, especially during initiation and "
                "discontinuation of the antibiotic. Risk factors for a clinically significant "
                "interaction include advanced age and polypharmacy.",
        "source": "HelloPharmacist_Ciprofloxacin_Warfarin_Interaction_Detail"
    },
    {
        "text": "Warfarin has a narrow therapeutic index and requires frequent monitoring and dose "
                "adjustments to maintain the delicate balance between adequate anticoagulation and "
                "the risk of bleeding or thrombotic complications, typically assessed via prothrombin "
                "time and INR within a target therapeutic window of 2.0 to 3.0.",
        "source": "PMC_DrugDrug_Interactions_Bleeding_Risk_Warfarin_Study"
    },

    # --- Category 3: Nitrofurantoin adult dosing for uncomplicated UTI ---
    # Used by: Dosing Agent — Scenario B (confirmed bacterial UTI, Nitrofurantoin requested)

    {
        "text": "For uncomplicated urinary tract infections in women, the recommended first-line "
                "regimen is nitrofurantoin monohydrate/macrocrystals 100 mg orally twice daily for "
                "5 days. This regimen achieves clinical cure rates of 88 to 93 percent and bacterial "
                "cure rates of 81 to 92 percent, and is endorsed by both the Infectious Diseases "
                "Society of America and the European Association of Urology.",
        "source": "DrOracle_Nitrofurantoin_Adult_UTI_Dosing_IDSA_EAU"
    },
    {
        "text": "The optimal treatment duration for nitrofurantoin in uncomplicated UTI is 5 to 7 days, "
                "with 5 days being the most commonly recommended duration, balancing efficacy with "
                "minimizing unnecessary antibiotic exposure. Three-day courses have been studied but "
                "lack robust supporting evidence compared to 5-day regimens.",
        "source": "DrOracle_Nitrofurantoin_Adult_UTI_Dosing_IDSA_EAU"
    },
    {
        "text": "Alternative dosing for nitrofurantoin macrocrystals (capsule or suspension form) is "
                "50 to 100 mg taken four times daily for 5 days if the dual-release "
                "monohydrate/macrocrystal formulation is unavailable. Maximum recommended adult dose "
                "with normal renal function is 100 mg four times daily, equal to 400 mg per day.",
        "source": "DrOracle_Nitrofurantoin_Maximum_Adult_Dose"
    },
    {
        "text": "Nitrofurantoin should not be used if creatinine clearance is below 60 mL/min, due to "
                "inadequate urinary drug concentration and increased risk of peripheral neuropathy "
                "and other toxicity. It should also be avoided if early pyelonephritis is suspected, "
                "as it does not achieve adequate renal tissue concentrations and is only appropriate "
                "for lower urinary tract (bladder) infections.",
        "source": "DrOracle_Nitrofurantoin_Adult_UTI_Dosing_IDSA_EAU"
    },
    {
        "text": "Most common side effects of nitrofurantoin are nausea and headache, with adverse "
                "event rates reported between 5.6 and 34 percent. Routine post-treatment urine "
                "cultures are not indicated for asymptomatic patients following a standard course.",
        "source": "DrOracle_Nitrofurantoin_Adult_UTI_Dosing_IDSA_EAU"
    },
]

# --- Quick reference: which demo scenario uses which categories ---
#
# Scenario A (45yo, 70kg, on Warfarin, sore throat, requests Ciprofloxacin):
#   -> Appropriateness Agent retrieves Category 1 (viral sore throat guidance)
#      Expected finding: antibiotic NOT warranted, low confidence in need
#   -> Contraindication Agent retrieves Category 2 (Cipro-Warfarin interaction)
#      Expected finding: SEVERE interaction, elevated bleeding risk
#   -> Overall risk level: CRITICAL
#
# Scenario B (28yo, 65kg, no current meds, confirmed bacterial UTI, requests Nitrofurantoin):
#   -> Appropriateness Agent: no viral-sore-throat context retrieved (different complaint),
#      should rely on confirmed bacterial diagnosis already stated in PatientProfile
#   -> Contraindication Agent retrieves nothing relevant (no current_medications) -> no interactions
#   -> Dosing Agent retrieves Category 3 (Nitrofurantoin dosing) -> confirms standard 100mg BID x5 days
#   -> Overall risk level: LOW
