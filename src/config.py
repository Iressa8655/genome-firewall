"""Central configuration for Genome Firewall.

Everything a judge might ask you to justify lives here in one readable place:
which species, which antibiotics, which genes drive resistance, which molecular
target each drug needs, and the thresholds behind the no-call rule.

NOTE ON SCOPE. This prototype does ONE species and a small panel of antibiotics
on purpose. The challenge brief explicitly rewards doing a narrow scope well with
calibrated confidence over claiming to cover every bug. Swap SPECIES and the
label data for real BV-BRC data when it is ready. The gene lists below are the
well-documented resistance determinants for this species.
"""

# The single supported species for this prototype.
SPECIES = "Klebsiella pneumoniae"

# The antibiotic panel we predict for. Standardised names.
ANTIBIOTICS = [
    "Ampicillin",
    "Ceftriaxone",
    "Ciprofloxacin",
    "Gentamicin",
    "Meropenem",
]

# Representative brand names shown alongside the generic (INN) names. The generic
# name stays the primary anchor for safety and international use, the brand is only
# for quick clinical recognition. Brands vary by country, these are common
# UK / international examples and are trivial to swap.
BRAND_NAMES = {
    "Ampicillin":    "Penbritin",
    "Ceftriaxone":   "Rocephin",
    "Ciprofloxacin": "Ciproxin",
    "Gentamicin":    "Cidomycin",
    "Meropenem":     "Meronem",
}

# Acquired resistance genes and resistance-associated mutations we look for.
# These become the presence/absence feature columns fed to the model.
RESISTANCE_GENES = [
    "blaTEM-1",       # broad-spectrum beta-lactamase
    "blaSHV-ESBL",    # extended-spectrum beta-lactamase
    "blaCTX-M-15",    # extended-spectrum beta-lactamase (cephalosporins)
    "gyrA_S83L",      # DNA gyrase mutation (fluoroquinolones)
    "qnrB",           # quinolone resistance, plasmid mediated
    "aac6-Ib-cr",     # modifies fluoroquinolones and aminoglycosides
    "aac6-Ib",        # aminoglycoside acetyltransferase
    "armA",           # 16S rRNA methyltransferase (aminoglycosides)
    "rmtB",           # 16S rRNA methyltransferase (aminoglycosides)
    "blaKPC-2",       # carbapenemase
    "blaNDM-1",       # metallo-carbapenemase
    "blaOXA-48",      # carbapenemase
]

# Known biological drivers per antibiotic. Used for the honest evidence category:
# if one of these is present, the call rests on a KNOWN resistance gene rather
# than a mere statistical association.
KNOWN_DRIVERS = {
    "Ampicillin":    ["blaTEM-1", "blaSHV-ESBL"],
    "Ceftriaxone":   ["blaCTX-M-15", "blaSHV-ESBL"],
    "Ciprofloxacin": ["gyrA_S83L", "qnrB", "aac6-Ib-cr"],
    "Gentamicin":    ["aac6-Ib", "armA", "rmtB"],
    "Meropenem":     ["blaKPC-2", "blaNDM-1", "blaOXA-48"],
}

# The deterministic drug-target gate. Each drug needs its molecular target to be
# present to have any chance of working. If the target is absent the organism is
# intrinsically non-susceptible, so we must never report "likely to work" purely
# because resistance markers happen to be absent. These target columns are a
# simplified stand-in and are generated alongside the demo data.
TARGET_GENES = {
    "Ampicillin":    ["target_Ampicillin"],
    "Ceftriaxone":   ["target_Ceftriaxone"],
    "Ciprofloxacin": ["target_Ciprofloxacin"],
    "Gentamicin":    ["target_Gentamicin"],
    "Meropenem":     ["target_Meropenem"],
}


def all_feature_columns():
    """The full ordered list of model input columns (resistance + target genes)."""
    target_cols = []
    for cols in TARGET_GENES.values():
        for c in cols:
            if c not in target_cols:
                target_cols.append(c)
    return list(RESISTANCE_GENES) + target_cols


# --- No-call rule thresholds -------------------------------------------------
# The model abstains (returns no-call) when the evidence is weak or conflicting.
# A confident but wrong result could send a care team to the wrong drug, so a
# no-call is a strength, not a failure.

# If the calibrated probability sits within this band around 0.5 the evidence is
# too weak to commit either way.
NO_CALL_MARGIN = 0.15

# Label encoding. In every label table, 1 means resistant (the antibiotic is
# LIKELY TO FAIL) and 0 means susceptible (LIKELY TO WORK).
RESISTANT = 1
SUSCEPTIBLE = 0
