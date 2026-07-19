"""The evidence lane. Explain a prediction, never make it.

This module powers the report's evidence panel. For a gene the model relied on,
it returns a plain-language note plus links to authoritative, open, domain
databases (CARD and NCBI). This is retrieval-augmented explanation done honestly:
the model already decided work / fail / no-call, and this only dresses that
decision in citable context so a clinician can see WHY.

Optionally, if an OpenAI API key is present, it asks a model to phrase the note
in one clinician-friendly sentence. The LLM is a narrator here, not the predictor.
If no key is set it uses the curated static note, so the app always works offline.
"""

from __future__ import annotations

import os

# Curated, human-checked notes for each resistance determinant. Leading with the
# mechanism keeps the explanation honest and grounded rather than a black box.
GENE_INFO = {
    "blaTEM-1":    ("Beta-lactamase enzyme that breaks down penicillins such as ampicillin.", "beta-lactam"),
    "blaSHV-ESBL": ("Extended-spectrum beta-lactamase, degrades penicillins and many cephalosporins.", "beta-lactam"),
    "blaCTX-M-15": ("Extended-spectrum beta-lactamase, a leading cause of cephalosporin resistance.", "beta-lactam"),
    "gyrA_S83L":   ("Point mutation in DNA gyrase that lowers fluoroquinolone binding.", "fluoroquinolone"),
    "qnrB":        ("Plasmid gene that protects DNA gyrase from fluoroquinolones.", "fluoroquinolone"),
    "aac6-Ib-cr":  ("Enzyme variant that inactivates both fluoroquinolones and aminoglycosides.", "multi"),
    "aac6-Ib":     ("Aminoglycoside-modifying enzyme, reduces gentamicin and related drugs.", "aminoglycoside"),
    "armA":        ("16S rRNA methyltransferase giving high-level aminoglycoside resistance.", "aminoglycoside"),
    "rmtB":        ("16S rRNA methyltransferase giving high-level aminoglycoside resistance.", "aminoglycoside"),
    "blaKPC-2":    ("Carbapenemase that hydrolyses carbapenems including meropenem.", "carbapenem"),
    "blaNDM-1":    ("Metallo-beta-lactamase conferring broad carbapenem resistance.", "carbapenem"),
    "blaOXA-48":   ("Carbapenem-hydrolysing oxacillinase, often hard to detect phenotypically.", "carbapenem"),
}

# Evidence categories required by the brief.
KNOWN_GENE = "known_resistance_gene"        # a known determinant was detected
STATISTICAL = "statistical_association"     # only a statistical signal, no known gene
NO_SIGNAL = "no_known_resistance_signal"    # nothing suggestive found
TARGET_ABSENT = "intrinsic_target_absent"   # drug's molecular target is missing


def card_url(gene: str) -> str:
    """Link to the Comprehensive Antibiotic Resistance Database search for a gene."""
    return f"https://card.mcmaster.ca/ontology/search?query={gene}"


def ncbi_url(gene: str) -> str:
    """Link to the NCBI reference gene catalog search for a gene."""
    return f"https://www.ncbi.nlm.nih.gov/pathogens/refgene/#{gene}"


def static_note(gene: str) -> str:
    info = GENE_INFO.get(gene)
    return info[0] if info else f"{gene} is associated with resistance in this species."


def plain_language_note(gene: str, call: str, use_llm: bool = True) -> str:
    """One clinician-friendly sentence about a gene. LLM optional, static fallback.

    The LLM only rephrases curated facts. It has no access to the prediction and
    cannot change the call. Set OPENAI_API_KEY to enable it.
    """
    base = static_note(gene)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not (use_llm and api_key):
        return base

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = (
            "Rephrase this antibiotic-resistance fact into one clear sentence for a "
            "busy clinician. Do not add new claims. Fact: "
            f"{base} (gene {gene}, model call: {call})."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # Any API problem falls back to the curated note. The app never breaks.
        return base


def build_evidence_panel(antibiotic: str, active_driver_genes: list[str], use_llm: bool = False):
    """Return a list of evidence items for the report, one per relevant gene."""
    panel = []
    for gene in active_driver_genes:
        panel.append({
            "gene": gene,
            "note": plain_language_note(gene, "likely to fail", use_llm=use_llm),
            "card_url": card_url(gene),
            "ncbi_url": ncbi_url(gene),
        })
    return panel
