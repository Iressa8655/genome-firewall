"""The evidence lane. Explain a prediction, never make it.

This module powers the report's evidence panel. For a gene the model relied on,
it returns the resistance MECHANISM, the protein the gene encodes, a plain-language
note, and links to authoritative, open sources: CARD and NCBI for the gene, PubMed
for the literature, and a protein-structure entry so the 3D fold can be explored.
This is retrieval-augmented explanation done honestly: the model already decided
work / fail / no-call, and this only dresses that decision in citable context so a
clinician can see the biological WHY.

Optionally, if an OpenAI API key is present, it asks a model to phrase the note in
one clinician-friendly sentence. The LLM is a narrator here, not the predictor.
If no key is set it uses the curated static note, so the app always works offline.
"""

from __future__ import annotations

import os

# Curated, human-checked facts for each resistance determinant. Each entry carries
# the resistance MECHANISM (how it defeats the drug), the drug class, the protein
# the gene encodes (for the structure link), and a plain-language note.
GENE_INFO = {
    "blaTEM-1":    {"mechanism": "Enzymatic degradation", "class": "beta-lactam",     "protein": "TEM-1 beta-lactamase",                 "note": "Beta-lactamase enzyme that breaks down penicillins such as ampicillin."},
    "blaSHV-ESBL": {"mechanism": "Enzymatic degradation", "class": "beta-lactam",     "protein": "SHV extended-spectrum beta-lactamase", "note": "Extended-spectrum beta-lactamase, degrades penicillins and many cephalosporins."},
    "blaCTX-M-15": {"mechanism": "Enzymatic degradation", "class": "beta-lactam",     "protein": "CTX-M-15 beta-lactamase",              "note": "Extended-spectrum beta-lactamase, a leading cause of cephalosporin resistance."},
    "gyrA_S83L":   {"mechanism": "Target modification",   "class": "fluoroquinolone", "protein": "DNA gyrase subunit A (S83L mutant)",   "note": "Point mutation in DNA gyrase that lowers fluoroquinolone binding."},
    "qnrB":        {"mechanism": "Target protection",     "class": "fluoroquinolone", "protein": "QnrB pentapeptide-repeat protein",     "note": "Plasmid protein that shields DNA gyrase from fluoroquinolones."},
    "aac6-Ib-cr":  {"mechanism": "Drug modification",     "class": "multi",           "protein": "AAC(6')-Ib-cr acetyltransferase",      "note": "Enzyme variant that inactivates both fluoroquinolones and aminoglycosides."},
    "aac6-Ib":     {"mechanism": "Drug modification",     "class": "aminoglycoside",  "protein": "AAC(6')-Ib acetyltransferase",         "note": "Aminoglycoside-modifying enzyme, reduces gentamicin and related drugs."},
    "armA":        {"mechanism": "Target modification",   "class": "aminoglycoside",  "protein": "ArmA 16S rRNA methyltransferase",      "note": "16S rRNA methyltransferase giving high-level aminoglycoside resistance."},
    "rmtB":        {"mechanism": "Target modification",   "class": "aminoglycoside",  "protein": "RmtB 16S rRNA methyltransferase",      "note": "16S rRNA methyltransferase giving high-level aminoglycoside resistance."},
    "blaKPC-2":    {"mechanism": "Enzymatic degradation", "class": "carbapenem",      "protein": "KPC-2 carbapenemase",                  "note": "Carbapenemase that hydrolyses carbapenems including meropenem."},
    "blaNDM-1":    {"mechanism": "Enzymatic degradation", "class": "carbapenem",      "protein": "NDM-1 metallo-beta-lactamase",         "note": "Metallo-beta-lactamase conferring broad carbapenem resistance."},
    "blaOXA-48":   {"mechanism": "Enzymatic degradation", "class": "carbapenem",      "protein": "OXA-48 oxacillinase",                  "note": "Carbapenem-hydrolysing oxacillinase, often hard to detect phenotypically."},
}

# Evidence categories required by the brief.
KNOWN_GENE = "known_resistance_gene"        # a known determinant was detected
STATISTICAL = "statistical_association"     # only a statistical signal, no known gene
NO_SIGNAL = "no_known_resistance_signal"    # nothing suggestive found
TARGET_ABSENT = "intrinsic_target_absent"   # drug's molecular target is missing


def mechanism(gene: str) -> str:
    """The resistance mechanism class for a gene (how it defeats the drug)."""
    return GENE_INFO.get(gene, {}).get("mechanism", "Unknown mechanism")


def protein_of(gene: str) -> str:
    """The protein the gene encodes (used for the structure link and the chain)."""
    return GENE_INFO.get(gene, {}).get("protein", gene)


def static_note(gene: str) -> str:
    info = GENE_INFO.get(gene)
    return info["note"] if info else f"{gene} is associated with resistance in this species."


def card_url(gene: str) -> str:
    """Comprehensive Antibiotic Resistance Database search for a gene."""
    return f"https://card.mcmaster.ca/ontology/search?query={gene}"


def ncbi_url(gene: str) -> str:
    """NCBI reference gene catalog search for a gene."""
    return f"https://www.ncbi.nlm.nih.gov/pathogens/refgene/#{gene}"


def pubmed_url(gene: str) -> str:
    """PubMed literature search for this gene and antibiotic resistance."""
    return f"https://pubmed.ncbi.nlm.nih.gov/?term={gene}+antibiotic+resistance"


def structure_url(gene: str) -> str:
    """UniProt entry for the encoded protein, which carries the AlphaFold 3D fold.

    A determined structural view (rotatable, close to an animation) without having
    to hard-code accession numbers. Falls back to the gene symbol.
    """
    query = protein_of(gene).replace(" ", "+")
    return f"https://www.uniprot.org/uniprotkb?query={query}"


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
            f"{base} (gene {gene}, mechanism {mechanism(gene)}, model call: {call})."
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
            "mechanism": mechanism(gene),
            "protein": protein_of(gene),
            "note": plain_language_note(gene, "likely to fail", use_llm=use_llm),
            "card_url": card_url(gene),
            "ncbi_url": ncbi_url(gene),
            "pubmed_url": pubmed_url(gene),
            "structure_url": structure_url(gene),
        })
    return panel
