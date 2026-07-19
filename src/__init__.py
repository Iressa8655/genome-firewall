"""Genome Firewall - a defensive AI decision-support tool for antibiotic resistance.

This package is organised around the three challenge modules:

    features.py   Module 1  Genome Reader     FASTA -> AMR gene/mutation features
    model.py      Module 2  Predictor         one calibrated model per antibiotic
    evidence.py   evidence  Report evidence    gene -> CARD/NCBI context (RAG lane)
    metrics.py    scoring   honest evaluation  balanced acc, calibration, no-call

The prediction lane and the evidence lane are kept strictly separate. The model
decides work / fail / no-call from the genome. Retrieved evidence only explains
that decision, it never changes it.
"""

__version__ = "0.1.0"
