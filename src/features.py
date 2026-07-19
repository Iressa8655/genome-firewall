"""Module 1 - The Genome Reader. From a FASTA genome to model features.

The gold standard annotation tool is NCBI AMRFinderPlus, which is public domain
and unrestricted. Given an assembled genome it detects known resistance genes and
resistance-associated mutations. We turn its output into a presence/absence
feature vector aligned to config.all_feature_columns().

This module tries to call a locally installed `amrfinder` binary. If it is not
installed (for example on Streamlit Cloud) it falls back to reading a precomputed
AMRFinderPlus results table, which is the documented, repeatable path the brief
asks for. Either way the OUTPUT FORMAT is the same: a 0/1 vector over the known
feature columns.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from . import config


def amrfinder_available() -> bool:
    """True if a local AMRFinderPlus binary is on the PATH."""
    return shutil.which("amrfinder") is not None


def run_amrfinderplus(fasta_path: str | Path) -> set[str]:
    """Run AMRFinderPlus on an assembled FASTA and return the set of gene symbols.

    Requires a local install (see https://github.com/ncbi/amr). Raises
    RuntimeError if the binary is missing so callers can fall back gracefully.
    """
    if not amrfinder_available():
        raise RuntimeError("AMRFinderPlus binary not found on PATH")

    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
        out_path = tmp.name

    cmd = [
        "amrfinder",
        "-n", str(fasta_path),      # nucleotide assembled FASTA
        "--organism", "Klebsiella_pneumoniae",
        "-o", out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    table = pd.read_csv(out_path, sep="\t")
    # The gene symbol column is usually "Gene symbol" (older) or "Element symbol".
    symbol_col = next(
        (c for c in table.columns if c.lower() in ("gene symbol", "element symbol")),
        None,
    )
    if symbol_col is None:
        return set()
    return set(table[symbol_col].astype(str))


def genes_to_feature_row(present_genes: set[str]) -> pd.Series:
    """Map a set of detected gene symbols to the ordered 0/1 feature vector.

    Unknown symbols are ignored. Missing known columns default to 0 (absent).
    Target-gene columns are assumed present unless explicitly reported absent,
    because the molecular target is part of the core genome for this species.
    """
    columns = config.all_feature_columns()
    row = pd.Series(0, index=columns, dtype=int)
    for gene in present_genes:
        if gene in row.index:
            row[gene] = 1
    # Core-genome targets: present by default for this species.
    for cols in config.TARGET_GENES.values():
        for tcol in cols:
            if tcol not in present_genes:
                row[tcol] = 1
    return row


def load_precomputed_features(path: str | Path) -> pd.DataFrame:
    """Load a precomputed feature matrix (genome_id index, gene columns of 0/1)."""
    df = pd.read_csv(path)
    if "genome_id" in df.columns:
        df = df.set_index("genome_id")
    # Guarantee every expected column exists, in order.
    for col in config.all_feature_columns():
        if col not in df.columns:
            df[col] = 0
    return df[config.all_feature_columns()]
