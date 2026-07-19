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


# AMRFinderPlus reports gene symbols in its own style (e.g. "aac(6')-Ib-cr",
# "blaSHV-12"). Normalise them onto the feature columns declared in config.
_ALIASES = {
    "aac(6')-ib-cr": "aac6-Ib-cr",
    "aac(6')-ib": "aac6-Ib",
    "blakpc-2": "blaKPC-2",
    "blandm-1": "blaNDM-1",
    "blaoxa-48": "blaOXA-48",
    "blactx-m-15": "blaCTX-M-15",
    "blatem-1": "blaTEM-1",
    "arma": "armA",
    "rmtb": "rmtB",
    "gyra_s83l": "gyrA_S83L",
}


def normalize_gene_symbol(symbol: str) -> str:
    """Map an AMRFinderPlus gene symbol onto a config feature column."""
    s = str(symbol).strip()
    low = s.lower()
    if low in _ALIASES:
        return _ALIASES[low]
    # Family-level rules for the beta-lactamase and quinolone variants.
    if low.startswith("blashv"):
        return "blaSHV-ESBL"
    if low.startswith("blactx-m"):
        return "blaCTX-M-15"
    if low.startswith("blakpc"):
        return "blaKPC-2"
    if low.startswith("blaoxa-48"):
        return "blaOXA-48"
    if low.startswith("blandm"):
        return "blaNDM-1"
    if low.startswith("blatem"):
        return "blaTEM-1"
    if low.startswith("qnrb"):
        return "qnrB"
    return s


def parse_amrfinder_tsv(path: str | Path):
    """Parse an AMRFinderPlus output TSV. Returns (raw_hits_df, gene_symbol_set)."""
    df = pd.read_csv(path, sep="\t")
    symbol_col = next(
        (c for c in df.columns if c.lower() in ("gene symbol", "element symbol")),
        None,
    )
    symbols = set(df[symbol_col].astype(str)) if symbol_col else set()
    return df, symbols


def genes_to_feature_row(present_genes: set[str]) -> pd.Series:
    """Map a set of detected gene symbols to the ordered 0/1 feature vector.

    Unknown symbols are ignored. Missing known columns default to 0 (absent).
    Target-gene columns are assumed present unless explicitly reported absent,
    because the molecular target is part of the core genome for this species.
    """
    columns = config.all_feature_columns()
    row = pd.Series(0, index=columns, dtype=int)
    for gene in present_genes:
        norm = normalize_gene_symbol(gene)
        if norm in row.index:
            row[norm] = 1
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
