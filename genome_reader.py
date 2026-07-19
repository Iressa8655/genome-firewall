"""Module 1 - The Genome Reader.

A documented, repeatable path from an assembled bacterial genome to model
features, using AMRFinderPlus (NCBI, public-domain) as the default annotation
tool. This is the required Module 1 deliverable.

USAGE
    # From an assembled FASTA (requires AMRFinderPlus installed, see github.com/ncbi/amr)
    python genome_reader.py path/to/genome.fasta -o features.csv

    # From a precomputed AMRFinderPlus TSV (works anywhere, no install needed)
    python genome_reader.py data/example/amrfinder_example.tsv -o features.csv

PIPELINE
    FASTA  --(AMRFinderPlus)-->  AMR gene / mutation hits (TSV)
           --(normalise symbols)-->  presence / absence over known determinants
           --(add core-genome drug targets)-->  one feature row

OUTPUT FORMAT SPECIFICATION
    A one-row CSV feature table. The columns are exactly config.all_feature_columns():
    every known AMR gene / mutation determinant, followed by one target_<Antibiotic>
    column per drug. Each value is 1 (present) or 0 (absent). This presence / absence
    row is precisely what Module 2 (the predictor, train.py / src/model.py) consumes.

        header: blaTEM-1,blaSHV-ESBL,blaCTX-M-15,...,blaOXA-48,target_Ampicillin,...,target_Meropenem
        row:    0,1,1,...,1,1,...,1

    The same format scales to many genomes, one row per genome, by concatenating rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import features


def read_genome(input_path: str | Path):
    """Return (raw_hits_df_or_None, feature_row) for a FASTA or an AMRFinderPlus TSV."""
    p = Path(input_path)
    if p.suffix.lower() in (".tsv", ".txt"):
        raw, symbols = features.parse_amrfinder_tsv(p)
        return raw, features.genes_to_feature_row(symbols)
    # Otherwise treat as an assembled FASTA and run AMRFinderPlus.
    symbols = features.run_amrfinderplus(p)
    return None, features.genes_to_feature_row(symbols)


def main():
    ap = argparse.ArgumentParser(
        description="Genome Reader (Module 1): FASTA or AMRFinderPlus TSV -> feature row"
    )
    ap.add_argument("input", help="an assembled FASTA, or a precomputed AMRFinderPlus TSV")
    ap.add_argument("-o", "--out", default="features.csv", help="output CSV path")
    args = ap.parse_args()

    raw, row = read_genome(args.input)
    if raw is not None:
        print(f"Parsed {len(raw)} AMRFinderPlus hits.")
    row.to_frame().T.to_csv(args.out, index=False)
    present = [g for g, v in row.items() if v == 1 and not g.startswith("target_")]
    print(f"Wrote feature row to {args.out}. Determinants present: {', '.join(present) or 'none'}")


if __name__ == "__main__":
    main()
