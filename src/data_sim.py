"""Synthetic demonstration data so the whole pipeline runs out of the box.

IMPORTANT. This is NOT real biology. It is a stand-in that lets the app, the
training script and the metrics all run before the real BV-BRC dataset is wired
in. It is deliberately built with lineage structure so that the grouped-split
story is real and demonstrable: genes cluster within a lineage, which is exactly
why a naive random split would leak near-identical genomes and inflate the score.

Replace generate() with a loader for the organiser dataset (or a BV-BRC pull of
1,000 to 3,000 assembled genomes with laboratory-measured results) when ready.
See data/README.md for the expected format.
"""

import numpy as np
import pandas as pd

from . import config


def generate(n_genomes: int = 1200, n_lineages: int = 8, seed: int = 0):
    """Return (X, y, meta) demo tables.

    X    : DataFrame of gene presence/absence (0/1), one row per genome.
    y    : DataFrame of resistance labels (0/1) per antibiotic. 1 = resistant.
    meta : DataFrame with genome_id and lineage_group (for the grouped split).
    """
    rng = np.random.default_rng(seed)
    genes = config.RESISTANCE_GENES
    target_cols = [c for cols in config.TARGET_GENES.values() for c in cols]
    target_cols = list(dict.fromkeys(target_cols))  # dedupe, keep order

    # Each lineage has its own baseline probability of carrying each gene, so
    # genes are correlated within a lineage.
    lineage_gene_prob = rng.uniform(0.03, 0.85, size=(n_lineages, len(genes)))

    lineage_ids = rng.integers(0, n_lineages, size=n_genomes)

    # Sample resistance-gene presence from the lineage profile.
    gene_matrix = np.zeros((n_genomes, len(genes)), dtype=int)
    for i in range(n_genomes):
        probs = lineage_gene_prob[lineage_ids[i]]
        gene_matrix[i] = (rng.random(len(genes)) < probs).astype(int)

    X = pd.DataFrame(gene_matrix, columns=genes)

    # Target genes are usually present (~92%) and mostly lineage independent.
    for tcol in target_cols:
        X[tcol] = (rng.random(n_genomes) < 0.92).astype(int)

    # Build labels. logit is driven by the known driver genes for each drug.
    y = pd.DataFrame(index=X.index)
    for abx in config.ANTIBIOTICS:
        drivers = config.KNOWN_DRIVERS[abx]
        weights = rng.uniform(2.0, 3.5, size=len(drivers))
        logit = np.full(n_genomes, -1.2)  # baseline slightly susceptible
        for g, w in zip(drivers, weights):
            logit = logit + w * X[g].to_numpy()
        p_resistant = 1.0 / (1.0 + np.exp(-logit))

        labels = (rng.random(n_genomes) < p_resistant).astype(int)

        # Deterministic target gate: if the drug's target is absent the organism
        # is intrinsically non-susceptible, so it is resistant regardless.
        for tcol in config.TARGET_GENES[abx]:
            labels = np.where(X[tcol].to_numpy() == 0, config.RESISTANT, labels)

        # A little label noise, since real lab measurements are imperfect.
        flip = rng.random(n_genomes) < 0.04
        labels = np.where(flip, 1 - labels, labels)
        y[abx] = labels

    meta = pd.DataFrame({
        "genome_id": [f"GEN_{i:05d}" for i in range(n_genomes)],
        "lineage_group": lineage_ids,
    })

    return X, y, meta


if __name__ == "__main__":
    X, y, meta = generate()
    print(f"Generated {len(X)} genomes, {X.shape[1]} features, "
          f"{meta['lineage_group'].nunique()} lineages")
    print("\nResistance rate per antibiotic (1 = likely to fail):")
    print(y.mean().round(3).to_string())
