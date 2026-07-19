"""Train and evaluate Genome Firewall on the grouped, held-out split.

Run this to (re)build the model and the evaluation artefacts:

    python train.py

It writes:
    artifacts/model.joblib        the trained, calibrated model
    artifacts/metrics.json        the full metric bundle per antibiotic
    artifacts/reliability/*.png   one reliability plot per antibiotic

The app imports build(save=False) so it can train in memory the first time it
runs, meaning the demo works even before you run this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src import config, data_sim, metrics
from src.model import GenomeFirewallModel

ARTIFACTS = Path("artifacts")


def load_data():
    """Swap this for the organiser / BV-BRC loader when the real data is ready."""
    X, y, meta = data_sim.generate()
    groups = meta["lineage_group"].to_numpy()
    return X, y, groups


def build(save: bool = True):
    """Train on a grouped split and evaluate on held-out lineages.

    Returns (model, report) where report[abx] holds 'metrics' and the
    'reliability' curve arrays computed on the test set.
    """
    X, y, groups = load_data()

    # Outer grouped split: whole lineages are held out for testing so no
    # near-identical genome can appear in both train and test.
    outer = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(outer.split(X, y[config.ANTIBIOTICS[0]], groups))

    X_train, y_train, g_train = X.iloc[train_idx], y.iloc[train_idx], groups[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    model = GenomeFirewallModel().fit(X_train, y_train, g_train)

    proba = model.predict_proba(X_test)

    report = {}
    for abx in config.ANTIBIOTICS:
        yt = y_test[abx].astype(int).to_numpy()
        yp = proba[abx].to_numpy()
        report[abx] = {
            "metrics": metrics.evaluate_probabilities(yt, yp),
            "reliability": metrics.reliability_curve(yt, yp),
        }

    _print_summary(report)

    if save:
        _save(model, report, train_groups=g_train, test_groups=groups[test_idx])

    return model, report


def _print_summary(report):
    print(f"\nHeld-out test performance ({config.SPECIES})")
    print("-" * 68)
    header = f"{'Antibiotic':<15}{'BalAcc':>8}{'AUROC':>8}{'PR-AUC':>8}{'Brier':>8}{'NoCall':>8}"
    print(header)
    for abx, r in report.items():
        m = r["metrics"]
        print(f"{abx:<15}{m['balanced_accuracy']:>8.3f}{m['auroc']:>8.3f}"
              f"{m['pr_auc']:>8.3f}{m['brier']:>8.3f}{m['no_call_rate']:>8.2f}")


def _save(model, report, train_groups, test_groups):
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "reliability").mkdir(exist_ok=True)

    model.save(ARTIFACTS / "model.joblib")

    metrics_out = {
        "species": config.SPECIES,
        "split": {
            "type": "de-duplicated by sequence-homology cluster (lineage); whole clusters held out, none in both train and test",
            "train_lineages": sorted(set(int(g) for g in train_groups)),
            "test_lineages": sorted(set(int(g) for g in test_groups)),
        },
        "antibiotics": {abx: r["metrics"] for abx, r in report.items()},
    }
    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    _save_reliability_plots(report)
    print(f"\nSaved model and metrics to {ARTIFACTS}/")


def _save_reliability_plots(report):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping reliability plots")
        return

    for abx, r in report.items():
        mean_pred, obs_frac, _ = r["reliability"]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
        ax.plot(mean_pred, obs_frac, "o-", label="model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction resistant")
        ax.set_title(f"Reliability - {abx}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(ARTIFACTS / "reliability" / f"{abx}.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    build(save=True)
