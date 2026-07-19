"""Honest evaluation. The brief is judged on safe performance, not one headline.

We report, per antibiotic:
  - balanced accuracy (guards against class imbalance)
  - recall for resistant cases and susceptible cases, separately
  - F1, AUROC, PR-AUC (PR-AUC matters most under imbalance)
  - Brier score and a reliability curve (does confidence match reality)
  - the no-call rate and the accuracy of the calls the model DID commit to

All of this is computed on the grouped, held-out test set (lineages the model
never saw during training), which is the difference between a strong and a weak
submission.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)

from . import config


def evaluate_probabilities(y_true, y_prob, no_call_margin: float = config.NO_CALL_MARGIN):
    """Full metric bundle for one antibiotic from true labels and P(resistant)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Forced hard predictions (ignoring no-call) for the discrimination metrics.
    y_hard = (y_prob >= 0.5).astype(int)

    # No-call mask: abstain when probability is near 0.5.
    committed = np.abs(y_prob - 0.5) >= no_call_margin
    n_total = len(y_true)
    n_committed = int(committed.sum())

    out = {
        "n": n_total,
        "prevalence_resistant": float(np.mean(y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_hard)),
        "recall_resistant": float(recall_score(y_true, y_hard, pos_label=config.RESISTANT, zero_division=0)),
        "recall_susceptible": float(recall_score(y_true, y_hard, pos_label=config.SUSCEPTIBLE, zero_division=0)),
        "f1": float(f1_score(y_true, y_hard, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "no_call_rate": float(1 - n_committed / n_total) if n_total else 0.0,
    }

    # AUROC and PR-AUC need both classes present.
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auroc"] = float("nan")
        out["pr_auc"] = float("nan")

    # Accuracy on the calls the model actually committed to.
    if n_committed:
        out["accuracy_on_committed"] = float(np.mean(y_hard[committed] == y_true[committed]))
    else:
        out["accuracy_on_committed"] = float("nan")

    return out


def reliability_curve(y_true, y_prob, n_bins: int = 10):
    """Return (mean_predicted, observed_fraction, bin_count) for a reliability plot."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    mean_pred, obs_frac, counts = [], [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() > 0:
            mean_pred.append(float(y_prob[sel].mean()))
            obs_frac.append(float(y_true[sel].mean()))
            counts.append(int(sel.sum()))
    return np.array(mean_pred), np.array(obs_frac), np.array(counts)
