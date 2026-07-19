"""Module 2 - The Predictor. Will each antibiotic work?

One regularised logistic regression per antibiotic on the AMRFinderPlus feature
matrix. This baseline is fast on CPU, calibrates cleanly and explains itself,
which is exactly what the brief recommends. On top of the raw model we add the
three things that turn a classifier into trustworthy decision support:

  1. a deterministic drug-target gate (never say "likely to work" just because
     resistance markers are absent, if the drug's target is missing it fails)
  2. probability calibration (so a stated 80 percent really means 80 percent)
  3. a no-call rule (abstain on weak or conflicting evidence)

Calibration is fitted on a separate calibration split, and the whole thing is
trained with a grouped split so lineages never leak between fit, calibrate and
test.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from . import config, evidence


def _calibrate(estimator, X, y, method="sigmoid"):
    """Calibrate an already-fitted estimator on a held-out calibration set.

    Handles the sklearn API change: >=1.6 wraps the fitted model in
    FrozenEstimator, older versions use cv='prefit'.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    except ImportError:  # scikit-learn < 1.6
        cal = CalibratedClassifierCV(estimator, method=method, cv="prefit")
    cal.fit(X, y)
    return cal


class GenomeFirewallModel:
    def __init__(self, no_call_margin: float = config.NO_CALL_MARGIN):
        self.antibiotics = list(config.ANTIBIOTICS)
        self.feature_columns = config.all_feature_columns()
        self.no_call_margin = no_call_margin
        self.calibrated: dict[str, CalibratedClassifierCV] = {}
        self.base: dict[str, LogisticRegression] = {}

    # -- training -------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, groups):
        """Fit one calibrated model per antibiotic on a grouped fit/calibrate split."""
        X = X[self.feature_columns]
        groups = np.asarray(groups)

        for abx in self.antibiotics:
            mask = y[abx].notna().to_numpy()
            Xa, ya, ga = X[mask], y[abx][mask].astype(int), groups[mask]

            # Split the training data again by lineage into fit and calibrate.
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=0)
            fit_idx, cal_idx = next(splitter.split(Xa, ya, ga))

            lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
            lr.fit(Xa.iloc[fit_idx], ya.iloc[fit_idx])

            self.base[abx] = lr
            self.calibrated[abx] = _calibrate(lr, Xa.iloc[cal_idx], ya.iloc[cal_idx])
        return self

    # -- prediction -----------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """P(resistant) per antibiotic for a table of genomes (no gate/no-call)."""
        X = X[self.feature_columns]
        out = {}
        for abx in self.antibiotics:
            out[abx] = self.calibrated[abx].predict_proba(X)[:, 1]
        return pd.DataFrame(out, index=X.index)

    def predict_one(self, feature_row: pd.Series) -> dict:
        """Full decision-support output for a single genome.

        Returns a dict keyed by antibiotic, each value carrying the call, the
        calibrated probability, the confidence, the evidence category and the
        list of driver genes behind the call.
        """
        row = feature_row.reindex(self.feature_columns).fillna(0)
        x = row.to_frame().T

        results = {}
        for abx in self.antibiotics:
            # 1. Deterministic target gate.
            targets = config.TARGET_GENES.get(abx, [])
            if targets and any(int(row.get(t, 0)) == 0 for t in targets):
                results[abx] = self._result(
                    call="likely to fail",
                    prob=None,
                    confidence=1.0,
                    category=evidence.TARGET_ABSENT,
                    drivers=[],
                    reason="The drug's molecular target is absent, so the organism is intrinsically non-susceptible.",
                )
                continue

            # 2. Calibrated probability of resistance.
            p = float(self.calibrated[abx].predict_proba(x)[0, 1])

            # 3. Which known drivers are actually present.
            active_drivers = [g for g in config.KNOWN_DRIVERS[abx] if int(row.get(g, 0)) == 1]

            call, confidence, category, reason = self._decide(abx, p, active_drivers)
            results[abx] = self._result(call, p, confidence, category, active_drivers, reason)
        return results

    def _decide(self, abx, p, active_drivers):
        confidence = max(p, 1 - p)

        # Conflict: a known resistance gene is present but the model leans
        # susceptible. That is exactly when we should abstain.
        if active_drivers and p < 0.5:
            return ("no-call", confidence, evidence.STATISTICAL,
                    "A known resistance gene is present but the overall signal is conflicting.")

        # Weak evidence near the decision boundary.
        if abs(p - 0.5) < self.no_call_margin:
            return ("no-call", confidence, evidence.STATISTICAL,
                    "The evidence is too weak to commit either way.")

        if p >= 0.5:
            category = evidence.KNOWN_GENE if active_drivers else evidence.STATISTICAL
            reason = ("A known resistance determinant was detected."
                      if active_drivers else
                      "The model found a statistical association with resistance, with no known gene.")
            return ("likely to fail", confidence, category, reason)

        return ("likely to work", confidence, evidence.NO_SIGNAL,
                "No known resistance signal was found for this drug.")

    @staticmethod
    def _result(call, prob, confidence, category, drivers, reason):
        return {
            "call": call,
            "probability_resistant": prob,
            "confidence": round(float(confidence), 3),
            "evidence_category": category,
            "driver_genes": drivers,
            "reason": reason,
        }

    # -- persistence ----------------------------------------------------------
    def save(self, path: str | Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "GenomeFirewallModel":
        return joblib.load(path)
