"""
engine.calibration — Calibration Tracking (#11)
==================================================
Tracks predicted confidence vs eventual verification.
Computes Brier score per variable, per run.
Flags overconfidence. Works even with simulated inputs.
"""

from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CalibrationEntry:
    variable: str
    run_id: str
    predicted_value: float
    predicted_confidence: float
    predicted_low: float
    predicted_high: float
    actual_value: Optional[float] = None     # Filled when verification arrives
    verified: bool = False
    brier_score: Optional[float] = None      # Filled on verification

    def verify(self, actual: float):
        """Record actual value and compute Brier score."""
        self.actual_value = actual
        self.verified = True
        # Brier: (predicted_probability - actual_outcome)^2
        in_range = self.predicted_low <= actual <= self.predicted_high
        outcome = 1.0 if in_range else 0.0
        self.brier_score = (self.predicted_confidence - outcome) ** 2


class CalibrationLog:
    """Persistent calibration log across runs.

    Records predictions now, verifies later.
    Reports Brier scores, overconfidence flags, and calibration curves.
    """

    def __init__(self):
        self._entries: List[CalibrationEntry] = []
        self._by_variable: Dict[str, List[int]] = defaultdict(list)
        self._by_run: Dict[str, List[int]] = defaultdict(list)

    def record_prediction(self, variable: str, run_id: str,
                          predicted_value: float, predicted_confidence: float,
                          predicted_low: float, predicted_high: float):
        """Record a prediction for later verification."""
        entry = CalibrationEntry(
            variable=variable, run_id=run_id,
            predicted_value=predicted_value,
            predicted_confidence=predicted_confidence,
            predicted_low=predicted_low, predicted_high=predicted_high,
        )
        idx = len(self._entries)
        self._entries.append(entry)
        self._by_variable[variable].append(idx)
        self._by_run[run_id].append(idx)

    def verify(self, variable: str, actual_value: float,
               run_id: str = None):
        """Verify predictions for a variable with actual value."""
        indices = self._by_variable.get(variable, [])
        for idx in indices:
            entry = self._entries[idx]
            if run_id and entry.run_id != run_id:
                continue
            if not entry.verified:
                entry.verify(actual_value)

    def brier_score(self, variable: str = None, run_id: str = None) -> Optional[float]:
        """Average Brier score (lower = better calibrated). Perfect = 0."""
        entries = self._filter(variable, run_id)
        verified = [e for e in entries if e.verified and e.brier_score is not None]
        if not verified:
            return None
        return round(sum(e.brier_score for e in verified) / len(verified), 4)

    def overconfidence_flags(self, threshold: float = 0.15) -> List[Dict]:
        """Flag variables where we're consistently overconfident.

        Overconfident = high confidence predictions that are often wrong.
        """
        flags = []
        for var, indices in self._by_variable.items():
            entries = [self._entries[i] for i in indices if self._entries[i].verified]
            if len(entries) < 3:
                continue
            # For entries with confidence > 0.7, what % are actually correct?
            high_conf = [e for e in entries if e.predicted_confidence > 0.7]
            if not high_conf:
                continue
            accuracy = sum(1 for e in high_conf
                          if e.predicted_low <= e.actual_value <= e.predicted_high) / len(high_conf)
            avg_conf = sum(e.predicted_confidence for e in high_conf) / len(high_conf)
            gap = avg_conf - accuracy
            if gap > threshold:
                flags.append({
                    "variable": var,
                    "avg_confidence": round(avg_conf, 2),
                    "actual_accuracy": round(accuracy, 2),
                    "overconfidence_gap": round(gap, 2),
                    "sample_size": len(high_conf),
                })
        return sorted(flags, key=lambda f: f["overconfidence_gap"], reverse=True)

    def calibration_curve(self, buckets: int = 5) -> List[Dict]:
        """Compute calibration curve: predicted confidence vs actual accuracy."""
        verified = [e for e in self._entries if e.verified]
        if not verified:
            return []
        bucket_size = 1.0 / buckets
        curve = []
        for i in range(buckets):
            lo = i * bucket_size
            hi = (i + 1) * bucket_size
            in_bucket = [e for e in verified if lo <= e.predicted_confidence < hi]
            if in_bucket:
                predicted = sum(e.predicted_confidence for e in in_bucket) / len(in_bucket)
                actual = sum(1 for e in in_bucket
                            if e.predicted_low <= e.actual_value <= e.predicted_high) / len(in_bucket)
                curve.append({
                    "bucket": f"{lo:.0%}-{hi:.0%}",
                    "predicted_confidence": round(predicted, 3),
                    "actual_accuracy": round(actual, 3),
                    "count": len(in_bucket),
                })
        return curve

    def report(self) -> Dict:
        total = len(self._entries)
        verified = sum(1 for e in self._entries if e.verified)
        return {
            "total_predictions": total,
            "verified": verified,
            "pending_verification": total - verified,
            "overall_brier": self.brier_score(),
            "overconfidence_flags": self.overconfidence_flags(),
            "calibration_curve": self.calibration_curve(),
        }

    def _filter(self, variable: str = None, run_id: str = None) -> List[CalibrationEntry]:
        if variable:
            indices = self._by_variable.get(variable, [])
            entries = [self._entries[i] for i in indices]
        elif run_id:
            indices = self._by_run.get(run_id, [])
            entries = [self._entries[i] for i in indices]
        else:
            entries = list(self._entries)
        return entries
