"""
engine.convergence_v2 — Mechanical Convergence Criteria (#5)
==============================================================
No vibes. Convergence is tested by:
  - Entropy delta threshold (belief uncertainty must drop)
  - Max iterations (hard cap)
  - Diminishing returns (IG/cost falling below threshold)
  - Contradiction stabilization (conflicts resolved or stable)
If not converged → "NOT_CONVERGED" with precise machine-readable reasons.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class NonConvergenceReason:
    """Machine-readable reason why convergence failed."""
    def __init__(self, code: str, message: str, metric: str = "",
                 current: float = 0, threshold: float = 0):
        self.code = code
        self.message = message
        self.metric = metric
        self.current = current
        self.threshold = threshold

    def to_dict(self) -> Dict:
        return {"code": self.code, "message": self.message,
                "metric": self.metric, "current": self.current,
                "threshold": self.threshold}


@dataclass
class ConvergenceConfig:
    """Thresholds for mechanical convergence testing."""
    min_entropy_reduction_pct: float = 2.0    # Must reduce entropy by ≥2%
    min_confidence_delta: float = 0.05         # Must gain ≥5% confidence
    max_iterations: int = 3                    # Hard cap on retries
    min_ig_per_call: float = 0.1               # Min info gain per tool call
    max_unresolved_conflicts: int = 1          # At most 1 unresolved conflict
    min_usable_pct: float = 60.0               # ≥60% variables must be usable
    min_grounding_pct: float = 30.0            # ≥30% must be evidence-backed
    schema_must_pass: bool = True              # Schema validation required


@dataclass
class ConvergenceReport:
    """Mechanical pass/fail report for an agent or pipeline."""
    converged: bool
    reasons: List[NonConvergenceReason] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    iteration: int = 0

    def to_dict(self) -> Dict:
        return {
            "converged": self.converged,
            "iteration": self.iteration,
            "metrics": self.metrics,
            "reasons": [r.to_dict() for r in self.reasons] if not self.converged else [],
        }


class MechanicalConvergence:
    """Tests convergence mechanically — no subjective judgments."""

    def __init__(self, config: ConvergenceConfig = None):
        self.config = config or ConvergenceConfig()
        self._history: List[Dict] = []  # Track metrics across iterations

    def record_iteration(self, metrics: Dict):
        """Record metrics for current iteration."""
        self._history.append(dict(metrics))

    def test(self, metrics: Dict, iteration: int = 0) -> ConvergenceReport:
        """Test all convergence criteria. Returns pass/fail with reasons."""
        reasons = []

        # 1. Entropy reduction
        entropy_initial = metrics.get("entropy_initial", 100)
        entropy_final = metrics.get("entropy_final", 100)
        if entropy_initial > 0:
            reduction_pct = (entropy_initial - entropy_final) / entropy_initial * 100
        else:
            reduction_pct = 0
        if reduction_pct < self.config.min_entropy_reduction_pct:
            reasons.append(NonConvergenceReason(
                code="ENTROPY_INSUFFICIENT",
                message=f"Entropy reduced by {reduction_pct:.1f}%, need ≥{self.config.min_entropy_reduction_pct}%",
                metric="entropy_reduction_pct",
                current=reduction_pct,
                threshold=self.config.min_entropy_reduction_pct,
            ))

        # 2. Max iterations
        if iteration >= self.config.max_iterations:
            reasons.append(NonConvergenceReason(
                code="MAX_ITERATIONS",
                message=f"Reached max iterations ({self.config.max_iterations})",
                metric="iterations", current=iteration,
                threshold=self.config.max_iterations,
            ))

        # 3. Diminishing returns (IG/cost ratio)
        ig_per_call = metrics.get("ig_per_call", 0)
        if ig_per_call < self.config.min_ig_per_call and iteration > 0:
            reasons.append(NonConvergenceReason(
                code="DIMINISHING_RETURNS",
                message=f"Info gain per call ({ig_per_call:.3f}) below threshold ({self.config.min_ig_per_call})",
                metric="ig_per_call", current=ig_per_call,
                threshold=self.config.min_ig_per_call,
            ))

        # 4. Contradiction stabilization
        unresolved = metrics.get("unresolved_conflicts", 0)
        if unresolved > self.config.max_unresolved_conflicts:
            reasons.append(NonConvergenceReason(
                code="UNRESOLVED_CONFLICTS",
                message=f"{unresolved} unresolved conflicts (max {self.config.max_unresolved_conflicts})",
                metric="unresolved_conflicts", current=unresolved,
                threshold=self.config.max_unresolved_conflicts,
            ))

        # 5. Schema validation
        schema_passed = metrics.get("schema_passed", True)
        if self.config.schema_must_pass and not schema_passed:
            reasons.append(NonConvergenceReason(
                code="SCHEMA_FAILED",
                message="Output failed schema validation after auto-repair",
                metric="schema_passed", current=0, threshold=1,
            ))

        # 6. Usable coverage
        usable_pct = metrics.get("usable_pct", 100)
        if usable_pct < self.config.min_usable_pct:
            reasons.append(NonConvergenceReason(
                code="INSUFFICIENT_COVERAGE",
                message=f"Only {usable_pct:.0f}% variables usable (need ≥{self.config.min_usable_pct:.0f}%)",
                metric="usable_pct", current=usable_pct,
                threshold=self.config.min_usable_pct,
            ))

        converged = len(reasons) == 0
        return ConvergenceReport(
            converged=converged, reasons=reasons, iteration=iteration,
            metrics={
                "entropy_reduction_pct": round(reduction_pct, 1),
                "ig_per_call": round(ig_per_call, 3),
                "unresolved_conflicts": unresolved,
                "usable_pct": usable_pct,
                "schema_passed": schema_passed,
            },
        )
