"""
engine.truth_maintenance — Contradiction Handling (#9)
========================================================
Detects conflicts between sources, assigns reliability scores,
and forces resolution (selection with rationale or "unresolved conflict").
Kills silent inconsistency.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

try:
    from .evidence import EvidenceRef
except ImportError:
    from evidence import EvidenceRef


class ConflictSeverity(Enum):
    MINOR = "minor"        # < 5% divergence
    MODERATE = "moderate"  # 5-15% divergence
    MAJOR = "major"        # 15-30% divergence
    CRITICAL = "critical"  # > 30% divergence


class ResolutionMethod(Enum):
    TRUST_WEIGHTED = "trust_weighted"     # Weight by source trust scores
    HIGHEST_GRADE = "highest_grade"       # Take highest-grade evidence
    MOST_RECENT = "most_recent"           # Take freshest source
    HUMAN_REQUIRED = "human_required"     # Escalate to human
    UNRESOLVED = "unresolved"             # Cannot resolve automatically


@dataclass
class Conflict:
    """A detected conflict between two or more evidence sources."""
    conflict_id: str
    variable: str
    values: List[Tuple[float, EvidenceRef]]   # (value, evidence_ref) pairs
    divergence_pct: float                       # Max pairwise divergence
    severity: ConflictSeverity
    detected_at: float = 0.0
    # Resolution
    resolved: bool = False
    resolution_method: Optional[ResolutionMethod] = None
    resolved_value: Optional[float] = None
    resolution_rationale: str = ""

    def to_dict(self) -> Dict:
        return {
            "conflict_id": self.conflict_id, "variable": self.variable,
            "severity": self.severity.value, "divergence_pct": round(self.divergence_pct, 1),
            "values": [(v, r.source_name, r.trust_score()) for v, r in self.values],
            "resolved": self.resolved,
            "resolution": {
                "method": self.resolution_method.value if self.resolution_method else None,
                "value": self.resolved_value,
                "rationale": self.resolution_rationale,
            } if self.resolved else None,
        }


class TruthMaintenanceSystem:
    """Detects and resolves contradictions between evidence sources.

    When two sources disagree on a variable:
    1. Detect: flag the conflict with severity
    2. Score: compute reliability of each source
    3. Resolve: pick a winner or escalate to human
    """

    # Auto-resolution thresholds
    MINOR_THRESHOLD = 0.05       # 5%
    MODERATE_THRESHOLD = 0.15    # 15%
    MAJOR_THRESHOLD = 0.30       # 30%

    def __init__(self):
        self._conflicts: List[Conflict] = []
        self._resolved: Dict[str, Conflict] = {}   # variable → most recent resolution

    def check_for_conflicts(self, variable: str,
                            evidence_refs: List[EvidenceRef]) -> Optional[Conflict]:
        """Check if multiple evidence refs for a variable conflict."""
        if len(evidence_refs) < 2:
            return None

        # Extract numeric values
        values = [(ref.value, ref) for ref in evidence_refs
                  if isinstance(ref.value, (int, float))]
        if len(values) < 2:
            return None

        # Compute max pairwise divergence
        nums = [v for v, _ in values]
        mean = sum(nums) / len(nums)
        if mean == 0:
            return None
        max_div = max(abs(a - b) / abs(mean) for a, _ in values for b, _ in values)

        # Classify severity
        if max_div < self.MINOR_THRESHOLD:
            return None  # Not a real conflict
        elif max_div < self.MODERATE_THRESHOLD:
            severity = ConflictSeverity.MINOR
        elif max_div < self.MAJOR_THRESHOLD:
            severity = ConflictSeverity.MODERATE
        else:
            severity = ConflictSeverity.MAJOR if max_div < 0.5 else ConflictSeverity.CRITICAL

        conflict = Conflict(
            conflict_id=f"conf_{variable}_{len(self._conflicts)}",
            variable=variable, values=values,
            divergence_pct=max_div * 100, severity=severity,
            detected_at=time.time(),
        )
        self._conflicts.append(conflict)
        return conflict

    def resolve(self, conflict: Conflict,
                method: ResolutionMethod = None) -> Conflict:
        """Attempt to resolve a conflict automatically."""
        if method is None:
            # Pick method based on severity
            if conflict.severity in (ConflictSeverity.MINOR, ConflictSeverity.MODERATE):
                method = ResolutionMethod.TRUST_WEIGHTED
            elif conflict.severity == ConflictSeverity.MAJOR:
                method = ResolutionMethod.HIGHEST_GRADE
            else:
                method = ResolutionMethod.HUMAN_REQUIRED

        if method == ResolutionMethod.TRUST_WEIGHTED:
            total_trust = sum(ref.trust_score() for _, ref in conflict.values)
            if total_trust > 0:
                weighted = sum(v * ref.trust_score() for v, ref in conflict.values) / total_trust
            else:
                weighted = sum(v for v, _ in conflict.values) / len(conflict.values)
            conflict.resolved = True
            conflict.resolved_value = round(weighted, 2)
            conflict.resolution_method = method
            sources = ", ".join(f"{ref.source_name}({ref.trust_score():.2f})" for _, ref in conflict.values)
            conflict.resolution_rationale = f"Trust-weighted average from: {sources}"

        elif method == ResolutionMethod.HIGHEST_GRADE:
            best = max(conflict.values, key=lambda x: x[1].trust_score())
            conflict.resolved = True
            conflict.resolved_value = best[0]
            conflict.resolution_method = method
            conflict.resolution_rationale = (
                f"Selected {best[1].source_name} (grade {best[1].grade.value}, "
                f"trust {best[1].trust_score():.2f}) over {len(conflict.values)-1} other sources"
            )

        elif method == ResolutionMethod.MOST_RECENT:
            newest = max(conflict.values, key=lambda x: x[1].retrieved_at)
            conflict.resolved = True
            conflict.resolved_value = newest[0]
            conflict.resolution_method = method
            conflict.resolution_rationale = f"Most recent: {newest[1].source_name}"

        elif method == ResolutionMethod.HUMAN_REQUIRED:
            conflict.resolved = False
            conflict.resolution_method = method
            conflict.resolution_rationale = (
                f"CRITICAL conflict ({conflict.divergence_pct:.0f}% divergence) "
                f"requires human resolution. Values: "
                + ", ".join(f"{v:,.2f} ({r.source_name})" for v, r in conflict.values)
            )

        else:
            conflict.resolved = False
            conflict.resolution_method = ResolutionMethod.UNRESOLVED

        self._resolved[conflict.variable] = conflict
        return conflict

    def get_conflicts(self) -> List[Conflict]:
        return list(self._conflicts)

    def unresolved_conflicts(self) -> List[Conflict]:
        return [c for c in self._conflicts if not c.resolved]

    def report(self) -> Dict:
        total = len(self._conflicts)
        resolved = sum(1 for c in self._conflicts if c.resolved)
        by_severity = {}
        for c in self._conflicts:
            by_severity[c.severity.value] = by_severity.get(c.severity.value, 0) + 1
        return {
            "total_conflicts": total, "resolved": resolved,
            "unresolved": total - resolved, "by_severity": by_severity,
            "conflicts": [c.to_dict() for c in self._conflicts],
        }
