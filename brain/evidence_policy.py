"""
engine.evidence_policy — Evidence Grade Gating (Gap D)
========================================================
Rules:
  1. No evidence_refs → cannot set status to EVIDENCE
  2. MODEL_OUTPUT grade cannot override A/B sources without conflict resolution
  3. Critical variables require 2-source confirmation before EVIDENCE status
  4. Grade-based precedence: A > B > C > D > F

Critical variables (require 2-source confirmation):
  NOI, traffic_count, crime_rate, environmental_risk, purchase_price
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple

try:
    from .evidence import EvidenceRef, EvidenceGrade, EvidenceMethod
    from .assumptions import AssumptionTable, DataStatus
except ImportError:
    from evidence import EvidenceRef, EvidenceGrade, EvidenceMethod
    from assumptions import AssumptionTable, DataStatus


# Variables that require 2-source confirmation before EVIDENCE status
CRITICAL_VARIABLES: Set[str] = {
    "noi", "traffic_count", "crime_rate", "environmental_risk",
    "purchase_price", "dscr", "interest_rate",
}

GRADE_RANK = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}


class EvidencePolicyViolation(Exception):
    """Raised when evidence policy is violated."""
    def __init__(self, variable: str, rule: str, message: str):
        self.variable = variable
        self.rule = rule
        super().__init__(f"[{rule}] {variable}: {message}")


class EvidencePolicy:
    """Enforces evidence grading and confirmation rules.

    Call check_before_update() before promoting any variable to EVIDENCE.
    """

    def __init__(self):
        self._violations: List[Dict] = []

    def check_before_update(
        self,
        variable: str,
        new_ref: EvidenceRef,
        existing_refs: List[EvidenceRef],
        assumption_table: AssumptionTable,
    ) -> Tuple[bool, str]:
        """Check whether this evidence update is allowed.

        Returns (allowed: bool, reason: str).
        If not allowed, the caller must NOT promote to EVIDENCE.
        """
        # Rule 1: Must have actual evidence ref
        if not new_ref or not new_ref.ref_id:
            self._violations.append({"variable": variable, "rule": "NO_REF",
                                     "message": "Cannot set EVIDENCE without an EvidenceRef"})
            return False, "No evidence ref provided"

        # Rule 2: MODEL_OUTPUT cannot override A/B sources
        if new_ref.method == EvidenceMethod.MODEL_OUTPUT:
            higher_grade_exists = any(
                GRADE_RANK.get(r.grade.value, 0) >= 4  # A or B
                for r in existing_refs
                if r.ref_id != new_ref.ref_id
            )
            if higher_grade_exists:
                self._violations.append({
                    "variable": variable, "rule": "GRADE_PRECEDENCE",
                    "message": f"MODEL_OUTPUT (grade {new_ref.grade.value}) "
                               f"cannot override existing A/B source"})
                return False, "MODEL_OUTPUT cannot override A/B evidence — use conflict resolution"

        # Rule 3: Critical variables need 2-source confirmation
        if variable in CRITICAL_VARIABLES:
            # Count distinct sources (including the new one)
            all_sources = set()
            for r in existing_refs:
                all_sources.add(r.source_name)
            all_sources.add(new_ref.source_name)

            if len(all_sources) < 2:
                # Allow the update but keep status as ASSUMPTION until confirmed
                rec = assumption_table.get(variable)
                if rec and rec.status == DataStatus.EVIDENCE:
                    # Already has EVIDENCE status — new single-source can't change it
                    pass
                else:
                    return True, (
                        f"Critical variable '{variable}' has only {len(all_sources)} source(s). "
                        f"Will set as ASSUMPTION (pending 2-source confirmation). "
                        f"Need evidence from at least 2 independent sources."
                    )

        # Rule 4: Grade precedence — lower grade can't override higher
        for existing in existing_refs:
            if existing.ref_id == new_ref.ref_id:
                continue
            new_rank = GRADE_RANK.get(new_ref.grade.value, 0)
            existing_rank = GRADE_RANK.get(existing.grade.value, 0)
            if new_rank < existing_rank - 1:
                # Allow but log a warning — significant grade downgrade
                self._violations.append({
                    "variable": variable, "rule": "GRADE_DOWNGRADE_WARNING",
                    "message": f"New evidence grade {new_ref.grade.value} is significantly "
                               f"lower than existing {existing.grade.value}"})

        return True, "OK"

    def get_effective_status(
        self,
        variable: str,
        evidence_refs: List[EvidenceRef],
    ) -> DataStatus:
        """Determine what status a variable should have based on evidence.

        Critical vars with single source → ASSUMPTION (not EVIDENCE).
        """
        if not evidence_refs:
            return DataStatus.UNKNOWN

        sources = set(r.source_name for r in evidence_refs)

        if variable in CRITICAL_VARIABLES and len(sources) < 2:
            return DataStatus.ASSUMPTION  # Single source → not full EVIDENCE

        return DataStatus.EVIDENCE

    @property
    def violations(self) -> List[Dict]:
        return list(self._violations)

    def report(self) -> Dict:
        return {
            "total_violations": len(self._violations),
            "by_rule": {},
            "violations": self._violations[-20:],  # last 20
        }
