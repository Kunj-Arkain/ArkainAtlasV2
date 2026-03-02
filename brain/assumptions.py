"""
engine.assumptions — Assumption Layer / Quarantine Missing Data (#1)
=====================================================================
Every variable in the engine is in exactly ONE of four states:

  EVIDENCE   — backed by EvidenceRef(s) with source + payload hash
  USER_INPUT — manually entered by a named user (signer recorded)
  ASSUMPTION — explicit placeholder with tagged bounds + rationale
  UNKNOWN    — no data at all, cannot be used in calculations

HARD RULE: The engine CANNOT upgrade UNKNOWN → number without going
through EVIDENCE, USER_INPUT, or ASSUMPTION first. No silent fills.

The AssumptionTable is a mandatory output showing every variable's
grounding status. This is how auditors know what's real vs assumed.
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class DataStatus(Enum):
    """The four possible states of any variable."""
    EVIDENCE = "EVIDENCE"       # Has EvidenceRef(s) with source + hash
    USER_INPUT = "USER_INPUT"   # Manually entered by named user
    ASSUMPTION = "ASSUMPTION"   # Explicit placeholder, bounded, tagged
    UNKNOWN = "UNKNOWN"         # No data — cannot be used in calcs


@dataclass
class VariableRecord:
    """Complete grounding record for a single variable.

    This is the single source of truth for "what do we know about X
    and where did it come from?"
    """
    variable: str
    status: DataStatus = DataStatus.UNKNOWN
    value: Optional[float] = None
    low: Optional[float] = None            # Bounded range (required for ASSUMPTION)
    high: Optional[float] = None
    confidence: float = 0.0
    # Grounding metadata
    evidence_refs: List[str] = field(default_factory=list)  # EvidenceRef IDs
    signer: Optional[str] = None           # Who entered it (USER_INPUT)
    assumption_rationale: Optional[str] = None  # Why this assumption (ASSUMPTION)
    assumption_tag: Optional[str] = None   # Category: "market_standard", "conservative", etc.
    last_updated: float = 0.0
    update_count: int = 0

    def is_usable(self) -> bool:
        """Can this variable be used in calculations?"""
        return self.status != DataStatus.UNKNOWN

    def is_grounded(self) -> bool:
        """Is this backed by real evidence?"""
        return self.status == DataStatus.EVIDENCE

    def promote_to_evidence(self, value: float, evidence_ref_ids: List[str],
                            confidence: float, low: float = None, high: float = None):
        """Upgrade status to EVIDENCE with refs."""
        self.status = DataStatus.EVIDENCE
        self.value = value
        self.confidence = confidence
        self.evidence_refs = evidence_ref_ids
        if low is not None:
            self.low = low
        if high is not None:
            self.high = high
        self.last_updated = time.time()
        self.update_count += 1

    def set_user_input(self, value: float, signer: str,
                       confidence: float = 0.7,
                       low: float = None, high: float = None):
        """Set as USER_INPUT with signer name."""
        self.status = DataStatus.USER_INPUT
        self.value = value
        self.signer = signer
        self.confidence = confidence
        if low is not None:
            self.low = low
        if high is not None:
            self.high = high
        self.last_updated = time.time()
        self.update_count += 1

    def set_assumption(self, value: float, low: float, high: float,
                       rationale: str, tag: str = "estimate",
                       confidence: float = 0.3):
        """Set as explicit ASSUMPTION with bounds + rationale."""
        self.status = DataStatus.ASSUMPTION
        self.value = value
        self.low = low
        self.high = high
        self.assumption_rationale = rationale
        self.assumption_tag = tag
        self.confidence = confidence
        self.last_updated = time.time()
        self.update_count += 1

    def to_dict(self) -> Dict:
        d = {
            "variable": self.variable, "status": self.status.value,
            "value": self.value, "confidence": round(self.confidence, 3) if self.confidence else 0,
            "is_usable": self.is_usable(), "is_grounded": self.is_grounded(),
            "update_count": self.update_count,
        }
        if self.low is not None:
            d["range"] = [self.low, self.high]
        if self.evidence_refs:
            d["evidence_refs"] = self.evidence_refs
        if self.signer:
            d["signer"] = self.signer
        if self.assumption_rationale:
            d["assumption"] = {"rationale": self.assumption_rationale, "tag": self.assumption_tag}
        return d


class AssumptionTable:
    """Tracks grounding status of ALL variables in a deal analysis.

    This is a mandatory output — every pipeline run MUST produce
    an assumption table showing what's evidence vs assumed vs unknown.
    """

    def __init__(self, required_variables: List[str] = None):
        self._records: Dict[str, VariableRecord] = {}
        self._required = set(required_variables or [])
        self._violations: List[Dict] = []

    def register(self, variable: str) -> VariableRecord:
        """Register a variable (starts as UNKNOWN)."""
        if variable not in self._records:
            self._records[variable] = VariableRecord(variable=variable)
        return self._records[variable]

    def get(self, variable: str) -> Optional[VariableRecord]:
        return self._records.get(variable)

    def get_value(self, variable: str) -> Optional[float]:
        """Get value ONLY if usable. Returns None for UNKNOWN."""
        rec = self._records.get(variable)
        if rec and rec.is_usable():
            return rec.value
        return None

    def require_value(self, variable: str) -> float:
        """Get value or raise error. Enforces the hard rule."""
        rec = self._records.get(variable)
        if not rec or not rec.is_usable():
            self._violations.append({
                "variable": variable, "violation": "UNKNOWN_USED_AS_NUMBER",
                "timestamp": time.time(),
                "message": f"Attempted to use UNKNOWN variable '{variable}' as a number. "
                           f"Must provide EVIDENCE, USER_INPUT, or ASSUMPTION first.",
            })
            raise ValueError(
                f"Cannot use '{variable}': status is "
                f"{rec.status.value if rec else 'UNREGISTERED'}. "
                f"Provide evidence, user input, or explicit assumption."
            )
        return rec.value

    def set_evidence(self, variable: str, value: float,
                     evidence_ref_ids: List[str], confidence: float,
                     low: float = None, high: float = None):
        """Set a variable as EVIDENCE-backed."""
        rec = self.register(variable)
        rec.promote_to_evidence(value, evidence_ref_ids, confidence, low, high)

    def set_user_input(self, variable: str, value: float,
                       signer: str, confidence: float = 0.7,
                       low: float = None, high: float = None):
        """Set a variable as USER_INPUT."""
        rec = self.register(variable)
        rec.set_user_input(value, signer, confidence, low, high)

    def set_assumption(self, variable: str, value: float,
                       low: float, high: float, rationale: str,
                       tag: str = "estimate", confidence: float = 0.3):
        """Set a variable as explicit ASSUMPTION."""
        rec = self.register(variable)
        rec.set_assumption(value, low, high, rationale, tag, confidence)

    # ── Summary + audit ──

    def by_status(self) -> Dict[str, List[str]]:
        """Group variables by their status."""
        groups = {s.value: [] for s in DataStatus}
        for var, rec in self._records.items():
            groups[rec.status.value].append(var)
        return groups

    def coverage_report(self) -> Dict:
        """How much of the analysis is grounded vs assumed?"""
        total = len(self._records)
        if total == 0:
            return {"total": 0, "coverage": 0}
        by_status = self.by_status()
        evidence_count = len(by_status["EVIDENCE"])
        user_count = len(by_status["USER_INPUT"])
        assumption_count = len(by_status["ASSUMPTION"])
        unknown_count = len(by_status["UNKNOWN"])
        grounded = evidence_count + user_count
        usable = grounded + assumption_count

        missing_required = [v for v in self._required
                           if v not in self._records or not self._records[v].is_usable()]

        return {
            "total_variables": total,
            "evidence": evidence_count,
            "user_input": user_count,
            "assumption": assumption_count,
            "unknown": unknown_count,
            "grounding_pct": round(grounded / total * 100, 1),
            "usable_pct": round(usable / total * 100, 1),
            "missing_required": missing_required,
            "violations": self._violations,
        }

    def render_table(self) -> str:
        """Render the assumption table as formatted text."""
        lines = ["╔══════════════════════════════════════════════════════════════╗"]
        lines.append("║                    ASSUMPTION TABLE                         ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")
        lines.append(f"║ {'Variable':<22} {'Status':<12} {'Value':>12} {'Conf':>6} {'Source':<12}║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        status_order = [DataStatus.EVIDENCE, DataStatus.USER_INPUT,
                       DataStatus.ASSUMPTION, DataStatus.UNKNOWN]
        for status in status_order:
            vars_in_status = [(v, r) for v, r in self._records.items() if r.status == status]
            if not vars_in_status:
                continue
            icon = {"EVIDENCE": "■", "USER_INPUT": "▲", "ASSUMPTION": "◆", "UNKNOWN": "○"}[status.value]
            for var, rec in sorted(vars_in_status, key=lambda x: x[0]):
                val_str = f"{rec.value:>10,.2f}" if rec.value is not None else "      —   "
                conf_str = f"{rec.confidence:>5.0%}" if rec.confidence else "   — "
                src = ""
                if rec.evidence_refs:
                    src = f"{len(rec.evidence_refs)} refs"
                elif rec.signer:
                    src = rec.signer[:10]
                elif rec.assumption_tag:
                    src = rec.assumption_tag[:10]
                lines.append(f"║ {icon} {var:<20} {status.value:<12} {val_str} {conf_str} {src:<12}║")

        cov = self.coverage_report()
        lines.append("╠══════════════════════════════════════════════════════════════╣")
        lines.append(f"║ Grounding: {cov['grounding_pct']:>5.1f}%  Usable: {cov['usable_pct']:>5.1f}%"
                     f"  Unknown: {cov['unknown']:<3}  Violations: {len(cov['violations']):<3}  ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "records": {v: r.to_dict() for v, r in self._records.items()},
            "coverage": self.coverage_report(),
        }
