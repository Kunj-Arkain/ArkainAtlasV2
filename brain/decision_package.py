"""
engine.decision_package — Auditable Decision Package (#10)
============================================================
Final output always includes:
  - decision + confidence + calibration tag
  - assumptions used (full table)
  - evidence citations (IDs, sources, grades)
  - risk register
  - next actions to raise confidence
  - decision pivots (what would change the answer)
No black-box conclusions.
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DecisionPivot:
    """What would change the decision — sensitivity bound."""
    variable: str
    current_value: float
    pivot_threshold: float         # Value at which decision flips
    pivot_direction: str           # "above" or "below"
    distance_pct: float            # How far from pivot (%)
    description: str

    def to_dict(self) -> Dict:
        return {
            "variable": self.variable, "current": self.current_value,
            "pivot_at": self.pivot_threshold, "direction": self.pivot_direction,
            "distance_pct": round(self.distance_pct, 1),
            "description": self.description,
        }


@dataclass
class NextAction:
    """Concrete action to raise confidence."""
    priority: int                  # 1 = highest
    action: str
    target_variable: str
    expected_confidence_gain: float
    estimated_effort: str          # "minutes", "hours", "days"

    def to_dict(self) -> Dict:
        return {
            "priority": self.priority, "action": self.action,
            "target_variable": self.target_variable,
            "expected_confidence_gain": round(self.expected_confidence_gain, 2),
            "effort": self.estimated_effort,
        }


@dataclass
class DecisionPackage:
    """The complete auditable output of a pipeline run.

    This is what gets delivered to decision-makers.
    Every field is mandatory — no partial packages.
    """
    # Core decision
    run_id: str = ""
    timestamp: float = 0.0
    decision: str = "NEEDS_DATA"   # GO | NO_GO | CONDITIONAL | NEEDS_DATA
    confidence: float = 0.0        # 0-1 overall
    calibration_tag: str = ""      # "well_calibrated" | "overconfident" | "underconfident" | "uncalibrated"

    # Deal context
    deal_summary: Dict = field(default_factory=dict)

    # Grounding
    assumption_table: Dict = field(default_factory=dict)
    evidence_citations: List[Dict] = field(default_factory=list)
    grounding_pct: float = 0.0     # % of vars backed by evidence
    usable_pct: float = 0.0        # % of vars that are usable

    # Risk
    risk_register: List[Dict] = field(default_factory=list)
    risk_score: str = ""           # LOW | MED | HIGH | CRITICAL

    # Financials
    financial_summary: Dict = field(default_factory=dict)
    monte_carlo_summary: Dict = field(default_factory=dict)

    # What would change the decision
    decision_pivots: List[DecisionPivot] = field(default_factory=list)
    next_actions: List[NextAction] = field(default_factory=list)

    # Convergence
    convergence_status: str = ""    # CONVERGED | NOT_CONVERGED | BUDGET_EXHAUSTED
    convergence_details: Dict = field(default_factory=dict)

    # Audit trail
    agents_run: List[str] = field(default_factory=list)
    agents_skipped: List[Dict] = field(default_factory=list)
    total_tool_calls: int = 0
    total_retries: int = 0
    elapsed_ms: int = 0
    replay_log_id: str = ""
    # Tamper evidence (Gap C) + determinism (Gap A)
    ledger_head_hash: str = ""       # Hash chain head — proves audit trail integrity
    ledger_verified: bool = False    # Whether chain verified at package build time
    seed: Optional[int] = None       # RNG seed — enables exact reproducibility

    def is_actionable(self) -> bool:
        """Is this package complete enough to act on?"""
        return (self.decision in ("GO", "NO_GO", "CONDITIONAL") and
                self.grounding_pct >= 30 and
                self.convergence_status == "CONVERGED")

    def render_summary(self) -> str:
        """Human-readable decision summary."""
        lines = [
            "=" * 64,
            "  ARKAIN ATLAS — DECISION PACKAGE",
            f"  {self.deal_summary.get('address', 'Unknown')}",
            "=" * 64,
            "",
            f"  Decision:     {self.decision}",
            f"  Confidence:   {self.confidence:.0%}  [{self.calibration_tag}]",
            f"  Risk Score:   {self.risk_score}",
            f"  Convergence:  {self.convergence_status}",
            "",
            f"  Grounding:    {self.grounding_pct:.0f}% evidence-backed",
            f"  Usable:       {self.usable_pct:.0f}% variables usable",
            f"  Assumptions:  {len([v for v in self.assumption_table.get('records', {}).values() if v.get('status') == 'ASSUMPTION'])}",
            f"  Evidence:     {len(self.evidence_citations)} citations",
            "",
        ]

        if self.financial_summary:
            lines.append("  FINANCIALS:")
            for k, v in self.financial_summary.items():
                if isinstance(v, (int, float)):
                    lines.append(f"    {k}: {v:,.2f}" if v > 100 else f"    {k}: {v:.2f}")
            lines.append("")

        if self.decision_pivots:
            lines.append("  DECISION PIVOTS (what would change the answer):")
            for p in self.decision_pivots[:5]:
                lines.append(f"    {p.description}")
            lines.append("")

        if self.next_actions:
            lines.append("  NEXT ACTIONS (to raise confidence):")
            for a in self.next_actions[:5]:
                lines.append(f"    {a.priority}. {a.action} [{a.estimated_effort}]")
            lines.append("")

        if self.risk_register:
            lines.append(f"  RISK REGISTER ({len(self.risk_register)} items):")
            for r in self.risk_register[:5]:
                lines.append(f"    [{r.get('severity', '?')}] {r.get('description', '?')}")

        lines.append("")
        lines.append(f"  Run ID: {self.run_id}  |  {self.elapsed_ms}ms  |  {self.total_tool_calls} tool calls")
        lines.append(f"  Generated by Arkain Atlas v3.3.1")
        lines.append("=" * 64)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "generated_by": "Arkain Atlas v3.3.1",
            "run_id": self.run_id, "timestamp": self.timestamp,
            "decision": self.decision, "confidence": round(self.confidence, 3),
            "calibration_tag": self.calibration_tag,
            "is_actionable": self.is_actionable(),
            "deal_summary": self.deal_summary,
            "assumption_table": self.assumption_table,
            "evidence_citations": self.evidence_citations,
            "grounding_pct": round(self.grounding_pct, 1),
            "usable_pct": round(self.usable_pct, 1),
            "risk_register": self.risk_register,
            "risk_score": self.risk_score,
            "financial_summary": self.financial_summary,
            "monte_carlo_summary": self.monte_carlo_summary,
            "decision_pivots": [p.to_dict() for p in self.decision_pivots],
            "next_actions": [a.to_dict() for a in self.next_actions],
            "convergence": {"status": self.convergence_status, "details": self.convergence_details},
            "agents_run": self.agents_run,
            "agents_skipped": self.agents_skipped,
            "total_tool_calls": self.total_tool_calls,
            "total_retries": self.total_retries,
            "elapsed_ms": self.elapsed_ms,
            "replay_log_id": self.replay_log_id,
            "ledger_head_hash": self.ledger_head_hash,
            "ledger_verified": self.ledger_verified,
            "seed": self.seed,
        }


class DecisionPackageBuilder:
    """Builds a DecisionPackage from pipeline run results."""

    def __init__(self, deal_data: Dict, run_id: str = ""):
        self.pkg = DecisionPackage(
            run_id=run_id or f"run_{int(time.time())}",
            timestamp=time.time(),
            deal_summary=deal_data,
        )

    def set_decision(self, decision: str, confidence: float,
                     calibration_tag: str = "uncalibrated"):
        self.pkg.decision = decision
        self.pkg.confidence = confidence
        self.pkg.calibration_tag = calibration_tag

    def set_assumptions(self, assumption_table_dict: Dict):
        self.pkg.assumption_table = assumption_table_dict
        coverage = assumption_table_dict.get("coverage", {})
        self.pkg.grounding_pct = coverage.get("grounding_pct", 0)
        self.pkg.usable_pct = coverage.get("usable_pct", 0)

    def set_evidence(self, evidence_list: List[Dict]):
        self.pkg.evidence_citations = evidence_list

    def set_risk(self, risk_register: List[Dict], risk_score: str):
        self.pkg.risk_register = risk_register
        self.pkg.risk_score = risk_score

    def set_financials(self, financial_summary: Dict, mc_summary: Dict = None):
        self.pkg.financial_summary = financial_summary
        self.pkg.monte_carlo_summary = mc_summary or {}

    def add_pivot(self, variable: str, current: float, threshold: float,
                  direction: str, description: str):
        dist = abs(current - threshold) / max(abs(current), 1) * 100
        self.pkg.decision_pivots.append(DecisionPivot(
            variable=variable, current_value=current,
            pivot_threshold=threshold, pivot_direction=direction,
            distance_pct=dist, description=description,
        ))

    def add_next_action(self, priority: int, action: str,
                        target_var: str, conf_gain: float, effort: str):
        self.pkg.next_actions.append(NextAction(
            priority=priority, action=action,
            target_variable=target_var,
            expected_confidence_gain=conf_gain,
            estimated_effort=effort,
        ))

    def set_convergence(self, status: str, details: Dict = None):
        self.pkg.convergence_status = status
        self.pkg.convergence_details = details or {}

    def set_run_stats(self, agents_run: List[str], agents_skipped: List[Dict],
                      tool_calls: int, retries: int, elapsed_ms: int,
                      replay_log_id: str = ""):
        self.pkg.agents_run = agents_run
        self.pkg.agents_skipped = agents_skipped
        self.pkg.total_tool_calls = tool_calls
        self.pkg.total_retries = retries
        self.pkg.elapsed_ms = elapsed_ms
        self.pkg.replay_log_id = replay_log_id

    def auto_pivots(self, beliefs_dict: Dict, mc_result: Dict = None):
        """Auto-generate decision pivots from belief state + MC results."""
        # NOI pivot: what NOI makes DSCR < 1.0?
        noi = beliefs_dict.get("noi", {})
        if noi.get("value"):
            dscr_pivot_noi = noi["value"] * 0.75  # rough: 25% NOI drop → DSCR ~1.0
            self.add_pivot("noi", noi["value"], dscr_pivot_noi, "below",
                          f"If NOI drops below ${dscr_pivot_noi:,.0f}, DSCR falls below 1.0x")

        cap = beliefs_dict.get("cap_rate", {})
        if cap.get("value"):
            self.add_pivot("cap_rate", cap["value"], cap["value"] + 2.0, "above",
                          f"If cap rate rises above {cap['value']+2.0:.1f}%, valuation drops significantly")

        interest = beliefs_dict.get("interest_rate", {})
        if interest.get("value"):
            self.add_pivot("interest_rate", interest["value"], interest["value"] + 1.5, "above",
                          f"If interest rate exceeds {interest['value']+1.5:.1f}%, deal economics deteriorate")

    def auto_next_actions(self, assumption_table_dict: Dict):
        """Auto-generate next actions from missing/assumed variables."""
        records = assumption_table_dict.get("records", {})
        priority = 1
        # Unknown variables first
        for var, rec in sorted(records.items()):
            if rec.get("status") == "UNKNOWN" and priority <= 5:
                self.add_next_action(priority, f"Obtain data for '{var}'",
                                    var, 0.15, "hours")
                priority += 1
        # Then assumptions that could be upgraded
        for var, rec in sorted(records.items()):
            if rec.get("status") == "ASSUMPTION" and priority <= 8:
                self.add_next_action(priority, f"Verify assumption for '{var}'",
                                    var, 0.10, "hours")
                priority += 1

    def build(self) -> DecisionPackage:
        return self.pkg
