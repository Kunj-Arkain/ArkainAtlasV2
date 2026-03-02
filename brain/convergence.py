"""
engine.brain.convergence — Output Validation & Convergence Checking
=====================================================================
Every agent output passes through 3 gates before it's accepted:

  GATE 1: SCHEMA — Did the agent return the required structure?
  GATE 2: QUALITY — Are the numbers reasonable and internally consistent?
  GATE 3: CONVERGENCE — Do the numbers match what other agents produced?

If any gate fails, the agent re-runs with specific error feedback
injected into its prompt. Max 3 retries per agent.

Architecture:
  AgentRunner          — Executes an agent with retry loop
  OutputValidator      — Gate 1+2: schema + quality checks per agent
  ConvergenceChecker   — Gate 3: cross-agent number matching
  QualityGate          — Pipeline stage gates (blocks downstream if critical fail)
  ReconciliationLedger — Single source of truth for shared numbers (NOI, cap rate, etc.)

Usage:
    runner = AgentRunner(max_retries=3)
    ledger = ReconciliationLedger()

    # Run underwriting — output validated automatically
    uw_result = runner.run("underwriting_analyst", deal_data, ledger)

    # Run deal structurer — convergence-checked against underwriting
    ds_result = runner.run("deal_structurer", deal_data, ledger)

    # Check pipeline health
    ledger.report()  # shows all reconciliation status
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# ENUMS & CONSTANTS
# ═══════════════════════════════════════════════════════════════

class GateResult(Enum):
    PASS = "pass"
    WARN = "warn"       # non-blocking — flag but continue
    FAIL = "fail"       # blocking — must retry
    CRITICAL = "critical"  # hard stop — cannot proceed


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Tolerance for numerical convergence (percentage)
DEFAULT_TOLERANCE_PCT = 5.0      # 5% for most financial numbers
STRICT_TOLERANCE_PCT = 2.0       # 2% for NOI, DSCR
LOOSE_TOLERANCE_PCT = 15.0       # 15% for estimates (construction, gaming projections)

# Max retries before giving up
MAX_RETRIES = 3


# ═══════════════════════════════════════════════════════════════
# VALIDATION ISSUE
# ═══════════════════════════════════════════════════════════════

@dataclass
class ValidationIssue:
    """A single validation failure or warning."""
    agent: str
    gate: str              # "schema", "quality", "convergence"
    field: str             # e.g. "noi", "dscr", "task_3"
    severity: Severity
    message: str
    expected: Any = None
    actual: Any = None
    suggestion: str = ""   # injected into retry prompt

    def to_dict(self) -> Dict:
        return {
            "agent": self.agent, "gate": self.gate, "field": self.field,
            "severity": self.severity.value, "message": self.message,
            "expected": self.expected, "actual": self.actual,
            "suggestion": self.suggestion,
        }

    def __str__(self) -> str:
        exp = f" (expected={self.expected}, got={self.actual})" if self.expected else ""
        return f"[{self.severity.value.upper()}] {self.agent}.{self.field}: {self.message}{exp}"


# ═══════════════════════════════════════════════════════════════
# GATE 1 + 2: OUTPUT VALIDATOR
# ═══════════════════════════════════════════════════════════════

class OutputValidator:
    """Validates a single agent's output for schema completeness and quality.

    Gate 1 (Schema): Does the output contain all required fields?
    Gate 2 (Quality): Are numbers within reasonable ranges?
    """

    def __init__(self):
        self._validators: Dict[str, Callable] = {
            "acquisition_scout":     self._validate_scout,
            "site_selector":         self._validate_site_selector,
            "market_analyst":        self._validate_market,
            "underwriting_analyst":  self._validate_underwriting,
            "deal_structurer":       self._validate_structurer,
            "gaming_optimizer":      self._validate_gaming,
            "risk_officer":          self._validate_risk,
            "due_diligence":         self._validate_dd,
            "contract_redliner":     self._validate_contract,
            "tax_strategist":        self._validate_tax,
            "renovation_planner":    self._validate_renovation,
            "architect":             self._validate_architect,
            "structural_engineer":   self._validate_structural,
            "mep_engineer":          self._validate_mep,
            "spec_writer":           self._validate_specs,
            "compliance_writer":     self._validate_compliance,
            "exit_strategist":       self._validate_exit,
        }

    def validate(self, agent_name: str, output: Dict,
                 context: Dict = None) -> List[ValidationIssue]:
        """Run schema + quality validation for an agent's output."""
        issues = []

        # Universal checks
        issues.extend(self._check_universal(agent_name, output))

        # Agent-specific checks
        validator = self._validators.get(agent_name)
        if validator:
            issues.extend(validator(output, context or {}))

        return issues

    # ── Universal Checks ──────────────────────────────────

    def _check_universal(self, agent: str, output: Dict) -> List[ValidationIssue]:
        issues = []

        # Must have some output
        if not output:
            issues.append(ValidationIssue(
                agent, "schema", "output", Severity.CRITICAL,
                "Agent returned empty output",
                suggestion="Re-run agent — previous execution returned no data.",
            ))
            return issues

        # Check for task completion markers
        tasks_found = set()
        if isinstance(output, dict):
            for key in output:
                if key.lower().startswith("task"):
                    tasks_found.add(key)

        # Check for recommendation/verdict (most agents should have one)
        verdict_agents = {
            "acquisition_scout", "underwriting_analyst", "risk_officer",
            "deal_structurer", "exit_strategist",
        }
        if agent in verdict_agents:
            has_verdict = any(
                k in output for k in
                ("recommendation", "verdict", "deal_verdict", "exit_recommendation",
                 "deal_recommendation", "deal_brief")
            )
            if not has_verdict:
                # Also check if it's in the text content
                text = json.dumps(output).lower()
                has_verdict = any(w in text for w in
                    ("go/", "pass/", "approve", "reject", "hold", "no-go",
                     "conditional", "recommend"))
            if not has_verdict:
                issues.append(ValidationIssue(
                    agent, "schema", "recommendation", Severity.ERROR,
                    "Missing GO/NO-GO recommendation or verdict",
                    suggestion="You MUST include a clear recommendation: GO, HOLD, NO-GO, APPROVE, CONDITIONAL, or REJECT.",
                ))

        # Check for tool call evidence (numbers should cite sources)
        text = json.dumps(output)
        if "tool_calls" in output:
            tool_count = len(output.get("tool_calls", []))
            if tool_count == 0:
                issues.append(ValidationIssue(
                    agent, "quality", "tool_usage", Severity.WARNING,
                    "No tool calls detected — output may contain assumed numbers",
                    suggestion="Every number must come from a tool call. Do not assume or estimate without data.",
                ))

        return issues

    # ── Agent-Specific Validators ─────────────────────────

    def _validate_scout(self, output: Dict, ctx: Dict) -> List[ValidationIssue]:
        issues = []
        # Must have cap rate
        if "cap_rate" in output:
            cr = output["cap_rate"]
            if isinstance(cr, (int, float)):
                if cr < 0 or cr > 25:
                    issues.append(ValidationIssue(
                        "acquisition_scout", "quality", "cap_rate", Severity.ERROR,
                        f"Cap rate {cr}% is outside reasonable range (3-20%)",
                        expected="3-20%", actual=cr,
                        suggestion=f"Recalculate cap rate. {cr}% is unrealistic. Cap Rate = NOI / Price × 100.",
                    ))
        return issues

    def _validate_site_selector(self, output: Dict, ctx: Dict) -> List[ValidationIssue]:
        issues = []
        # Must have composite scores
        text = json.dumps(output).lower()
        required_scores = ["market", "location", "competition", "risk", "financial"]
        for score in required_scores:
            if f"{score}_score" not in text and f"{score} score" not in text:
                issues.append(ValidationIssue(
                    "site_selector", "schema", f"{score}_score", Severity.ERROR,
                    f"Missing {score} score (1-10)",
                    suggestion=f"Calculate and include {score}_score (1-10) per the scoring framework.",
                ))
        return issues

    def _validate_market(self, output: Dict, ctx: Dict) -> List[ValidationIssue]:
        issues = []
        # Must have site score and grade
        text = json.dumps(output).lower()
        if "site_score" not in text and "site score" not in text:
            issues.append(ValidationIssue(
                "market_analyst", "schema", "site_score", Severity.ERROR,
                "Missing site score (1-10)",
                suggestion="Include a site_score (1-10) and site_grade (A-F) in TASK 10 SYNTHESIS.",
            ))
        return issues

    def _validate_underwriting(self, output: Dict, ctx: Dict) -> List[ValidationIssue]:
        """Critical validator — financial numbers must be internally consistent."""
        issues = []

        noi = _extract_number(output, "noi", "net_operating_income")
        price = _extract_number(output, "price", "purchase_price", "asking_price")
        cap_rate = _extract_number(output, "cap_rate")
        dscr = _extract_number(output, "dscr")

        # NOI must exist
        if noi is None:
            issues.append(ValidationIssue(
                "underwriting_analyst", "schema", "noi", Severity.CRITICAL,
                "Missing NOI — cannot underwrite without Net Operating Income",
                suggestion="TASK 3 requires calculating verified NOI = Revenue - Vacancy - Expenses. This is mandatory.",
            ))
        elif noi < 0:
            issues.append(ValidationIssue(
                "underwriting_analyst", "quality", "noi", Severity.CRITICAL,
                f"Negative NOI (${noi:,.0f}) — property operates at a loss",
                actual=noi,
                suggestion="Verify expense and revenue figures. Negative NOI means expenses exceed revenue.",
            ))

        # Cap rate internal consistency: cap = NOI / price
        if noi and price and cap_rate:
            expected_cap = (noi / price) * 100
            if abs(expected_cap - cap_rate) > 0.5:
                issues.append(ValidationIssue(
                    "underwriting_analyst", "quality", "cap_rate_consistency", Severity.ERROR,
                    f"Cap rate ({cap_rate:.1f}%) doesn't match NOI/Price ({expected_cap:.1f}%)",
                    expected=round(expected_cap, 1), actual=cap_rate,
                    suggestion=f"Cap Rate = NOI / Price = ${noi:,.0f} / ${price:,.0f} = {expected_cap:.1f}%. Fix the inconsistency.",
                ))

        # DSCR reasonableness
        if dscr is not None:
            if dscr < 0:
                issues.append(ValidationIssue(
                    "underwriting_analyst", "quality", "dscr", Severity.CRITICAL,
                    f"Negative DSCR ({dscr:.2f}) — NOI doesn't cover debt",
                    actual=dscr,
                    suggestion="DSCR must be positive. Check NOI and annual debt service calculation.",
                ))
            elif dscr < 1.0:
                issues.append(ValidationIssue(
                    "underwriting_analyst", "quality", "dscr", Severity.WARNING,
                    f"DSCR {dscr:.2f} < 1.0 — property cannot service debt",
                    actual=dscr,
                ))

        # Must have 3 valuation approaches
        text = json.dumps(output).lower()
        for approach in ["direct cap", "dcf", "comparable"]:
            if approach not in text and approach.replace(" ", "_") not in text:
                issues.append(ValidationIssue(
                    "underwriting_analyst", "schema", f"valuation_{approach}", Severity.ERROR,
                    f"Missing {approach} valuation approach",
                    suggestion=f"TASK 4 requires 3 valuation approaches: Direct Cap, DCF, and Comparable Sales.",
                ))

        # Must have stress tests
        if "stress" not in text:
            issues.append(ValidationIssue(
                "underwriting_analyst", "schema", "stress_tests", Severity.ERROR,
                "Missing stress test results",
                suggestion="TASK 7 requires stress testing: NOI -20%, Rate +200bps, Vacancy +15%, Combined.",
            ))

        return issues

    def _validate_structurer(self, output: Dict, ctx: Dict) -> List[ValidationIssue]:
        issues = []
        text = json.dumps(output).lower()
        # Must have multiple capital stack variants
        variant_count = sum(1 for w in ["variant a", "variant b", "variant c",
                                         "max leverage", "conventional", "creative",
                                         "sba", "option 1", "option 2"]
                           if w in text)
        if variant_count < 2:
            issues.append(ValidationIssue(
                "deal_structurer", "schema", "capital_variants", Severity.ERROR,
                "Must present at least 2 capital stack variants",
                suggestion="TASK 2 requires 3 variants: (a) Max leverage SBA, (b) Conventional, (c) Creative.",
            ))
        # Must have sensitivity matrix
        if "sensitivity" not in text and "matrix" not in text:
            issues.append(ValidationIssue(
                "deal_structurer", "schema", "sensitivity_matrix", Severity.ERROR,
                "Missing sensitivity matrix",
                suggestion="TASK 7: Build 5×5 matrix varying price (±10%) and rate (±100bps).",
            ))
        return issues

    def _validate_gaming(self, output: Dict, ctx: Dict) -> List[ValidationIssue]:
        issues = []
        nti = _extract_number(output, "nti", "net_terminal_income")
        if nti is not None and nti < 0:
            issues.append(ValidationIssue(
                "gaming_optimizer", "quality", "nti", Severity.ERROR,
                f"Negative NTI (${nti:,.0f})",
                actual=nti,
                suggestion="NTI cannot be negative. Re-run egm_predict with correct parameters.",
            ))
        text = json.dumps(output).lower()
        if "operator" not in text:
            issues.append(ValidationIssue(
                "gaming_optimizer", "schema", "operator_comparison", Severity.ERROR,
                "Missing operator comparison",
                suggestion="TASK 6 requires comparing 3+ terminal operators on split, service, machines.",
            ))
        return issues

    def _validate_risk(self, output: Dict, ctx: Dict) -> List[ValidationIssue]:
        issues = []
        text = json.dumps(output).lower()
        required_categories = ["market", "credit", "regulatory", "environmental",
                               "concentration", "financial", "physical"]
        for cat in required_categories:
            if f"{cat} risk" not in text and f"{cat}_risk" not in text:
                issues.append(ValidationIssue(
                    "risk_officer", "schema", f"{cat}_risk", Severity.WARNING,
                    f"Missing {cat} risk assessment",
                    suggestion=f"TASK: Assess {cat} risk with probability, impact, and mitigation.",
                ))
        if "risk_register" not in text and "risk register" not in text and "risk id" not in text:
            issues.append(ValidationIssue(
                "risk_officer", "schema", "risk_register", Severity.ERROR,
                "Missing risk register table",
                suggestion="TASK 8: Produce risk register with ID, Category, Probability, Impact, Mitigation, Residual, Owner.",
            ))
        return issues

    def _validate_dd(self, output: Dict, ctx: Dict) -> List[ValidationIssue]:
        issues = []
        text = json.dumps(output).lower()
        # Check for completion tracking
        status_symbols = ["✅", "⚠", "❌", "⏳", "clear", "flag", "issue", "pending"]
        has_tracking = any(s in text for s in status_symbols)
        if not has_tracking:
            issues.append(ValidationIssue(
                "due_diligence", "schema", "status_tracking", Severity.ERROR,
                "Missing DD checklist status tracking (✅⚠️❌⏳)",
                suggestion="TASK 8: Each of 40 items needs status: ✅ Clear, ⚠️ Flag, ❌ Issue, ⏳ Pending.",
            ))
        # Check completion percentage
        if "completion" not in text and "%" not in text:
            issues.append(ValidationIssue(
                "due_diligence", "schema", "completion_pct", Severity.WARNING,
                "Missing completion percentage",
                suggestion="Report completion % and list of blockers.",
            ))
        return issues

    def _validate_contract(self, output: Dict, ctx: Dict) -> List[ValidationIssue]:
        issues = []
        text = json.dumps(output).lower()
        if "redline" not in text and "replacement" not in text:
            issues.append(ValidationIssue(
                "contract_redliner", "schema", "redlines", Severity.ERROR,
                "No redlines produced",
                suggestion="TASK 6: Each issue needs: quote language, explain risk, replacement language, severity.",
            ))
        return issues

    def _validate_tax(self, output: Dict, ctx: Dict) -> List[ValidationIssue]:
        issues = []
        text = json.dumps(output).lower()
        if "cost seg" not in text and "cost_seg" not in text and "depreciation" not in text:
            issues.append(ValidationIssue(
                "tax_strategist", "schema", "cost_segregation", Severity.WARNING,
                "Missing cost segregation analysis",
                suggestion="TASK 3: Separate land/building/site/personal property and calculate Year 1 tax savings.",
            ))
        return issues

    def _validate_renovation(self, output: Dict, ctx: Dict) -> List[ValidationIssue]:
        issues = []
        text = json.dumps(output).lower()
        if "roi" not in text and "return" not in text:
            issues.append(ValidationIssue(
                "renovation_planner", "schema", "roi_analysis", Severity.ERROR,
                "Missing renovation ROI analysis",
                suggestion="TASK 7: Calculate ROI = incremental NOI / cost for each renovation tier.",
            ))
        return issues

    def _validate_architect(self, output: Dict, ctx: Dict) -> List[ValidationIssue]:
        issues = []
        text = json.dumps(output).lower()
        if "code" not in text and "occupancy" not in text:
            issues.append(ValidationIssue(
                "architect", "schema", "code_analysis", Severity.ERROR,
                "Missing building code analysis",
                suggestion="TASK 1: Call code_analysis first — occupancy, sprinkler, egress drive all design.",
            ))
        return issues

    def _validate_structural(self, output: Dict, ctx: Dict) -> List[ValidationIssue]:
        issues = []
        text = json.dumps(output).lower()
        if "foundation" not in text:
            issues.append(ValidationIssue(
                "structural_engineer", "schema", "foundation", Severity.ERROR,
                "Missing foundation design",
                suggestion="TASK 3: Design spread footings from structural_calc soil bearing.",
            ))
        return issues

    def _validate_mep(self, output: Dict, ctx: Dict) -> List[ValidationIssue]:
        issues = []
        text = json.dumps(output).lower()
        if "panel" not in text and "circuit" not in text:
            issues.append(ValidationIssue(
                "mep_engineer", "schema", "electrical", Severity.ERROR,
                "Missing electrical load calc / panel schedule",
                suggestion="TASK 1: Call electrical_load_calc. Each VGT needs dedicated 20A circuit.",
            ))
        return issues

    def _validate_specs(self, output: Dict, ctx: Dict) -> List[ValidationIssue]:
        issues = []
        text = json.dumps(output).lower()
        if "division" not in text and "csi" not in text:
            issues.append(ValidationIssue(
                "spec_writer", "schema", "spec_book", Severity.ERROR,
                "Missing spec book generation",
                suggestion="TASK 6: Call generate_spec_book to produce the formatted PDF.",
            ))
        return issues

    def _validate_compliance(self, output: Dict, ctx: Dict) -> List[ValidationIssue]:
        issues = []
        text = json.dumps(output).lower()
        if "checklist" not in text and "license" not in text:
            issues.append(ValidationIssue(
                "compliance_writer", "schema", "checklist", Severity.WARNING,
                "Missing compliance checklist",
                suggestion="TASK 4: Build state-specific compliance checklist with all licenses and deadlines.",
            ))
        return issues

    def _validate_exit(self, output: Dict, ctx: Dict) -> List[ValidationIssue]:
        issues = []
        text = json.dumps(output).lower()
        # Must compare hold vs sell
        if "hold" not in text or "sell" not in text:
            issues.append(ValidationIssue(
                "exit_strategist", "schema", "hold_vs_sell", Severity.ERROR,
                "Missing hold vs sell comparison",
                suggestion="TASK 1: Compare (a) hold 5yr, (b) sell now, (c) refi + hold.",
            ))
        return issues


# ═══════════════════════════════════════════════════════════════
# GATE 3: CONVERGENCE CHECKER
# ═══════════════════════════════════════════════════════════════

class ConvergenceChecker:
    """Cross-checks numbers between agents for consistency.

    When Agent B uses a number that Agent A produced, they must match
    within tolerance. If they don't, one of them re-runs.

    Tracked fields (shared across agents):
      - noi: underwriting → structurer, risk, gaming, tax, exit
      - cap_rate: underwriting → scout, market, structurer, exit
      - dscr: underwriting → structurer, risk
      - price/valuation: underwriting → structurer, exit
      - gaming_nti: gaming → underwriting, risk, tax
      - terminal_count: gaming → mep, architect, compliance
      - sqft: market → architect, mep, structural, specs
      - occupant_load: architect → mep, plumbing
    """

    # Define which fields must converge between which agents
    CONVERGENCE_RULES = [
        # (field_name, source_agent, consuming_agents, tolerance_pct)
        ("noi", "underwriting_analyst",
         ["deal_structurer", "risk_officer", "gaming_optimizer", "tax_strategist", "exit_strategist"],
         STRICT_TOLERANCE_PCT),

        ("cap_rate", "underwriting_analyst",
         ["acquisition_scout", "market_analyst", "deal_structurer", "exit_strategist"],
         DEFAULT_TOLERANCE_PCT),

        ("dscr", "underwriting_analyst",
         ["deal_structurer", "risk_officer"],
         STRICT_TOLERANCE_PCT),

        ("purchase_price", "underwriting_analyst",
         ["deal_structurer", "tax_strategist", "exit_strategist"],
         STRICT_TOLERANCE_PCT),

        ("gaming_net_revenue", "gaming_optimizer",
         ["underwriting_analyst", "risk_officer", "tax_strategist"],
         LOOSE_TOLERANCE_PCT),

        ("terminal_count", "gaming_optimizer",
         ["mep_engineer", "architect", "compliance_writer"],
         0.0),  # must be exact

        ("sqft", "market_analyst",
         ["architect", "mep_engineer", "structural_engineer", "spec_writer"],
         0.0),  # must be exact

        ("occupant_load", "architect",
         ["mep_engineer"],
         0.0),  # must be exact (code-driven)

        ("electrical_service_amps", "mep_engineer",
         ["spec_writer"],
         0.0),

        ("hvac_tonnage", "mep_engineer",
         ["spec_writer"],
         DEFAULT_TOLERANCE_PCT),

        ("construction_cost", "renovation_planner",
         ["deal_structurer", "underwriting_analyst"],
         LOOSE_TOLERANCE_PCT),
    ]

    def check(self, agent_name: str, output: Dict,
              ledger: "ReconciliationLedger") -> List[ValidationIssue]:
        """Check an agent's output against the reconciliation ledger."""
        issues = []

        for field_name, source_agent, consumers, tolerance in self.CONVERGENCE_RULES:
            # If this agent is a consumer of this field, check against source
            if agent_name in consumers:
                source_value = ledger.get(field_name)
                if source_value is not None:
                    agent_value = _extract_number(output, field_name)
                    if agent_value is not None:
                        if not _within_tolerance(source_value, agent_value, tolerance):
                            pct_diff = abs(agent_value - source_value) / max(abs(source_value), 1) * 100
                            issues.append(ValidationIssue(
                                agent_name, "convergence", field_name, Severity.ERROR,
                                f"{field_name} = {agent_value:,.2f} doesn't match "
                                f"{source_agent}'s value of {source_value:,.2f} "
                                f"(diff: {pct_diff:.1f}%, tolerance: {tolerance}%)",
                                expected=source_value, actual=agent_value,
                                suggestion=f"Use {field_name} = {source_value:,.2f} from {source_agent}. "
                                           f"Your value of {agent_value:,.2f} is {pct_diff:.1f}% off.",
                            ))

            # If this agent is the source of this field, register it
            if agent_name == source_agent:
                value = _extract_number(output, field_name)
                if value is not None:
                    ledger.set(field_name, value, agent_name)

        return issues


# ═══════════════════════════════════════════════════════════════
# RECONCILIATION LEDGER
# ═══════════════════════════════════════════════════════════════

@dataclass
class LedgerEntry:
    field: str
    value: float
    source_agent: str
    timestamp: float = 0
    version: int = 1

    def to_dict(self) -> Dict:
        return {
            "field": self.field, "value": self.value,
            "source": self.source_agent, "version": self.version,
        }


class ReconciliationLedger:
    """Single source of truth for shared numbers across all agents.

    When an upstream agent (e.g., underwriting) produces NOI,
    it's registered here. All downstream agents must use this value.
    If they produce a different number, convergence fails.
    """

    def __init__(self):
        self._entries: Dict[str, LedgerEntry] = {}
        self._history: List[Dict] = []

    def set(self, field: str, value: float, source_agent: str):
        """Register a canonical value from a source agent."""
        version = 1
        if field in self._entries:
            version = self._entries[field].version + 1
        entry = LedgerEntry(field, value, source_agent, time.time(), version)
        self._entries[field] = entry
        self._history.append({
            "action": "set", "field": field, "value": value,
            "source": source_agent, "version": version,
        })
        logger.info(f"Ledger: {field} = {value:,.2f} (from {source_agent} v{version})")

    def get(self, field: str) -> Optional[float]:
        """Get the canonical value for a field."""
        entry = self._entries.get(field)
        return entry.value if entry else None

    def get_entry(self, field: str) -> Optional[LedgerEntry]:
        return self._entries.get(field)

    def has(self, field: str) -> bool:
        return field in self._entries

    def all_entries(self) -> Dict[str, Dict]:
        return {k: v.to_dict() for k, v in self._entries.items()}

    def report(self) -> Dict:
        """Full reconciliation report."""
        return {
            "field_count": len(self._entries),
            "entries": self.all_entries(),
            "history": self._history[-50:],  # last 50 changes
        }

    def inject_into_prompt(self) -> str:
        """Generate a prompt fragment that tells the agent what numbers to use."""
        if not self._entries:
            return ""
        lines = ["\n## RECONCILIATION LEDGER — USE THESE NUMBERS ##"]
        lines.append("The following values have been verified by upstream agents. YOU MUST USE THEM:")
        for field, entry in self._entries.items():
            lines.append(
                f"  {field} = {entry.value:,.2f}  (source: {entry.source_agent})"
            )
        lines.append("If your calculations produce different numbers, explain the discrepancy.\n")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# QUALITY GATE
# ═══════════════════════════════════════════════════════════════

class QualityGate:
    """Pipeline-level quality gate that blocks downstream agents
    if upstream quality is insufficient.

    Gates:
      SCREENING → ANALYSIS: Scout must produce GO recommendation
      ANALYSIS → STRUCTURING: Underwriting must have valid NOI + DSCR
      STRUCTURING → CONSTRUCTION: Structurer must have recommended variant
      CONSTRUCTION → COMPLIANCE: Architect must have code analysis
    """

    GATES = {
        "screening_to_analysis": {
            "required_agents": ["acquisition_scout"],
            "blocking_severities": [Severity.CRITICAL],
            "description": "Scout must pass before deeper analysis",
        },
        "analysis_to_structuring": {
            "required_agents": ["underwriting_analyst"],
            "blocking_severities": [Severity.CRITICAL, Severity.ERROR],
            "required_fields": ["noi", "cap_rate"],
            "description": "Underwriting must produce valid NOI before structuring",
        },
        "structuring_to_construction": {
            "required_agents": ["deal_structurer"],
            "blocking_severities": [Severity.CRITICAL],
            "description": "Deal structure must be defined before construction planning",
        },
        "construction_to_compliance": {
            "required_agents": ["architect"],
            "blocking_severities": [Severity.CRITICAL, Severity.ERROR],
            "required_fields": ["occupant_load", "sqft"],
            "description": "Architect code analysis required before compliance docs",
        },
    }

    def check_gate(self, gate_name: str, agent_results: Dict[str, Dict],
                   all_issues: List[ValidationIssue],
                   ledger: ReconciliationLedger) -> Tuple[GateResult, List[str]]:
        """Check if a pipeline gate allows proceeding.

        Returns: (result, list of blocking reasons)
        """
        gate = self.GATES.get(gate_name)
        if not gate:
            return GateResult.PASS, []

        reasons = []

        # Check required agents have run
        for agent in gate["required_agents"]:
            if agent not in agent_results:
                reasons.append(f"{agent} has not run yet")

        # Check for blocking issues
        blocking = gate["blocking_severities"]
        for issue in all_issues:
            if issue.agent in gate["required_agents"] and issue.severity in blocking:
                reasons.append(f"{issue.agent}: {issue.message}")

        # Check required ledger fields
        for field in gate.get("required_fields", []):
            if not ledger.has(field):
                reasons.append(f"Missing {field} in reconciliation ledger")

        if reasons:
            return GateResult.FAIL, reasons
        return GateResult.PASS, []


# ═══════════════════════════════════════════════════════════════
# AGENT RUNNER (with retry loop)
# ═══════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    """Result of running an agent through the validation pipeline."""
    agent_name: str
    output: Dict
    issues: List[ValidationIssue]
    retries: int
    converged: bool
    gate_result: GateResult
    duration_ms: int = 0

    @property
    def has_errors(self) -> bool:
        return any(i.severity in (Severity.ERROR, Severity.CRITICAL) for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity in (Severity.ERROR, Severity.CRITICAL))

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    def to_dict(self) -> Dict:
        return {
            "agent": self.agent_name,
            "converged": self.converged,
            "retries": self.retries,
            "gate_result": self.gate_result.value,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
            "duration_ms": self.duration_ms,
        }


class AgentRunner:
    """Runs an agent with validation and automatic retry on failure.

    Flow per agent:
      1. Build prompt (system + ledger injection + deal data + retry feedback)
      2. Execute agent (LLM call with tools)
      3. Parse output
      4. Gate 1: Schema validation
      5. Gate 2: Quality validation
      6. Gate 3: Convergence check against ledger
      7. If FAIL → inject error feedback, retry (up to MAX_RETRIES)
      8. If PASS → register values in ledger, return result
    """

    def __init__(
        self,
        execute_fn: Callable = None,
        max_retries: int = MAX_RETRIES,
        validator: OutputValidator = None,
        convergence: ConvergenceChecker = None,
        quality_gate: QualityGate = None,
    ):
        self.execute_fn = execute_fn or self._default_execute
        self.max_retries = max_retries
        self.validator = validator or OutputValidator()
        self.convergence = convergence or ConvergenceChecker()
        self.quality_gate = quality_gate or QualityGate()

    def run(
        self,
        agent_name: str,
        deal_data: Dict,
        ledger: ReconciliationLedger,
        context: Dict = None,
    ) -> RunResult:
        """Run an agent with full validation and retry loop."""
        start = time.perf_counter()
        context = context or {}
        all_issues: List[ValidationIssue] = []
        output = {}

        for attempt in range(self.max_retries + 1):
            # Build retry feedback from previous issues
            retry_feedback = ""
            if attempt > 0 and all_issues:
                retry_feedback = self._build_retry_prompt(all_issues, attempt)
                logger.info(f"Retry {attempt}/{self.max_retries} for {agent_name}")

            # Execute agent
            try:
                output = self.execute_fn(
                    agent_name=agent_name,
                    deal_data=deal_data,
                    ledger_injection=ledger.inject_into_prompt(),
                    retry_feedback=retry_feedback,
                    context=context,
                )
            except Exception as e:
                all_issues.append(ValidationIssue(
                    agent_name, "execution", "runtime", Severity.CRITICAL,
                    f"Agent execution failed: {str(e)}",
                    suggestion="Fix the execution error and retry.",
                ))
                continue

            # Gate 1 + 2: Schema + Quality validation
            all_issues = self.validator.validate(agent_name, output, context)

            # Gate 3: Convergence check
            conv_issues = self.convergence.check(agent_name, output, ledger)
            all_issues.extend(conv_issues)

            # Check if we can proceed
            critical = [i for i in all_issues if i.severity == Severity.CRITICAL]
            errors = [i for i in all_issues if i.severity == Severity.ERROR]

            if not critical and not errors:
                # CONVERGED — all gates passed
                elapsed = int((time.perf_counter() - start) * 1000)
                return RunResult(
                    agent_name=agent_name, output=output,
                    issues=all_issues, retries=attempt,
                    converged=True, gate_result=GateResult.PASS,
                    duration_ms=elapsed,
                )

            if not critical and attempt == self.max_retries:
                # Errors but max retries — pass with warnings
                elapsed = int((time.perf_counter() - start) * 1000)
                return RunResult(
                    agent_name=agent_name, output=output,
                    issues=all_issues, retries=attempt,
                    converged=False, gate_result=GateResult.WARN,
                    duration_ms=elapsed,
                )

        # Failed after all retries
        elapsed = int((time.perf_counter() - start) * 1000)
        return RunResult(
            agent_name=agent_name, output=output,
            issues=all_issues, retries=self.max_retries,
            converged=False, gate_result=GateResult.FAIL,
            duration_ms=elapsed,
        )

    def _build_retry_prompt(self, issues: List[ValidationIssue], attempt: int) -> str:
        """Build a prompt fragment that tells the agent what to fix."""
        lines = [
            f"\n## RETRY {attempt}/{self.max_retries} — FIX THESE ISSUES ##",
            f"Your previous output had {len(issues)} issue(s). Fix ALL of them:\n",
        ]
        for i, issue in enumerate(issues, 1):
            lines.append(f"  ISSUE {i} [{issue.severity.value.upper()}] — {issue.field}:")
            lines.append(f"    Problem: {issue.message}")
            if issue.suggestion:
                lines.append(f"    Fix: {issue.suggestion}")
            if issue.expected is not None:
                lines.append(f"    Expected: {issue.expected}")
            lines.append("")

        lines.append("Address EVERY issue above. Do not skip any.\n")
        return "\n".join(lines)

    def _default_execute(self, **kwargs) -> Dict:
        """Placeholder — replaced by actual LLM execution in production."""
        return {"status": "placeholder", "message": "Replace with actual agent execution"}


# ═══════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR (with convergence)
# ═══════════════════════════════════════════════════════════════

class ValidatedPipeline:
    """Runs the full agent pipeline with convergence checking.

    Usage:
        pipeline = ValidatedPipeline(execute_fn=my_llm_caller)
        results = pipeline.run(deal_data, config)
        report = pipeline.convergence_report()
    """

    def __init__(self, execute_fn: Callable = None, max_retries: int = 3):
        self.runner = AgentRunner(execute_fn=execute_fn, max_retries=max_retries)
        self.ledger = ReconciliationLedger()
        self.results: Dict[str, RunResult] = {}
        self.all_issues: List[ValidationIssue] = []

    def run(self, deal_data: Dict, active_agents: List[str],
            context: Dict = None) -> Dict:
        """Execute the full pipeline with validation."""
        context = context or {}
        pipeline_start = time.perf_counter()

        for agent_name in active_agents:
            # Run agent with validation + retry
            result = self.runner.run(agent_name, deal_data, self.ledger, context)
            self.results[agent_name] = result
            self.all_issues.extend(result.issues)

            # Log result
            status = "✅" if result.converged else ("⚠️" if result.gate_result == GateResult.WARN else "❌")
            logger.info(
                f"{status} {agent_name}: {result.error_count} errors, "
                f"{result.warning_count} warnings, {result.retries} retries, "
                f"{result.duration_ms}ms"
            )

            # Check quality gates
            if agent_name == "acquisition_scout":
                gate_result, reasons = self.runner.quality_gate.check_gate(
                    "screening_to_analysis", self.results, self.all_issues, self.ledger
                )
                if gate_result == GateResult.FAIL:
                    logger.warning(f"Gate FAIL at screening: {reasons}")

            elif agent_name == "underwriting_analyst":
                gate_result, reasons = self.runner.quality_gate.check_gate(
                    "analysis_to_structuring", self.results, self.all_issues, self.ledger
                )
                if gate_result == GateResult.FAIL:
                    logger.warning(f"Gate FAIL at analysis: {reasons}")

        elapsed = int((time.perf_counter() - pipeline_start) * 1000)
        return self.convergence_report(elapsed)

    def convergence_report(self, elapsed_ms: int = 0) -> Dict:
        """Generate the full convergence report."""
        total_errors = sum(r.error_count for r in self.results.values())
        total_warnings = sum(r.warning_count for r in self.results.values())
        total_retries = sum(r.retries for r in self.results.values())
        converged_count = sum(1 for r in self.results.values() if r.converged)

        return {
            "pipeline_health": "CONVERGED" if total_errors == 0 else (
                "WARNINGS" if total_errors < 3 else "FAILED"
            ),
            "agents_run": len(self.results),
            "agents_converged": converged_count,
            "agents_failed": len(self.results) - converged_count,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "total_retries": total_retries,
            "elapsed_ms": elapsed_ms,
            "agent_results": {
                name: result.to_dict()
                for name, result in self.results.items()
            },
            "reconciliation_ledger": self.ledger.report(),
            "issues_by_severity": {
                "critical": [i.to_dict() for i in self.all_issues if i.severity == Severity.CRITICAL],
                "error": [i.to_dict() for i in self.all_issues if i.severity == Severity.ERROR],
                "warning": [i.to_dict() for i in self.all_issues if i.severity == Severity.WARNING],
            },
        }


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _extract_number(data: Dict, *field_names) -> Optional[float]:
    """Try to extract a numeric value from output using multiple possible field names."""
    if not isinstance(data, dict):
        return None

    for name in field_names:
        # Direct lookup
        if name in data:
            val = data[name]
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                try:
                    cleaned = val.replace(",", "").replace("$", "").replace("%", "").strip()
                    return float(cleaned)
                except (ValueError, TypeError):
                    pass

        # Nested search (one level deep)
        for key, val in data.items():
            if isinstance(val, dict) and name in val:
                nested = val[name]
                if isinstance(nested, (int, float)):
                    return float(nested)

    # Deep search in JSON text
    text = json.dumps(data).lower()
    for name in field_names:
        # Look for "field_name": 12345 pattern
        import re
        pattern = rf'"{name.lower()}":\s*([\d,]+\.?\d*)'
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass

    return None


def _within_tolerance(expected: float, actual: float, tolerance_pct: float) -> bool:
    """Check if actual is within tolerance_pct of expected."""
    if tolerance_pct == 0:
        return expected == actual
    if expected == 0:
        return actual == 0
    pct_diff = abs(actual - expected) / abs(expected) * 100
    return pct_diff <= tolerance_pct
