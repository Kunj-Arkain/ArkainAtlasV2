"""
engine.brain.seal_ceca — SEAL Opportunity Detection + CECA Critique
=====================================================================

SEAL (Sense-Evaluate-Act-Learn):
  Sense    — Environmental scanning: market shifts, policy changes, distressed sellers
  Evaluate — Opportunity scoring: is this a real edge or noise?
  Act      — Prioritized action with conviction levels
  Learn    — Post-hoc outcome tracking to calibrate future sensing

CECA (Cognitive-Emotional Checklist Architecture):
  Cognitive appraisal  — "Is this deal what it appears to be?"
  Emotional weighting  — "Am I anchoring on seller's ask? Fear of missing out?"
  Critique generation  — Devil's advocate: "What would make this deal fail?"
  Bias detection       — "Which cognitive biases might distort my analysis?"

These run during the ORIENT phase of OODA — after we've Observed (gathered data)
but before we Decide. They're the "pause and think critically" step.

Also includes Monte Carlo simulator for outcome modeling.
"""

from __future__ import annotations

import json
import math
import random
try:
    from .determinism import get_rng
except ImportError:
    from determinism import get_rng
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# SEAL: OPPORTUNITY DETECTION
# ═══════════════════════════════════════════════════════════════

class SignalType(Enum):
    OPPORTUNITY = "opportunity"
    THREAT = "threat"
    NEUTRAL = "neutral"
    NOISE = "noise"


@dataclass
class Signal:
    """A market signal detected during environmental scanning."""
    signal_type: SignalType
    category: str       # "market", "regulatory", "competitive", "financial", "gaming"
    description: str
    strength: float     # 0-1 (how strong is this signal?)
    confidence: float   # 0-1 (how confident are we it's real?)
    source: str         # where did we detect this?
    impact_estimate: float = 0.0  # estimated $ impact
    time_sensitivity: str = "low"  # low/medium/high/critical
    actionable: bool = True

    @property
    def salience(self) -> float:
        """Signal strength × confidence — is this worth paying attention to?"""
        return self.strength * self.confidence

    def to_dict(self) -> Dict:
        return {
            "type": self.signal_type.value, "category": self.category,
            "description": self.description,
            "strength": round(self.strength, 2),
            "confidence": round(self.confidence, 2),
            "salience": round(self.salience, 2),
            "impact_estimate": self.impact_estimate,
            "time_sensitivity": self.time_sensitivity,
            "actionable": self.actionable, "source": self.source,
        }


class SEALDetector:
    """SEAL-style opportunity/threat detection engine.

    Scans deal data and market observations for signals that
    indicate hidden value, hidden risk, or time-sensitive action.
    """

    def __init__(self):
        self.signals: List[Signal] = []
        self.signal_history: List[Dict] = []

    def sense(self, deal_data: Dict, market_data: Dict = None,
              agent_outputs: Dict = None) -> List[Signal]:
        """SENSE: Scan all available data for signals.

        Runs pattern detectors across deal, market, and agent outputs.
        """
        self.signals = []
        market_data = market_data or {}
        agent_outputs = agent_outputs or {}

        # Financial signal detection
        self._detect_financial_signals(deal_data, agent_outputs)

        # Market signal detection
        self._detect_market_signals(deal_data, market_data, agent_outputs)

        # Gaming signal detection
        self._detect_gaming_signals(deal_data, agent_outputs)

        # Regulatory signal detection
        self._detect_regulatory_signals(deal_data, agent_outputs)

        # Competitive signal detection
        self._detect_competitive_signals(deal_data, agent_outputs)

        # Distress signal detection
        self._detect_distress_signals(deal_data, agent_outputs)

        return self.signals

    def evaluate(self) -> Dict:
        """EVALUATE: Score the opportunity quality.

        Returns overall deal signal assessment.
        """
        if not self.signals:
            return {"score": 0.5, "grade": "C", "signals": 0,
                    "opportunities": 0, "threats": 0, "assessment": "Insufficient data"}

        opportunities = [s for s in self.signals if s.signal_type == SignalType.OPPORTUNITY]
        threats = [s for s in self.signals if s.signal_type == SignalType.THREAT]
        noise = [s for s in self.signals if s.signal_type == SignalType.NOISE]

        # Weighted opportunity score
        opp_score = sum(s.salience * (1 + s.impact_estimate / 1000000) for s in opportunities)
        threat_score = sum(s.salience * (1 + abs(s.impact_estimate) / 1000000) for s in threats)

        # Net score normalized to 0-1
        raw_score = (opp_score - threat_score * 0.7) / max(opp_score + threat_score, 1)
        score = max(0, min(1.0, (raw_score + 1) / 2))  # rescale [-1,1] → [0,1]

        # Grade
        if score >= 0.8: grade = "A"
        elif score >= 0.65: grade = "B"
        elif score >= 0.5: grade = "C"
        elif score >= 0.35: grade = "D"
        else: grade = "F"

        # Time-sensitive actions
        urgent = [s for s in self.signals if s.time_sensitivity in ("high", "critical") and s.actionable]

        return {
            "score": round(score, 3),
            "grade": grade,
            "total_signals": len(self.signals),
            "opportunities": len(opportunities),
            "threats": len(threats),
            "noise": len(noise),
            "opportunity_score": round(opp_score, 2),
            "threat_score": round(threat_score, 2),
            "urgent_actions": [s.to_dict() for s in urgent],
            "top_opportunities": [s.to_dict() for s in
                sorted(opportunities, key=lambda s: s.salience, reverse=True)[:5]],
            "top_threats": [s.to_dict() for s in
                sorted(threats, key=lambda s: s.salience, reverse=True)[:5]],
            "assessment": self._generate_assessment(score, opportunities, threats, urgent),
        }

    def _generate_assessment(self, score, opps, threats, urgent) -> str:
        parts = []
        if score >= 0.7:
            parts.append(f"Strong opportunity with {len(opps)} positive signals.")
        elif score >= 0.5:
            parts.append(f"Moderate opportunity. {len(opps)} positives vs {len(threats)} risks.")
        else:
            parts.append(f"Weak opportunity. {len(threats)} threats dominate {len(opps)} positives.")
        if urgent:
            parts.append(f"URGENT: {len(urgent)} time-sensitive actions required.")
        return " ".join(parts)

    # ── Signal Detectors ──────────────────────────────────

    def _detect_financial_signals(self, deal: Dict, outputs: Dict):
        price = deal.get("price", deal.get("purchase_price", 0))
        noi = _get_nested(outputs, "underwriting_analyst", "noi")

        # Below-market cap rate (opportunity)
        cap_rate = _get_nested(outputs, "underwriting_analyst", "cap_rate")
        market_cap = _get_nested(outputs, "market_analyst", "market_cap_rate")
        if cap_rate and market_cap and cap_rate > market_cap + 1.0:
            self.signals.append(Signal(
                SignalType.OPPORTUNITY, "financial",
                f"Cap rate {cap_rate:.1f}% exceeds market average {market_cap:.1f}% by {cap_rate-market_cap:.1f}%",
                strength=min(1.0, (cap_rate - market_cap) / 3.0),
                confidence=0.7, source="underwriting vs market",
                impact_estimate=(cap_rate - market_cap) / 100 * price if price else 0,
            ))
        elif cap_rate and market_cap and cap_rate < market_cap - 1.0:
            self.signals.append(Signal(
                SignalType.THREAT, "financial",
                f"Cap rate {cap_rate:.1f}% below market {market_cap:.1f}% — overpaying?",
                strength=min(1.0, (market_cap - cap_rate) / 3.0),
                confidence=0.6, source="underwriting vs market",
                impact_estimate=-(market_cap - cap_rate) / 100 * price if price else 0,
            ))

        # DSCR warning
        dscr = _get_nested(outputs, "underwriting_analyst", "dscr")
        if dscr and dscr < 1.15:
            self.signals.append(Signal(
                SignalType.THREAT, "financial",
                f"Thin DSCR ({dscr:.2f}x) — minimal debt service cushion",
                strength=0.8, confidence=0.8, source="underwriting",
                impact_estimate=-50000,
            ))
        elif dscr and dscr > 1.6:
            self.signals.append(Signal(
                SignalType.OPPORTUNITY, "financial",
                f"Strong DSCR ({dscr:.2f}x) — room for additional leverage or value-add",
                strength=0.6, confidence=0.8, source="underwriting",
            ))

    def _detect_market_signals(self, deal: Dict, market: Dict, outputs: Dict):
        # Population growth
        pop_growth = _get_nested(outputs, "market_analyst", "population_growth")
        if pop_growth and pop_growth > 2.0:
            self.signals.append(Signal(
                SignalType.OPPORTUNITY, "market",
                f"Above-average population growth ({pop_growth:.1f}%)",
                strength=min(1.0, pop_growth / 5.0), confidence=0.7,
                source="census",
            ))
        elif pop_growth and pop_growth < -1.0:
            self.signals.append(Signal(
                SignalType.THREAT, "market",
                f"Population decline ({pop_growth:.1f}%)",
                strength=min(1.0, abs(pop_growth) / 3.0), confidence=0.7,
                source="census",
            ))

        # Income levels
        median_income = _get_nested(outputs, "market_analyst", "median_income")
        if median_income and median_income > 75000:
            self.signals.append(Signal(
                SignalType.OPPORTUNITY, "market",
                f"High median income (${median_income:,.0f}) — strong consumer spending",
                strength=0.5, confidence=0.7, source="census",
            ))

    def _detect_gaming_signals(self, deal: Dict, outputs: Dict):
        gaming = deal.get("gaming_eligible", False)
        if not gaming:
            return

        nti = _get_nested(outputs, "gaming_optimizer", "nti_per_terminal")
        market_nti = _get_nested(outputs, "gaming_optimizer", "market_avg_nti")

        if nti and market_nti and nti > market_nti * 1.2:
            self.signals.append(Signal(
                SignalType.OPPORTUNITY, "gaming",
                f"NTI ${nti:.0f}/day exceeds market avg ${market_nti:.0f} by {(nti/market_nti-1)*100:.0f}%",
                strength=0.7, confidence=0.6, source="egm_predict",
                impact_estimate=(nti - market_nti) * 365 * deal.get("terminal_count", 5),
            ))

        terminal_cap = _get_nested(outputs, "gaming_optimizer", "state_terminal_cap")
        terminals = deal.get("terminal_count", 0)
        if terminal_cap and terminals and terminals < terminal_cap:
            self.signals.append(Signal(
                SignalType.OPPORTUNITY, "gaming",
                f"Room for {terminal_cap - terminals} additional terminals (at cap: {terminal_cap})",
                strength=0.5, confidence=0.8, source="state_context",
            ))

    def _detect_regulatory_signals(self, deal: Dict, outputs: Dict):
        text = json.dumps(outputs).lower()
        # Gaming legislation risk
        if any(w in text for w in ["pending legislation", "proposed ban", "moratorium"]):
            self.signals.append(Signal(
                SignalType.THREAT, "regulatory",
                "Pending gaming legislation detected — revenue at risk",
                strength=0.7, confidence=0.5, source="news/regulatory",
                time_sensitivity="high",
            ))
        # Opportunity zone
        if any(w in text for w in ["opportunity zone", "qoz", "qualified opportunity"]):
            self.signals.append(Signal(
                SignalType.OPPORTUNITY, "regulatory",
                "Property in Qualified Opportunity Zone — tax benefits available",
                strength=0.6, confidence=0.7, source="tax analysis",
                impact_estimate=50000,
            ))

    def _detect_competitive_signals(self, deal: Dict, outputs: Dict):
        competitors = _get_nested(outputs, "market_analyst", "competitor_count")
        if competitors is not None:
            if competitors <= 2:
                self.signals.append(Signal(
                    SignalType.OPPORTUNITY, "competitive",
                    f"Low competition ({competitors} competitors in area)",
                    strength=0.6, confidence=0.7, source="competitor_scan",
                ))
            elif competitors >= 8:
                self.signals.append(Signal(
                    SignalType.THREAT, "competitive",
                    f"High saturation ({competitors} competitors in area)",
                    strength=0.6, confidence=0.7, source="competitor_scan",
                ))

    def _detect_distress_signals(self, deal: Dict, outputs: Dict):
        text = json.dumps(deal).lower()
        distress_keywords = ["motivated seller", "estate sale", "foreclosure",
                             "bank owned", "reo", "divorce", "tax lien", "distressed"]
        for kw in distress_keywords:
            if kw in text:
                self.signals.append(Signal(
                    SignalType.OPPORTUNITY, "financial",
                    f"Distress indicator: '{kw}' — potential below-market pricing",
                    strength=0.8, confidence=0.5, source="deal_data",
                    time_sensitivity="high",
                ))
                break


# ═══════════════════════════════════════════════════════════════
# CECA: COGNITIVE-EMOTIONAL CRITIQUE
# ═══════════════════════════════════════════════════════════════

class BiasType(Enum):
    ANCHORING = "anchoring"
    CONFIRMATION = "confirmation_bias"
    OPTIMISM = "optimism_bias"
    SUNK_COST = "sunk_cost"
    HERDING = "herding"
    AVAILABILITY = "availability_bias"
    OVERCONFIDENCE = "overconfidence"
    FRAMING = "framing_effect"
    STATUS_QUO = "status_quo_bias"
    RECENCY = "recency_bias"


@dataclass
class CritiqueFinding:
    """A single critique or bias detection."""
    category: str           # "cognitive", "emotional", "structural"
    finding: str
    severity: str           # "low", "medium", "high", "critical"
    bias_type: Optional[BiasType] = None
    counter_argument: str = ""
    suggested_action: str = ""

    def to_dict(self) -> Dict:
        return {
            "category": self.category, "finding": self.finding,
            "severity": self.severity,
            "bias": self.bias_type.value if self.bias_type else None,
            "counter_argument": self.counter_argument,
            "suggested_action": self.suggested_action,
        }


class CECACritic:
    """Cognitive-Emotional Checklist Architecture.

    The internal devil's advocate. Runs during ORIENT to challenge
    the analysis before any decision is made.

    Three layers:
      1. Cognitive Appraisal — challenge factual assumptions
      2. Emotional Weighting — detect decision-distorting emotions
      3. Pre-mortem — "Imagine this deal failed. Why?"
    """

    def critique(self, deal_data: Dict, agent_outputs: Dict,
                 beliefs: "BeliefState" = None,
                 signals: List[Signal] = None) -> Dict:
        """Run full CECA critique.

        Returns critique report with findings, bias detections,
        and counter-arguments.
        """
        findings: List[CritiqueFinding] = []

        # Layer 1: Cognitive appraisal
        findings.extend(self._cognitive_appraisal(deal_data, agent_outputs))

        # Layer 2: Emotional weighting / bias detection
        findings.extend(self._emotional_check(deal_data, agent_outputs))

        # Layer 3: Pre-mortem
        findings.extend(self._pre_mortem(deal_data, agent_outputs, signals))

        # Layer 4: Uncertainty audit
        if beliefs:
            findings.extend(self._uncertainty_audit(beliefs))

        # Score overall critique
        critical = [f for f in findings if f.severity == "critical"]
        high = [f for f in findings if f.severity == "high"]
        medium = [f for f in findings if f.severity == "medium"]

        # Override confidence if too many critiques
        confidence_penalty = len(critical) * 0.15 + len(high) * 0.08 + len(medium) * 0.03
        adjusted_confidence = max(0, 1.0 - confidence_penalty)

        return {
            "total_findings": len(findings),
            "critical": len(critical),
            "high": len(high),
            "medium": len(medium),
            "adjusted_confidence": round(adjusted_confidence, 2),
            "proceed_recommendation": "PROCEED" if not critical and len(high) < 3 else (
                "PROCEED WITH CAUTION" if not critical else "HALT AND REVIEW"
            ),
            "findings": [f.to_dict() for f in findings],
            "bias_detections": [f.to_dict() for f in findings if f.bias_type],
            "pre_mortem_scenarios": [f.to_dict() for f in findings if f.category == "pre_mortem"],
        }

    def _cognitive_appraisal(self, deal: Dict, outputs: Dict) -> List[CritiqueFinding]:
        """Challenge factual assumptions in the analysis."""
        findings = []

        # Check: Are revenue projections based on actual data or assumptions?
        noi = _get_nested(outputs, "underwriting_analyst", "noi")
        if noi and not _get_nested(outputs, "underwriting_analyst", "rent_roll_verified"):
            findings.append(CritiqueFinding(
                "cognitive", "NOI may be based on proforma rather than actual financials",
                "high", BiasType.OPTIMISM,
                counter_argument="Proforma NOI often overstates by 10-20%. Demand trailing 12-month actuals.",
                suggested_action="Request actual P&L statements for last 24 months before committing.",
            ))

        # Check: Is comp data representative?
        comp_count = _get_nested(outputs, "underwriting_analyst", "comp_count")
        if comp_count is not None and comp_count < 3:
            findings.append(CritiqueFinding(
                "cognitive", f"Only {comp_count} comparable sales — thin evidence for valuation",
                "medium", BiasType.AVAILABILITY,
                counter_argument="Fewer than 5 comps means the valuation range could be ±15% wider.",
                suggested_action="Expand comp search radius or use income approach as primary valuation.",
            ))

        # Check: Gaming revenue assumptions
        gaming_rev = _get_nested(outputs, "gaming_optimizer", "gaming_revenue")
        if gaming_rev and gaming_rev > 200000:
            findings.append(CritiqueFinding(
                "cognitive", f"Gaming revenue projection ${gaming_rev:,.0f}/yr — is this realistic for the location?",
                "medium", BiasType.OPTIMISM,
                counter_argument="NTI predictions are highly variable. Actual performance depends on foot traffic, demographics, and competition.",
                suggested_action="Use 70% of projected NTI as base case. Build downside scenario at 50%.",
            ))

        # Check: Single-source dependency
        text = json.dumps(outputs).lower()
        if "single tenant" in text or "sole tenant" in text:
            findings.append(CritiqueFinding(
                "cognitive", "Single-tenant dependency — if they leave, 100% vacancy",
                "high", None,
                counter_argument="Single-tenant properties have binary risk. Tenant credit quality is critical.",
                suggested_action="Verify tenant financials, lease term remaining, and renewal probability.",
            ))

        return findings

    def _emotional_check(self, deal: Dict, outputs: Dict) -> List[CritiqueFinding]:
        """Detect emotional/cognitive biases in the analysis."""
        findings = []

        # Anchoring on asking price
        asking = deal.get("price", deal.get("asking_price"))
        appraised = _get_nested(outputs, "underwriting_analyst", "direct_cap_value")
        if asking and appraised and abs(asking - appraised) < asking * 0.03:
            findings.append(CritiqueFinding(
                "emotional", "Valuation suspiciously close to asking price — possible anchoring",
                "medium", BiasType.ANCHORING,
                counter_argument="Independent valuation should diverge from asking. If they match, you may be anchored on seller's number.",
                suggested_action="Re-run valuation without knowledge of asking price. Compare results.",
            ))

        # Confirmation bias — all signals positive
        signal_types = {s.get("type") for s in
                        (outputs.get("seal_signals", []) if isinstance(outputs.get("seal_signals"), list) else [])}
        opp_count = sum(1 for s in outputs.get("seal_signals", [])
                        if isinstance(s, dict) and s.get("type") == "opportunity")
        threat_count = sum(1 for s in outputs.get("seal_signals", [])
                           if isinstance(s, dict) and s.get("type") == "threat")
        if opp_count > 5 and threat_count == 0:
            findings.append(CritiqueFinding(
                "emotional", f"{opp_count} opportunities detected with zero threats — confirmation bias likely",
                "high", BiasType.CONFIRMATION,
                counter_argument="No deal is risk-free. Zero threats detected usually means threats are being overlooked.",
                suggested_action="Actively search for 3+ reasons this deal could fail.",
            ))

        # Overconfidence — very high confidence on uncertain variables
        text = json.dumps(outputs)
        if "GO" in text and "stress" not in text.lower():
            findings.append(CritiqueFinding(
                "emotional", "GO recommendation without stress testing — overconfidence risk",
                "high", BiasType.OVERCONFIDENCE,
                counter_argument="Stress tests are not optional. Rate+200bps and NOI-20% scenarios are minimum.",
                suggested_action="Run stress tests before finalizing recommendation.",
            ))

        return findings

    def _pre_mortem(self, deal: Dict, outputs: Dict,
                    signals: List[Signal] = None) -> List[CritiqueFinding]:
        """Pre-mortem: 'It's 3 years later and this deal failed. Why?'"""
        findings = []
        ptype = deal.get("property_type", "")

        # Universal failure modes
        scenarios = [
            ("Interest rates rise 300bps, refinance is underwater",
             "Rate shock destroys equity on variable-rate debt",
             "Lock in fixed-rate financing. Stress-test at current rate + 300bps."),

            ("Key tenant vacates at lease expiration",
             "Vacancy kills NOI and DSCR drops below 1.0x",
             "Verify lease term, renewal options, and tenant credit quality."),

            ("Environmental contamination discovered post-closing",
             "Phase I missed a UST leak. Remediation costs $200K+",
             "Never skip Phase II if Phase I shows any RECs. Budget for contingency."),
        ]

        # Property-type specific
        if ptype == "gas_station":
            scenarios.extend([
                ("EV adoption accelerates faster than projected",
                 "Fuel volume drops 30% in 5 years, revenue model breaks",
                 "Model EV transition scenarios. Value non-fuel revenue streams separately."),
                ("State bans gaming terminals",
                 "Gaming revenue (often 30-50% of profit) disappears overnight",
                 "Never underwrite with gaming as majority of NOI. Cap gaming at 30% of deal thesis."),
            ])

        if deal.get("gaming_eligible"):
            scenarios.extend([
                ("Gaming market saturates — new licenses flood the area",
                 "NTI drops 40% as 5 new gaming locations open within 2 miles",
                 "Check state licensing pipeline. Model NTI at 60% of current average."),
                ("Operator goes bankrupt, terminals removed",
                 "6 months of zero gaming revenue while new operator onboards",
                 "Verify operator financials. Have backup operator relationship."),
            ])

        for scenario, impact, mitigation in scenarios:
            findings.append(CritiqueFinding(
                "pre_mortem", f"FAILURE SCENARIO: {scenario}",
                "medium", None,
                counter_argument=impact,
                suggested_action=mitigation,
            ))

        return findings

    def _uncertainty_audit(self, beliefs: "BeliefState") -> List[CritiqueFinding]:
        """Flag decisions being made with high uncertainty."""
        findings = []

        critical_vars = ["noi", "cap_rate", "purchase_price", "dscr", "gaming_revenue"]
        for var_name in critical_vars:
            belief = beliefs.get(var_name)
            if belief and belief.confidence < 0.4:
                findings.append(CritiqueFinding(
                    "cognitive",
                    f"Making decision with low confidence on {var_name} "
                    f"({belief.confidence:.0%} confident, range: "
                    f"{belief.low:,.0f} to {belief.high:,.0f})",
                    "high" if var_name in ("noi", "purchase_price") else "medium",
                    BiasType.OVERCONFIDENCE,
                    counter_argument=f"The range on {var_name} is {belief.relative_spread:.0%} wide. "
                                     f"Decision could be very different at the extremes.",
                    suggested_action=f"Gather more data on {var_name} before deciding. "
                                     f"Current entropy: {belief.entropy:.1f}",
                ))

        return findings

    def to_prompt_injection(self, critique_result: Dict) -> str:
        """Convert critique results into a prompt fragment for the deciding agent."""
        lines = ["\n## CECA CRITIQUE — DEVIL'S ADVOCATE FINDINGS ##"]
        lines.append(f"Recommendation: {critique_result['proceed_recommendation']}")
        lines.append(f"Confidence adjustment: {critique_result['adjusted_confidence']:.0%}")

        if critique_result.get("bias_detections"):
            lines.append("\nDETECTED BIASES (correct for these):")
            for b in critique_result["bias_detections"]:
                lines.append(f"  ⚠ {b['bias']}: {b['finding']}")
                if b["counter_argument"]:
                    lines.append(f"    Counter: {b['counter_argument']}")

        if critique_result.get("pre_mortem_scenarios"):
            lines.append("\nPRE-MORTEM SCENARIOS (address each):")
            for i, s in enumerate(critique_result["pre_mortem_scenarios"][:5], 1):
                lines.append(f"  {i}. {s['finding']}")
                lines.append(f"     Mitigation: {s['suggested_action']}")

        lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATOR
# ═══════════════════════════════════════════════════════════════

class MonteCarloSimulator:
    """Monte Carlo deal outcome simulation with correlated draws.

    Uses Cholesky-decomposed correlation structure so that
    NOI, interest_rate, exit_cap, and noi_growth co-move
    realistically instead of being drawn independently.
    """

    def __init__(self, num_simulations: int = 2000, seed: int = None):
        self.num_simulations = num_simulations
        self._adaptive = None
        self._correlated_engine = None
        try:
            try:
                from .v2_fixes import AdaptiveMonteCarlo
            except ImportError:
                from v2_fixes import AdaptiveMonteCarlo
            self._adaptive = AdaptiveMonteCarlo(
                max_sims=num_simulations, batch_size=200,
                convergence_threshold=0.02, min_batches=2,
            )
        except ImportError:
            pass
        # Initialize correlated draw engine
        try:
            try:
                from .correlated_mc import CorrelatedDrawEngine
            except ImportError:
                from correlated_mc import CorrelatedDrawEngine
            self._correlated_engine = CorrelatedDrawEngine()
        except ImportError:
            self._correlated_engine = None
        if seed is not None:
            get_rng()  # seed handled by determinism module

    def simulate_deal(self, params: Dict) -> Dict:
        """Run full Monte Carlo simulation on a deal.

        Uses correlated Cholesky draws when available, falls back
        to independent triangular draws otherwise.
        """
        price = params.get("purchase_price", 2000000)
        ltv = params.get("loan_ltv", 0.75)
        hold = params.get("hold_years", 5)
        loan_amount = price * ltv
        equity = price - loan_amount
        loan_term = params.get("loan_term", 25)

        # Build draw params for correlated engine
        draw_params = {
            "noi": params.get("noi", {}),
            "loan_rate": params.get("loan_rate", {"point": 7.0, "low": 5.5, "high": 9.0}),
            "noi_growth": params.get("noi_growth", {"point": 2.0, "low": -2.0, "high": 5.0}),
            "exit_cap": params.get("exit_cap", {"point": 7.5, "low": 6.0, "high": 10.0}),
        }
        use_correlated = self._correlated_engine is not None

        # Results storage
        irrs = []
        equity_multiples = []
        dscrs_yr1 = []
        coc_yr1s = []
        npvs = []
        exit_values = []
        total_profits = []
        rng = get_rng()

        for _ in range(self.num_simulations):
            # ── CORRELATED DRAWS (Cholesky) ──
            if use_correlated:
                draws = self._correlated_engine.draw(draw_params, rng)
                noi = draws["noi"]
                rate = draws["loan_rate"] / 100
                noi_growth = draws["noi_growth"] / 100
                exit_cap = draws["exit_cap"] / 100
            else:
                # Fallback: independent triangular draws
                noi = _triangular(draw_params["noi"])
                rate = _triangular(draw_params["loan_rate"]) / 100
                noi_growth = _triangular(draw_params["noi_growth"]) / 100
                exit_cap = _triangular(draw_params["exit_cap"]) / 100

            # Gaming overlay
            gaming_nti = 0
            if "gaming_nti" in params:
                gaming_nti = _triangular(params["gaming_nti"]) * 365
                terminal_count = params.get("terminal_count", 5)
                gaming_revenue = gaming_nti * terminal_count
                # Net to location (typically 35% after taxes and operator split)
                gaming_net = gaming_revenue * params.get("gaming_net_pct", 0.35)
                noi += gaming_net

            # Annual debt service
            if rate > 0 and loan_amount > 0:
                monthly_rate = rate / 12
                months = loan_term * 12
                monthly_pmt = loan_amount * (monthly_rate * (1 + monthly_rate) ** months) / \
                              ((1 + monthly_rate) ** months - 1)
                annual_ds = monthly_pmt * 12
            else:
                annual_ds = 0

            # Year 1 metrics
            dscr_1 = noi / max(annual_ds, 1)
            coc_1 = (noi - annual_ds) / max(equity, 1)

            dscrs_yr1.append(dscr_1)
            coc_yr1s.append(coc_1 * 100)

            # Cash flows over hold period
            cashflows = [-equity]  # initial equity investment
            for yr in range(1, hold + 1):
                yr_noi = noi * (1 + noi_growth) ** (yr - 1)
                yr_cf = yr_noi - annual_ds
                cashflows.append(yr_cf)

            # Exit
            exit_noi = noi * (1 + noi_growth) ** hold
            exit_value = exit_noi / max(exit_cap, 0.01)
            # Remaining loan balance — proper amortization formula:
            #   B(n) = P * [(1+r)^N - (1+r)^n] / [(1+r)^N - 1]
            if rate > 0 and loan_amount > 0:
                monthly_rate = rate / 12
                total_months = loan_term * 12
                elapsed_months = hold * 12
                factor_total = (1 + monthly_rate) ** total_months
                factor_elapsed = (1 + monthly_rate) ** elapsed_months
                loan_bal = loan_amount * (factor_total - factor_elapsed) / (factor_total - 1)
            else:
                loan_bal = loan_amount * (1 - hold / max(loan_term, 1))
            net_proceeds = exit_value - loan_bal
            cashflows[-1] += net_proceeds

            exit_values.append(exit_value)
            total_profit = sum(cashflows)
            total_profits.append(total_profit)

            # Equity multiple
            total_distributions = sum(cf for cf in cashflows[1:])
            eq_mult = total_distributions / max(equity, 1)
            equity_multiples.append(eq_mult)

            # IRR (Newton's method approximation)
            irr = _compute_irr(cashflows)
            irrs.append(irr * 100 if irr else 0)

            # NPV at 10% discount
            npv = _compute_npv(cashflows, 0.10)
            npvs.append(npv)

        return {
            "simulations": self.num_simulations,
            "correlated_draws": use_correlated,
            "irr": _distribution_stats(irrs, "IRR %"),
            "equity_multiple": _distribution_stats(equity_multiples, "Equity Multiple"),
            "dscr_year1": _distribution_stats(dscrs_yr1, "DSCR Year 1"),
            "cash_on_cash_year1": _distribution_stats(coc_yr1s, "CoC Year 1 %"),
            "exit_value": _distribution_stats(exit_values, "Exit Value $"),
            "npv_at_10pct": _distribution_stats(npvs, "NPV @ 10%"),
            "total_profit": _distribution_stats(total_profits, "Total Profit $"),
            "probability_analysis": {
                "prob_positive_irr": round(sum(1 for x in irrs if x > 0) / len(irrs), 3),
                "prob_irr_above_10": round(sum(1 for x in irrs if x > 10) / len(irrs), 3),
                "prob_irr_above_15": round(sum(1 for x in irrs if x > 15) / len(irrs), 3),
                "prob_irr_above_20": round(sum(1 for x in irrs if x > 20) / len(irrs), 3),
                "prob_dscr_below_1": round(sum(1 for x in dscrs_yr1 if x < 1.0) / len(dscrs_yr1), 3),
                "prob_dscr_below_1_15": round(sum(1 for x in dscrs_yr1 if x < 1.15) / len(dscrs_yr1), 3),
                "prob_loss": round(sum(1 for x in total_profits if x < 0) / len(total_profits), 3),
                "value_at_risk_5pct": round(sorted(total_profits)[int(len(total_profits) * 0.05)]),
            },
        }


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _get_nested(data: Dict, *keys) -> Any:
    """Safely get a nested value."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def _triangular(dist: Dict) -> float:
    """Sample from triangular distribution {point, low, high}."""
    if isinstance(dist, (int, float)):
        return float(dist)
    point = dist.get("point", 0)
    low = dist.get("low", point * 0.7)
    high = dist.get("high", point * 1.3)
    try:
        return get_rng().triangular(low, high, point)
    except ValueError:
        return point


def _distribution_stats(values: List[float], label: str = "") -> Dict:
    """Compute distribution statistics."""
    if not values:
        return {"label": label, "mean": 0, "median": 0}
    n = len(values)
    s = sorted(values)
    mean = sum(values) / n
    median = s[n // 2]
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
    return {
        "label": label,
        "mean": round(mean, 2), "median": round(median, 2),
        "std": round(std, 2),
        "p5": round(s[int(n * 0.05)], 2),
        "p25": round(s[int(n * 0.25)], 2),
        "p75": round(s[int(n * 0.75)], 2),
        "p95": round(s[int(n * 0.95)], 2),
        "min": round(s[0], 2), "max": round(s[-1], 2),
    }


def _compute_irr(cashflows: List[float], max_iter: int = 100) -> Optional[float]:
    """Newton-Raphson IRR calculation."""
    if not cashflows or all(cf == 0 for cf in cashflows):
        return None
    rate = 0.10  # initial guess
    for _ in range(max_iter):
        npv = sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))
        dnpv = sum(-t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cashflows))
        if abs(dnpv) < 1e-10:
            break
        new_rate = rate - npv / dnpv
        if abs(new_rate - rate) < 1e-7:
            return new_rate
        rate = new_rate
        if abs(rate) > 10:
            return None
    return rate


def _compute_npv(cashflows: List[float], discount_rate: float) -> float:
    """Compute Net Present Value."""
    return sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(cashflows))
