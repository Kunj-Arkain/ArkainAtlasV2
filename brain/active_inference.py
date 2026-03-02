"""
engine.brain.active_inference — Active Inference Engine
=========================================================
Agents don't just run tools — they maintain BELIEFS about the deal
and actively seek information that reduces UNCERTAINTY the most.

Core Loop (per agent):
  1. Initialize prior beliefs (from ledger + deal data)
  2. Calculate FREE ENERGY (how surprised are we by observations?)
  3. Pick next action that MINIMIZES EXPECTED FREE ENERGY
  4. Update beliefs with observation (Bayesian update)
  5. Repeat until free energy is below threshold or budget exhausted

This replaces blind sequential tool execution with intelligent
information foraging — the agent calls the tools that will
teach it the most, not just the tools in order.

Theory:
  Free Energy F = D_KL(q(s) || p(s)) - E_q[ln p(o|s)]
  Simplified: F = Uncertainty + Surprise
  Goal: minimize F by either updating beliefs (perception)
        or taking actions that produce informative observations

Implementation:
  BeliefState — probability distributions over deal variables
  FreeEnergyCalc — measures current uncertainty + prediction error
  EpistemicForager — ranks next actions by expected info gain
  ActiveInferenceLoop — full perception-action cycle
"""

from __future__ import annotations

import math
import json
import logging
import random
try:
    from .determinism import get_rng
except ImportError:
    from determinism import get_rng
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# BELIEF STATE
# ═══════════════════════════════════════════════════════════════

@dataclass
class Belief:
    """A single belief about a deal variable.

    Represents uncertainty as a distribution:
      - point: best estimate
      - low/high: 90% confidence interval
      - confidence: 0.0 (no idea) to 1.0 (certain)
      - source: where this belief came from
      - entropy: information-theoretic uncertainty (bits)
    """
    variable: str           # e.g. "noi", "cap_rate", "traffic_count"
    point: float = 0.0      # best estimate
    low: float = 0.0        # 5th percentile
    high: float = 0.0       # 95th percentile
    confidence: float = 0.0 # 0=unknown, 1=verified
    source: str = "prior"   # "prior", "tool:census_demographics", "agent:market_analyst"
    observation_count: int = 0
    last_updated: float = 0.0

    @property
    def spread(self) -> float:
        """Width of confidence interval (absolute)."""
        return self.high - self.low

    @property
    def relative_spread(self) -> float:
        """Spread as percentage of point estimate."""
        if self.point == 0:
            return float('inf')
        return self.spread / abs(self.point)

    @property
    def entropy(self) -> float:
        """Shannon entropy estimate (higher = more uncertain).

        Uses log-normal approximation for financial variables.
        H ≈ ln(σ√(2πe)) where σ = spread/3.29 (90% CI → σ)
        """
        if self.spread <= 0:
            return 0.0
        sigma = self.spread / 3.29  # 90% CI → 1 std dev
        if sigma <= 0:
            return 0.0
        return math.log(sigma * math.sqrt(2 * math.pi * math.e))

    def update(self, observed_value: float, observation_confidence: float,
               source: str = "observation"):
        """Proper conjugate Bayesian belief update (v2).

        Uses BayesianUpdater from v2_fixes for distribution-appropriate
        updates (lognormal for financials, beta for rates, normal for general).
        Falls back to weighted average if import fails.
        """
        import time
        try:
            try:
                from .v2_fixes import BayesianUpdater
            except ImportError:
                from v2_fixes import BayesianUpdater
            updater = BayesianUpdater()
            var_type = updater.get_var_type(self.variable)
            prior = {"point": self.point, "low": self.low, "high": self.high,
                     "confidence": self.confidence}
            obs_precision = observation_confidence * 3.0  # scale confidence to precision
            posterior = updater.update(prior, observed_value, obs_precision, var_type)
            self.point = posterior["point"]
            self.low = posterior["low"]
            self.high = posterior["high"]
            self.confidence = posterior["confidence"]
        except (ImportError, Exception):
            # Fallback: weighted average (v1 behavior)
            prior_weight = self.confidence
            obs_weight = observation_confidence
            total_weight = prior_weight + obs_weight
            if total_weight == 0:
                total_weight = 1.0
            self.point = (self.point * prior_weight + observed_value * obs_weight) / total_weight
            shrink = obs_weight / total_weight
            new_spread = self.spread * (1 - shrink * 0.5)
            self.low = self.point - new_spread / 2
            self.high = self.point + new_spread / 2
            self.confidence = min(1.0, total_weight / (total_weight + 0.5))

        self.source = source
        self.observation_count += 1
        self.last_updated = time.time()

    def to_dict(self) -> Dict:
        return {
            "variable": self.variable, "point": round(self.point, 2),
            "low": round(self.low, 2), "high": round(self.high, 2),
            "confidence": round(self.confidence, 3),
            "entropy": round(self.entropy, 3),
            "source": self.source, "observations": self.observation_count,
        }


class BeliefState:
    """Full belief state over all deal variables.

    Think of this as the agent's mental model of the deal.
    High entropy variables = "I don't know much about this"
    Low entropy variables = "I'm fairly certain about this"
    """

    # Default priors for CRE deal variables
    DEFAULT_PRIORS = {
        # Financial
        "noi":              {"point": 150000, "low": 50000,   "high": 400000, "confidence": 0.1},
        "cap_rate":         {"point": 7.0,    "low": 4.0,     "high": 12.0,   "confidence": 0.15},
        "interest_rate":    {"point": 7.5,    "low": 5.0,     "high": 11.0,   "confidence": 0.10},
        "exit_cap_rate":    {"point": 7.5,    "low": 5.0,     "high": 12.5,   "confidence": 0.08},
        "purchase_price":   {"point": 2000000,"low": 500000,  "high": 5000000,"confidence": 0.1},
        "dscr":             {"point": 1.25,   "low": 0.8,     "high": 2.5,    "confidence": 0.1},
        "irr":              {"point": 12.0,   "low": 5.0,     "high": 25.0,   "confidence": 0.05},
        "cash_on_cash":     {"point": 8.0,    "low": 3.0,     "high": 20.0,   "confidence": 0.05},
        # Market
        "population":       {"point": 50000,  "low": 5000,    "high": 500000, "confidence": 0.05},
        "median_income":    {"point": 55000,  "low": 25000,   "high": 120000, "confidence": 0.05},
        "unemployment":     {"point": 5.0,    "low": 2.0,     "high": 15.0,   "confidence": 0.05},
        "traffic_count":    {"point": 15000,  "low": 2000,    "high": 80000,  "confidence": 0.05},
        "competitor_count": {"point": 5,      "low": 0,       "high": 20,     "confidence": 0.05},
        # Gaming
        "nti_per_terminal": {"point": 250,    "low": 80,      "high": 600,    "confidence": 0.05},
        "terminal_count":   {"point": 5,      "low": 0,       "high": 10,     "confidence": 0.1},
        "gaming_revenue":   {"point": 120000, "low": 0,       "high": 500000, "confidence": 0.05},
        # Property
        "sqft":             {"point": 3000,   "low": 800,     "high": 15000,  "confidence": 0.1},
        "year_built":       {"point": 1990,   "low": 1950,    "high": 2020,   "confidence": 0.1},
        "lot_size_acres":   {"point": 1.0,    "low": 0.25,    "high": 5.0,    "confidence": 0.05},
        # Risk
        "environmental_risk": {"point": 0.3,  "low": 0.0,     "high": 1.0,    "confidence": 0.05},
        "crime_rate":       {"point": 5.0,    "low": 0.5,     "high": 20.0,   "confidence": 0.05},
        "flood_risk":       {"point": 0.2,    "low": 0.0,     "high": 1.0,    "confidence": 0.05},
    }

    def __init__(self, initial_data: Dict = None):
        self.beliefs: Dict[str, Belief] = {}
        self._init_priors()
        if initial_data:
            self._incorporate_known_data(initial_data)

    def _init_priors(self):
        """Initialize with vague prior beliefs."""
        for var, params in self.DEFAULT_PRIORS.items():
            self.beliefs[var] = Belief(
                variable=var,
                point=params["point"], low=params["low"], high=params["high"],
                confidence=params["confidence"], source="prior",
            )

    def _incorporate_known_data(self, data: Dict):
        """Incorporate known deal data as high-confidence observations."""
        mapping = {
            "price": "purchase_price", "asking_price": "purchase_price",
            "sqft": "sqft", "square_feet": "sqft",
            "year_built": "year_built", "lot_size": "lot_size_acres",
            "terminals": "terminal_count", "terminal_count": "terminal_count",
            "address": None, "state": None, "property_type": None,
        }
        for key, value in data.items():
            if isinstance(value, (int, float)):
                var_name = mapping.get(key, key)
                if var_name and var_name in self.beliefs:
                    self.beliefs[var_name].update(float(value), 0.9, f"deal_data:{key}")
                elif var_name:
                    # Create new belief for unknown variables
                    spread = abs(value) * 0.3  # 30% uncertainty
                    self.beliefs[var_name] = Belief(
                        variable=var_name, point=float(value),
                        low=value - spread, high=value + spread,
                        confidence=0.9, source=f"deal_data:{key}",
                    )

    def get(self, variable: str) -> Optional[Belief]:
        return self.beliefs.get(variable)

    def set_observation(self, variable: str, value: float,
                        confidence: float, source: str):
        """Update belief with new observation + propagate correlations (v2)."""
        old_point = self.beliefs[variable].point if variable in self.beliefs else value

        if variable in self.beliefs:
            self.beliefs[variable].update(value, confidence, source)
        else:
            spread = abs(value) * 0.2
            self.beliefs[variable] = Belief(
                variable=variable, point=value,
                low=value - spread, high=value + spread,
                confidence=confidence, source=source,
            )

        # v2: Propagate to correlated variables
        try:
            try:
                from .v2_fixes import CorrelatedBeliefEngine
            except ImportError:
                from v2_fixes import CorrelatedBeliefEngine
            cbe = CorrelatedBeliefEngine()
            new_point = self.beliefs[variable].point
            propagations = cbe.propagate_update(variable, old_point, new_point, self.beliefs)
            for neighbor_var, info in propagations.items():
                if neighbor_var in self.beliefs:
                    b = self.beliefs[neighbor_var]
                    shift = info["new_point"] - info["old_point"]
                    b.point += shift
                    b.low += shift
                    b.high += shift
        except (ImportError, Exception):
            pass  # v1 fallback: no correlation propagation

    def total_entropy(self) -> float:
        """Total uncertainty across all beliefs."""
        return sum(b.entropy for b in self.beliefs.values())

    def max_entropy_variables(self, n: int = 5) -> List[Belief]:
        """Get the N most uncertain variables."""
        return sorted(self.beliefs.values(), key=lambda b: b.entropy, reverse=True)[:n]

    def low_confidence_variables(self, threshold: float = 0.3) -> List[Belief]:
        """Get variables below confidence threshold."""
        return [b for b in self.beliefs.values() if b.confidence < threshold]

    def snapshot(self) -> Dict:
        """Full belief state snapshot."""
        return {
            "total_entropy": round(self.total_entropy(), 2),
            "belief_count": len(self.beliefs),
            "beliefs": {k: v.to_dict() for k, v in self.beliefs.items()},
            "most_uncertain": [b.to_dict() for b in self.max_entropy_variables(5)],
        }

    def to_prompt_fragment(self) -> str:
        """Generate a prompt injection showing current beliefs + uncertainties."""
        lines = ["\n## CURRENT BELIEF STATE (what you know vs don't know) ##"]
        lines.append("HIGH CONFIDENCE (use these values):")
        for b in sorted(self.beliefs.values(), key=lambda x: -x.confidence):
            if b.confidence >= 0.6:
                lines.append(f"  {b.variable} = {b.point:,.2f} [confidence: {b.confidence:.0%}]")
        lines.append("\nLOW CONFIDENCE (investigate these — highest value info):")
        for b in self.max_entropy_variables(8):
            if b.confidence < 0.6:
                lines.append(
                    f"  {b.variable}: estimate {b.point:,.2f} "
                    f"but range [{b.low:,.2f} - {b.high:,.2f}] "
                    f"[confidence: {b.confidence:.0%}, entropy: {b.entropy:.1f}]"
                )
        lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# FREE ENERGY CALCULATOR
# ═══════════════════════════════════════════════════════════════

class FreeEnergyCalc:
    """Measures how 'surprised' the agent is by current state.

    Free Energy F = Complexity + Inaccuracy
      Complexity = how far beliefs are from priors (D_KL)
      Inaccuracy = how poorly beliefs predict observations (-log likelihood)

    In practice for CRE deals:
      High F = "This deal has unexpected features, I need more info"
      Low F  = "This deal matches my model, I understand it well"
    """

    @staticmethod
    def compute(beliefs: BeliefState, observations: Dict[str, float] = None) -> Dict:
        """Compute free energy decomposition.

        Returns:
          total_free_energy: overall surprise/uncertainty
          complexity: divergence from priors
          inaccuracy: prediction errors from observations
          variable_contributions: per-variable breakdown
        """
        observations = observations or {}
        total_entropy = 0.0
        total_surprise = 0.0
        variable_contribs = {}

        for var, belief in beliefs.beliefs.items():
            # Complexity: entropy of current belief (uncertainty)
            entropy = belief.entropy
            total_entropy += max(0, entropy)

            # Inaccuracy: if we have an observation, how far off were we?
            surprise = 0.0
            if var in observations:
                obs = observations[var]
                if belief.point != 0:
                    prediction_error = abs(obs - belief.point) / abs(belief.point)
                    surprise = prediction_error * (1 - belief.confidence)
                total_surprise += surprise

            variable_contribs[var] = {
                "entropy": round(entropy, 3),
                "surprise": round(surprise, 3),
                "free_energy": round(entropy + surprise, 3),
            }

        total_fe = total_entropy + total_surprise

        return {
            "total_free_energy": round(total_fe, 2),
            "complexity": round(total_entropy, 2),
            "inaccuracy": round(total_surprise, 2),
            "variable_contributions": variable_contribs,
            "convergence_ratio": round(
                1.0 - (total_fe / max(total_fe + 1, 1)), 3
            ),
        }

    @staticmethod
    def is_converged(beliefs: BeliefState, threshold: float = 15.0) -> bool:
        """Check if free energy is below convergence threshold."""
        fe = FreeEnergyCalc.compute(beliefs)
        return fe["total_free_energy"] < threshold


# ═══════════════════════════════════════════════════════════════
# EPISTEMIC FORAGER
# ═══════════════════════════════════════════════════════════════

class EpistemicForager:
    """Decides which tool to call next based on expected information gain.

    Instead of calling tools in order, the forager calculates:
      "If I call census_demographics, how much entropy will it reduce?"
      "If I call traffic_counts, how much will I learn?"

    Then picks the action with highest expected info gain per cost.

    This is the ACTIVE part of Active Inference — the agent
    actively seeks out the most informative observations.
    """

    # Maps tools to the belief variables they inform
    TOOL_INFORMATION_MAP = {
        "census_demographics":   {"population": 0.8, "median_income": 0.8, "unemployment": 0.5},
        "bls_employment":        {"unemployment": 0.9, "population": 0.3},
        "fred_economic_data":    {"cap_rate": 0.3, "median_income": 0.4, "interest_rate": 0.6},
        "traffic_counts":        {"traffic_count": 0.9},
        "competitor_scan":       {"competitor_count": 0.9},
        "property_records":      {"sqft": 0.9, "year_built": 0.9, "lot_size_acres": 0.8, "purchase_price": 0.5},
        "pull_comps":            {"cap_rate": 0.7, "purchase_price": 0.5, "noi": 0.4, "exit_cap_rate": 0.5},
        "market_cap_rates":      {"cap_rate": 0.8},
        "environmental_risk":    {"environmental_risk": 0.9, "flood_risk": 0.5},
        "crime_data":            {"crime_rate": 0.9},
        "insurance_estimate":    {"noi": 0.2},
        "utility_costs":         {"noi": 0.2},
        "gaming_board_data":     {"nti_per_terminal": 0.7, "terminal_count": 0.5, "gaming_revenue": 0.5},
        "egm_predict":           {"nti_per_terminal": 0.8, "gaming_revenue": 0.7},
        "egm_market_health":     {"gaming_revenue": 0.5, "competitor_count": 0.3},
        "generate_term_sheets":  {"dscr": 0.6, "irr": 0.3, "interest_rate": 0.8},
        "evaluate_deal":         {"noi": 0.5, "cap_rate": 0.5, "irr": 0.5, "cash_on_cash": 0.5, "exit_cap_rate": 0.4},
        "state_context":         {"terminal_count": 0.3, "gaming_revenue": 0.2},
        "zoning_lookup":         {"environmental_risk": 0.3},
        "location_score":        {"traffic_count": 0.3, "crime_rate": 0.2},
        "code_analysis":         {"sqft": 0.2},
        "electrical_load_calc":  {},
        "hvac_sizing":           {},
        "structural_calc":       {},
    }

    # Cost per tool call (relative — affects priority ordering)
    TOOL_COSTS = {
        "census_demographics": 1.0, "bls_employment": 1.0,
        "traffic_counts": 1.0, "competitor_scan": 1.0,
        "property_records": 1.0, "pull_comps": 1.5,
        "market_cap_rates": 1.0, "environmental_risk": 1.0,
        "crime_data": 1.0, "gaming_board_data": 1.0,
        "egm_predict": 1.5, "generate_term_sheets": 2.0,
        "evaluate_deal": 2.0, "state_context": 0.5,
    }

    def __init__(self):
        """Initialize with v2 caches for performance."""
        self._eig_cache = None
        self._tool_cache = None
        try:
            try:
                from .v2_fixes import SurrogateEIG, ToolCallCache
            except ImportError:
                from v2_fixes import SurrogateEIG, ToolCallCache
            self._eig_cache = SurrogateEIG()
            self._tool_cache = ToolCallCache()
        except ImportError:
            pass

    def rank_actions(self, beliefs: BeliefState,
                     available_tools: List[str],
                     already_called: Set[str] = None) -> List[Dict]:
        """Rank tools by expected information gain / cost.

        Returns ordered list of {tool, expected_info_gain, cost, efficiency,
        variables_informed, rationale}.
        """
        already_called = already_called or set()
        rankings = []

        for tool in available_tools:
            if tool in already_called:
                continue

            info_map = self.TOOL_INFORMATION_MAP.get(tool, {})
            if not info_map:
                continue

            # Calculate expected entropy reduction
            total_info_gain = 0.0
            variables_helped = []

            for var, reduction_strength in info_map.items():
                belief = beliefs.get(var)
                if belief is None:
                    continue

                # Info gain = current_entropy × reduction_strength × (1 - confidence)
                # Higher entropy + lower confidence = more to gain
                current_entropy = max(0, belief.entropy)
                info_gain = current_entropy * reduction_strength * (1 - belief.confidence)
                total_info_gain += info_gain

                if info_gain > 0.1:
                    variables_helped.append({
                        "variable": var,
                        "current_confidence": round(belief.confidence, 2),
                        "current_entropy": round(current_entropy, 2),
                        "expected_reduction": round(info_gain, 2),
                    })

            cost = self.TOOL_COSTS.get(tool, 1.0)
            efficiency = total_info_gain / max(cost, 0.1)

            if total_info_gain > 0.01:
                rankings.append({
                    "tool": tool,
                    "expected_info_gain": round(total_info_gain, 3),
                    "cost": cost,
                    "efficiency": round(efficiency, 3),
                    "variables_informed": variables_helped,
                    "rationale": self._generate_rationale(tool, variables_helped),
                })

        # Sort by efficiency (info gain per unit cost)
        rankings.sort(key=lambda x: x["efficiency"], reverse=True)
        return rankings

    def suggest_next_action(self, beliefs: BeliefState,
                            available_tools: List[str],
                            already_called: Set[str] = None) -> Optional[Dict]:
        """Get the single best next tool to call."""
        ranked = self.rank_actions(beliefs, available_tools, already_called)
        return ranked[0] if ranked else None

    def _generate_rationale(self, tool: str, variables: List[Dict]) -> str:
        if not variables:
            return f"Call {tool} for supplementary data"
        top_var = max(variables, key=lambda v: v["expected_reduction"])
        return (
            f"Call {tool} — highest info gain on {top_var['variable']} "
            f"(currently {top_var['current_confidence']:.0%} confident, "
            f"entropy={top_var['current_entropy']:.1f})"
        )

    def generate_foraging_plan(self, beliefs: BeliefState,
                               available_tools: List[str],
                               budget: int = 10) -> List[Dict]:
        """Generate optimal tool-call sequence using expected posterior sampling.

        For each candidate tool, draws N samples from the current belief's
        uncertainty range, computes the posterior entropy under each sample
        (as if that sample were the observation), and uses the average
        entropy reduction as the expected information gain. This replaces
        the v1 heuristic of directly bumping confidence.
        """
        N_SAMPLES = 20  # lightweight expected-posterior approximation
        plan = []
        called = set()
        # Deep copy beliefs for simulation
        sim_beliefs = BeliefState()
        for var, belief in beliefs.beliefs.items():
            sim_beliefs.beliefs[var] = Belief(
                variable=var, point=belief.point, low=belief.low,
                high=belief.high, confidence=belief.confidence,
                source=belief.source, observation_count=belief.observation_count,
            )

        for step in range(budget):
            ranked = self.rank_actions(sim_beliefs, available_tools, called)
            if not ranked:
                break

            best = ranked[0]
            if best["expected_info_gain"] < 0.05:
                break  # diminishing returns

            plan.append({
                "step": step + 1,
                "tool": best["tool"],
                "expected_info_gain": best["expected_info_gain"],
                "rationale": best["rationale"],
                "cumulative_calls": step + 1,
            })
            called.add(best["tool"])

            # Expected-posterior belief update: sample N hypothetical
            # observations from each target variable's current uncertainty
            # range, compute what the posterior would be, and average.
            info_map = self.TOOL_INFORMATION_MAP.get(best["tool"], {})
            for var, strength in info_map.items():
                belief = sim_beliefs.get(var)
                if not belief:
                    continue
                pre_entropy = belief.entropy
                # Draw N samples uniformly from current [low, high]
                samples = [belief.low + get_rng().random() * (belief.high - belief.low)
                           for _ in range(N_SAMPLES)]
                post_entropies = []
                for obs in samples:
                    # Simulate a Bayesian update at this observation
                    obs_conf = strength * 0.6  # tool strength → observation confidence
                    # Posterior via weighted combination (conjugate approximation)
                    total_w = belief.confidence + obs_conf
                    if total_w == 0:
                        total_w = 0.01
                    new_point = (belief.point * belief.confidence + obs * obs_conf) / total_w
                    shrink = obs_conf / total_w
                    new_spread = belief.spread * (1 - shrink * 0.5)
                    new_spread = max(new_spread, belief.spread * 0.05)  # floor
                    new_conf = min(1.0, total_w / (total_w + 0.5))
                    # Entropy of the hypothetical posterior
                    if new_spread > 0:
                        post_entropy = math.log(new_spread + 1) * (1 - new_conf)
                    else:
                        post_entropy = 0
                    post_entropies.append(post_entropy)
                # Expected posterior entropy = mean across samples
                expected_post_entropy = sum(post_entropies) / len(post_entropies)
                # Apply the expected posterior to sim_beliefs for next ranking
                expected_reduction = pre_entropy - expected_post_entropy
                if expected_reduction > 0:
                    ratio = expected_reduction / max(pre_entropy, 0.01)
                    new_spread = belief.spread * (1 - ratio * 0.5)
                    new_spread = max(new_spread, belief.spread * 0.05)
                    new_conf = min(1.0, belief.confidence + ratio * 0.5)
                    belief.confidence = new_conf
                    belief.low = belief.point - new_spread / 2
                    belief.high = belief.point + new_spread / 2

        return plan
