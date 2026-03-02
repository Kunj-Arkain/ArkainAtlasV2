"""
engine.cost_aware_forager — EIG/$ Epistemic Forager (Upgrade)
===============================================================
Upgrades tool ranking from EIG/relative_cost to:

  Priority = Expected Information Gain / Total Cost

Where Total Cost = API_cost + compute_cost + latency_cost + opportunity_cost

This is the missing alpha: economic optimality of information acquisition.

Tracks:
  - Per-tool: API $ cost, avg latency, compute time, call count
  - Running budget: total $ spent, total time spent
  - ROI: actual information gained vs predicted, per tool
  - Diminishing returns: penalty for re-querying same variable

The forager now answers: "Is spending $0.50 and 2 seconds on this
API call worth the expected 0.3 bits of information?"
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .active_inference import BeliefState, EpistemicForager
except ImportError:
    from active_inference import BeliefState, EpistemicForager


# ═══════════════════════════════════════════════════════════════
# COST MODEL
# ═══════════════════════════════════════════════════════════════

@dataclass
class ToolCostProfile:
    """Real cost profile for a single tool."""
    name: str
    api_cost_usd: float = 0.0       # Per-call API cost
    avg_latency_ms: int = 500        # Average response time
    compute_cost_usd: float = 0.0    # Internal compute cost
    reliability: float = 0.95        # P(success) — affects expected cost

    # Tracked at runtime
    total_calls: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: int = 0
    total_info_gained: float = 0.0
    failures: int = 0

    @property
    def effective_cost(self) -> float:
        """Expected cost per successful call."""
        if self.reliability <= 0:
            return float('inf')
        return (self.api_cost_usd + self.compute_cost_usd) / self.reliability

    @property
    def actual_roi(self) -> float:
        """Actual info gained per dollar spent."""
        if self.total_cost_usd <= 0:
            return 0.0
        return self.total_info_gained / self.total_cost_usd

    def record_call(self, cost_usd: float, latency_ms: int,
                    info_gained: float, success: bool):
        """Record a completed tool call."""
        self.total_calls += 1
        self.total_cost_usd += cost_usd
        self.total_latency_ms += latency_ms
        self.total_info_gained += info_gained
        if not success:
            self.failures += 1


# Default cost profiles for all tools
DEFAULT_COSTS: Dict[str, ToolCostProfile] = {
    # Free / government APIs
    "census_demographics":   ToolCostProfile("census_demographics", api_cost_usd=0.00, avg_latency_ms=800, reliability=0.90),
    "census_business_patterns": ToolCostProfile("census_business_patterns", api_cost_usd=0.00, avg_latency_ms=800, reliability=0.90),
    "bls_employment":        ToolCostProfile("bls_employment", api_cost_usd=0.00, avg_latency_ms=600, reliability=0.92),
    "fred_economic_data":    ToolCostProfile("fred_economic_data", api_cost_usd=0.00, avg_latency_ms=400, reliability=0.95),
    "crime_data":            ToolCostProfile("crime_data", api_cost_usd=0.00, avg_latency_ms=700, reliability=0.88),
    "environmental_risk":    ToolCostProfile("environmental_risk", api_cost_usd=0.00, avg_latency_ms=1000, reliability=0.85),
    "zoning_lookup":         ToolCostProfile("zoning_lookup", api_cost_usd=0.00, avg_latency_ms=600, reliability=0.80),

    # Paid data APIs
    "traffic_counts":        ToolCostProfile("traffic_counts", api_cost_usd=0.10, avg_latency_ms=1200, reliability=0.85),
    "property_records":      ToolCostProfile("property_records", api_cost_usd=0.25, avg_latency_ms=500, reliability=0.95),
    "pull_comps":            ToolCostProfile("pull_comps", api_cost_usd=0.50, avg_latency_ms=2000, reliability=0.80),
    "market_cap_rates":      ToolCostProfile("market_cap_rates", api_cost_usd=0.50, avg_latency_ms=1500, reliability=0.82),
    "competitor_scan":       ToolCostProfile("competitor_scan", api_cost_usd=0.15, avg_latency_ms=1800, reliability=0.75),
    "location_scores":       ToolCostProfile("location_scores", api_cost_usd=0.05, avg_latency_ms=800, reliability=0.90),

    # Gaming-specific (higher value)
    "gaming_board_data":     ToolCostProfile("gaming_board_data", api_cost_usd=0.00, avg_latency_ms=1000, reliability=0.90),

    # Computation tools (no API cost, just compute)
    "insurance_estimate":    ToolCostProfile("insurance_estimate", api_cost_usd=0.00, avg_latency_ms=100, compute_cost_usd=0.01),
    "utility_costs":         ToolCostProfile("utility_costs", api_cost_usd=0.00, avg_latency_ms=100, compute_cost_usd=0.01),
    "generate_term_sheets":  ToolCostProfile("generate_term_sheets", api_cost_usd=0.00, avg_latency_ms=200, compute_cost_usd=0.02),
    "evaluate_deal":         ToolCostProfile("evaluate_deal", api_cost_usd=0.00, avg_latency_ms=300, compute_cost_usd=0.03),
}


# ═══════════════════════════════════════════════════════════════
# COST-AWARE FORAGER
# ═══════════════════════════════════════════════════════════════

@dataclass
class ForagingBudget:
    """Economic budget for information acquisition."""
    max_cost_usd: float = 5.00       # Max $ per deal analysis
    max_latency_ms: int = 30000      # 30 second wall clock budget
    max_calls: int = 50              # Hard cap on tool calls
    min_eig_per_dollar: float = 0.5  # Minimum info gain per dollar
    diminishing_returns_penalty: float = 0.5  # Penalty multiplier per repeat query


class CostAwareForager(EpistemicForager):
    """Epistemic forager that optimizes Expected Information Gain per Dollar.

    Upgrades the base EpistemicForager with:
      1. Real $ costs per tool (API fees, compute)
      2. Latency budgeting (wall-clock time allocation)
      3. Diminishing returns detection (penalize re-querying same variable)
      4. ROI tracking (actual vs predicted info gain per tool)
      5. Economic stopping criterion (stop when EIG/$ < threshold)

    Usage:
        forager = CostAwareForager()
        ranking = forager.rank_actions_costed(beliefs, available_tools)
        # ranking[0] = highest EIG/$ tool
    """

    def __init__(self, budget: ForagingBudget = None,
                 cost_profiles: Dict[str, ToolCostProfile] = None):
        super().__init__()
        self.budget = budget or ForagingBudget()
        self.costs = cost_profiles or {k: ToolCostProfile(
            k, v.api_cost_usd, v.avg_latency_ms, v.compute_cost_usd, v.reliability)
            for k, v in DEFAULT_COSTS.items()}

        # Runtime tracking
        self._total_cost_usd: float = 0.0
        self._total_latency_ms: int = 0
        self._total_calls: int = 0
        self._variable_query_counts: Dict[str, int] = {}
        self._call_history: List[Dict] = []

    def rank_actions_costed(self, beliefs: BeliefState,
                             available_tools: List[str],
                             already_called: Set[str] = None) -> List[Dict]:
        """Rank tools by EIG/$ with real cost model.

        Returns ordered list with full cost breakdown.
        """
        already_called = already_called or set()
        rankings = []

        remaining_budget = self.budget.max_cost_usd - self._total_cost_usd
        remaining_latency = self.budget.max_latency_ms - self._total_latency_ms

        for tool in available_tools:
            if tool in already_called:
                continue

            info_map = self.TOOL_INFORMATION_MAP.get(tool, {})
            if not info_map:
                continue

            cost_profile = self.costs.get(tool, ToolCostProfile(tool))

            # Skip if over budget
            if cost_profile.effective_cost > remaining_budget:
                continue
            if cost_profile.avg_latency_ms > remaining_latency:
                continue

            # Calculate EIG with diminishing returns penalty
            total_eig = 0.0
            variables_helped = []

            for var, reduction_strength in info_map.items():
                belief = beliefs.get(var)
                if belief is None:
                    continue

                # Base EIG
                current_entropy = max(0, belief.entropy)
                eig = current_entropy * reduction_strength * (1 - belief.confidence)

                # Diminishing returns: penalize re-querying same variable
                query_count = self._variable_query_counts.get(var, 0)
                if query_count > 0:
                    eig *= self.budget.diminishing_returns_penalty ** query_count

                total_eig += eig
                if eig > 0.01:
                    variables_helped.append({
                        "variable": var,
                        "entropy": round(current_entropy, 2),
                        "confidence": round(belief.confidence, 2),
                        "eig": round(eig, 3),
                        "prior_queries": query_count,
                    })

            # EIG per dollar
            effective_cost = max(cost_profile.effective_cost, 0.001)
            eig_per_dollar = total_eig / effective_cost

            # EIG per second (latency-adjusted)
            latency_sec = max(cost_profile.avg_latency_ms / 1000, 0.01)
            eig_per_second = total_eig / latency_sec

            # Combined score: weighted sum of EIG/$  and EIG/sec
            # Weights: 60% cost efficiency, 40% time efficiency
            combined_score = 0.6 * eig_per_dollar + 0.4 * eig_per_second

            if total_eig > 0.01:
                rankings.append({
                    "tool": tool,
                    "eig": round(total_eig, 3),
                    "cost_usd": round(cost_profile.effective_cost, 3),
                    "latency_ms": cost_profile.avg_latency_ms,
                    "eig_per_dollar": round(eig_per_dollar, 3),
                    "eig_per_second": round(eig_per_second, 3),
                    "combined_score": round(combined_score, 3),
                    "reliability": cost_profile.reliability,
                    "variables": variables_helped,
                    "within_budget": True,
                })

        # Sort by combined score
        rankings.sort(key=lambda x: x["combined_score"], reverse=True)

        # Mark economic stopping point
        for i, r in enumerate(rankings):
            if r["eig_per_dollar"] < self.budget.min_eig_per_dollar:
                r["below_eig_threshold"] = True

        return rankings

    def should_stop(self) -> Tuple[bool, str]:
        """Should we stop foraging? Returns (stop, reason)."""
        if self._total_cost_usd >= self.budget.max_cost_usd:
            return True, f"Cost budget exhausted (${self._total_cost_usd:.2f}/${self.budget.max_cost_usd:.2f})"
        if self._total_latency_ms >= self.budget.max_latency_ms:
            return True, f"Latency budget exhausted ({self._total_latency_ms}ms/{self.budget.max_latency_ms}ms)"
        if self._total_calls >= self.budget.max_calls:
            return True, f"Call budget exhausted ({self._total_calls}/{self.budget.max_calls})"
        return False, "OK"

    def record_call(self, tool: str, actual_cost_usd: float,
                    actual_latency_ms: int, info_gained: float,
                    variables_updated: List[str], success: bool = True):
        """Record a completed tool call for tracking."""
        self._total_cost_usd += actual_cost_usd
        self._total_latency_ms += actual_latency_ms
        self._total_calls += 1

        for var in variables_updated:
            self._variable_query_counts[var] = self._variable_query_counts.get(var, 0) + 1

        # Update cost profile
        if tool in self.costs:
            self.costs[tool].record_call(actual_cost_usd, actual_latency_ms,
                                          info_gained, success)

        self._call_history.append({
            "tool": tool, "cost": actual_cost_usd,
            "latency_ms": actual_latency_ms,
            "eig_actual": info_gained,
            "variables": variables_updated,
            "success": success,
            "cumulative_cost": self._total_cost_usd,
        })

    def cost_report(self) -> Dict:
        """Generate a cost report for the current analysis."""
        tool_roi = {}
        for name, profile in self.costs.items():
            if profile.total_calls > 0:
                tool_roi[name] = {
                    "calls": profile.total_calls,
                    "total_cost": round(profile.total_cost_usd, 3),
                    "total_eig": round(profile.total_info_gained, 3),
                    "roi": round(profile.actual_roi, 2),
                    "avg_latency": profile.total_latency_ms // max(profile.total_calls, 1),
                    "failures": profile.failures,
                }

        return {
            "total_cost_usd": round(self._total_cost_usd, 3),
            "total_latency_ms": self._total_latency_ms,
            "total_calls": self._total_calls,
            "budget_utilization": {
                "cost_pct": round(self._total_cost_usd / max(self.budget.max_cost_usd, 0.01) * 100, 1),
                "latency_pct": round(self._total_latency_ms / max(self.budget.max_latency_ms, 1) * 100, 1),
                "calls_pct": round(self._total_calls / max(self.budget.max_calls, 1) * 100, 1),
            },
            "variable_query_counts": dict(self._variable_query_counts),
            "tool_roi": tool_roi,
            "call_history": self._call_history[-20:],
        }
