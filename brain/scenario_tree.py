"""
engine.scenario_tree — Strategic Scenario Tree Engine
=======================================================
Not just MC. Branching:
  Base case → Regulatory shock → Demand collapse → Supply shock → Execution failure

And compute expected utility across paths.

Structure:
  ScenarioNode: one state of the world
    - probability: how likely this branch is
    - parameter_overrides: what changes (NOI, rates, etc.)
    - children: sub-scenarios that branch from here
    - utility: computed deal outcome at this node

  ScenarioTree: root node + all branches
    - Compute expected utility = Σ(prob × utility) across leaf nodes
    - Find dominant strategy (which capital structure wins across scenarios)
    - Identify fragile strategies (high variance across branches)

This is executive-tier: "Given 5 possible futures,
which deal structure maximizes expected risk-adjusted return?"
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

try:
    from .determinism import get_rng, seed_engine
    from .correlated_mc import CorrelatedDrawEngine
except ImportError:
    from determinism import get_rng, seed_engine
    from correlated_mc import CorrelatedDrawEngine


# ═══════════════════════════════════════════════════════════════
# SCENARIO NODE
# ═══════════════════════════════════════════════════════════════

@dataclass
class ScenarioNode:
    """A single node in the scenario tree."""
    name: str
    probability: float                  # P(this branch) — must sum to 1.0 among siblings
    description: str = ""
    parameter_overrides: Dict = field(default_factory=dict)
    children: List["ScenarioNode"] = field(default_factory=list)
    # Computed after evaluation
    outcome: Optional[Dict] = None      # MC result or direct calc
    utility: Optional[float] = None     # Risk-adjusted return metric
    _path_prob: float = 0.0             # Cumulative probability from root

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def add_child(self, name: str, probability: float,
                  description: str = "", overrides: Dict = None) -> "ScenarioNode":
        child = ScenarioNode(
            name=name, probability=probability,
            description=description,
            parameter_overrides=overrides or {},
        )
        self.children.append(child)
        return child

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "probability": self.probability,
            "description": self.description,
            "overrides": self.parameter_overrides,
            "utility": round(self.utility, 2) if self.utility is not None else None,
            "outcome_summary": _summarize_outcome(self.outcome) if self.outcome else None,
            "path_probability": round(self._path_prob, 4),
            "children": [c.to_dict() for c in self.children],
        }


# ═══════════════════════════════════════════════════════════════
# STANDARD SCENARIO TEMPLATES
# ═══════════════════════════════════════════════════════════════

def cre_standard_tree(base_params: Dict) -> ScenarioNode:
    """Build a standard CRE scenario tree with 5 branches.

    Probabilities calibrated to historical CRE cycle frequencies.
    """
    root = ScenarioNode(name="Root", probability=1.0, description="All scenarios")

    # Base case (55% probability)
    base = root.add_child("Base Case", 0.55,
        "Economy continues on current trajectory. Rates stable, modest NOI growth.",
        overrides={})

    # Rate shock (15% probability)
    rate_shock = root.add_child("Rate Shock (+200bp)", 0.15,
        "Fed raises rates 200bp. Exit caps widen, debt costs spike.",
        overrides={
            "loan_rate": _shift_dist(base_params.get("loan_rate", {}), +2.0),
            "exit_cap": _shift_dist(base_params.get("exit_cap", {}), +1.5),
            "noi_growth": _shift_dist(base_params.get("noi_growth", {}), -1.0),
        })

    # Demand collapse (10% probability)
    demand = root.add_child("Demand Collapse", 0.10,
        "Local market downturn. Population decline, competitor oversupply.",
        overrides={
            "noi": _scale_dist(base_params.get("noi", {}), 0.75),
            "noi_growth": {"point": -3.0, "low": -8.0, "high": 0.0},
            "exit_cap": _shift_dist(base_params.get("exit_cap", {}), +2.0),
        })

    # Supply shock (10% probability)
    supply = root.add_child("Supply Shock", 0.10,
        "New competition enters market. Gas station oversupply or EV transition accelerates.",
        overrides={
            "noi": _scale_dist(base_params.get("noi", {}), 0.85),
            "noi_growth": {"point": -1.0, "low": -5.0, "high": 2.0},
        })

    # Execution failure (10% probability)
    execution = root.add_child("Execution Failure", 0.10,
        "Environmental remediation required, permit delays, construction overruns.",
        overrides={
            "noi": _scale_dist(base_params.get("noi", {}), 0.70),
            "exit_cap": _shift_dist(base_params.get("exit_cap", {}), +1.0),
        })

    return root


def expansion_tree(base_params: Dict, num_locations: int = 12) -> ScenarioNode:
    """Build a scenario tree for multi-location expansion.

    Models the question: "Should I open 12 locations?"
    """
    root = ScenarioNode(name="Root", probability=1.0,
                        description=f"Expand to {num_locations} locations")

    # Aggressive expansion
    aggressive = root.add_child("Full Expansion", 0.30,
        f"Open all {num_locations} locations within 18 months.",
        overrides={"location_count": num_locations, "execution_risk": 0.3})

    # Phased expansion
    phased = root.add_child("Phased Expansion", 0.40,
        f"Open {num_locations // 3} first, then {num_locations // 3}, then rest.",
        overrides={"location_count": num_locations, "execution_risk": 0.15,
                    "noi_growth": _shift_dist(base_params.get("noi_growth", {}), +0.5)})

    # Conservative
    conservative = root.add_child("Conservative (3 only)", 0.20,
        "Open 3 best locations only. Preserve capital.",
        overrides={"location_count": 3, "execution_risk": 0.05,
                    "noi": _scale_dist(base_params.get("noi", {}), 0.25)})

    # Abort
    abort = root.add_child("No Expansion", 0.10,
        "Hold current portfolio. Wait for better conditions.",
        overrides={"location_count": 0, "noi": {"point": 0, "low": 0, "high": 0}})

    return root


# ═══════════════════════════════════════════════════════════════
# TREE EVALUATOR
# ═══════════════════════════════════════════════════════════════

class ScenarioTreeEvaluator:
    """Evaluate a scenario tree by running MC at each leaf node.

    The evaluator:
      1. Walks the tree, computing path probabilities
      2. At each leaf: merges parameter overrides with base params
      3. Runs correlated MC to get outcome distributions
      4. Computes utility = risk-adjusted return metric
      5. Rolls up expected utility to root

    Utility function: CRRA (Constant Relative Risk Aversion)
      U(x) = x^(1-γ) / (1-γ)  where γ = risk aversion coefficient
      γ = 0: risk neutral (just use expected return)
      γ = 1: log utility
      γ = 2: moderately risk averse (typical institutional investor)
      γ = 5: highly risk averse (pension fund)
    """

    def __init__(self, base_params: Dict, num_simulations: int = 500,
                 risk_aversion: float = 2.0):
        self.base_params = base_params
        self.num_sims = num_simulations
        self.gamma = risk_aversion
        self.mc_engine = CorrelatedDrawEngine()

    def evaluate(self, root: ScenarioNode) -> Dict:
        """Evaluate the full tree. Returns summary with expected utility."""
        self._assign_path_probs(root, 1.0)
        self._evaluate_node(root)

        leaves = self._collect_leaves(root)
        expected_utility = sum(leaf._path_prob * (leaf.utility or 0) for leaf in leaves)
        certainty_equivalent = self._inverse_utility(expected_utility)

        # Risk decomposition
        utilities = [(leaf._path_prob, leaf.utility or 0, leaf.name) for leaf in leaves]
        variance = sum(p * (u - expected_utility) ** 2 for p, u, _ in utilities)

        return {
            "expected_utility": round(expected_utility, 4),
            "certainty_equivalent_irr": round(certainty_equivalent, 2),
            "utility_variance": round(variance, 4),
            "risk_aversion": self.gamma,
            "scenarios_evaluated": len(leaves),
            "scenario_breakdown": [
                {"name": leaf.name, "path_prob": round(leaf._path_prob, 3),
                 "utility": round(leaf.utility, 4) if leaf.utility else None,
                 "irr_median": leaf.outcome.get("irr_median") if leaf.outcome else None,
                 "prob_loss": leaf.outcome.get("prob_loss") if leaf.outcome else None}
                for leaf in leaves
            ],
            "tree": root.to_dict(),
        }

    def compare_strategies(self, root: ScenarioNode,
                           strategies: List[Dict]) -> Dict:
        """Compare multiple strategies across the same scenario tree.

        strategies: list of parameter override dicts, e.g.:
          [{"name": "Conservative 65% LTV", "loan_ltv": 0.65},
           {"name": "Aggressive 85% LTV", "loan_ltv": 0.85}]
        """
        results = {"strategies": []}

        for strategy in strategies:
            # Apply strategy overrides to base params
            merged = {**self.base_params, **strategy}
            # Re-evaluate tree with these params
            import copy
            tree_copy = copy.deepcopy(root)
            self.base_params = merged
            eval_result = self.evaluate(tree_copy)
            self.base_params = {**self.base_params}  # Reset

            results["strategies"].append({
                "name": strategy.get("name", "Unnamed"),
                "expected_utility": eval_result["expected_utility"],
                "certainty_equivalent_irr": eval_result["certainty_equivalent_irr"],
                "utility_variance": eval_result["utility_variance"],
                "worst_case_irr": min(
                    (s["irr_median"] or -99 for s in eval_result["scenario_breakdown"]),
                    default=-99),
                "best_case_irr": max(
                    (s["irr_median"] or -99 for s in eval_result["scenario_breakdown"]),
                    default=0),
            })

        # Find dominant strategy
        results["strategies"].sort(key=lambda s: s["expected_utility"], reverse=True)
        results["dominant_strategy"] = results["strategies"][0]["name"]
        results["fragility_ranking"] = sorted(
            results["strategies"],
            key=lambda s: s["utility_variance"], reverse=True)

        return results

    def _assign_path_probs(self, node: ScenarioNode, parent_prob: float):
        """Recursively assign cumulative path probabilities."""
        node._path_prob = parent_prob * node.probability
        for child in node.children:
            self._assign_path_probs(child, node._path_prob)

    def _evaluate_node(self, node: ScenarioNode):
        """Recursively evaluate. Leaves get MC; parents get rolled-up utility."""
        if node.is_leaf():
            merged_params = {**self.base_params, **node.parameter_overrides}
            node.outcome = self._run_mc(merged_params)
            node.utility = self._compute_utility(node.outcome)
        else:
            for child in node.children:
                self._evaluate_node(child)
            # Parent utility = probability-weighted child utilities
            node.utility = sum(
                c.probability * (c.utility or 0) for c in node.children)
            node.outcome = {"rolled_up": True,
                           "irr_median": node.utility}

    def _run_mc(self, params: Dict) -> Dict:
        """Run a small MC simulation for a scenario leaf."""
        try:
            from .seal_ceca import MonteCarloSimulator
        except ImportError:
            from seal_ceca import MonteCarloSimulator
        mc = MonteCarloSimulator(num_simulations=self.num_sims)
        if not any(isinstance(params.get(k), dict) for k in ["noi", "exit_cap"]):
            return {"irr_median": 0, "prob_loss": 1.0}
        result = mc.simulate_deal(params)
        return {
            "irr_median": result["irr"]["median"],
            "irr_p5": result["irr"]["p5"],
            "irr_p95": result["irr"]["p95"],
            "prob_loss": result["probability_analysis"]["prob_loss"],
            "prob_irr_above_15": result["probability_analysis"]["prob_irr_above_15"],
            "dscr_median": result["dscr_year1"]["median"],
            "var_5pct": result["probability_analysis"]["value_at_risk_5pct"],
        }

    def _compute_utility(self, outcome: Dict) -> float:
        """CRRA utility of the median IRR."""
        irr = outcome.get("irr_median", 0) / 100  # Convert to decimal
        # Shift to ensure positive (utility of wealth, not return)
        wealth = 1.0 + irr  # $1 invested becomes $1+r
        if wealth <= 0:
            return -1e6  # Catastrophic loss
        if abs(self.gamma - 1.0) < 0.001:
            return math.log(wealth)
        return (wealth ** (1 - self.gamma)) / (1 - self.gamma)

    def _inverse_utility(self, u: float) -> float:
        """Convert utility back to certainty-equivalent IRR."""
        if abs(self.gamma - 1.0) < 0.001:
            return (math.exp(u) - 1) * 100
        wealth = (u * (1 - self.gamma)) ** (1 / (1 - self.gamma)) if u * (1 - self.gamma) > 0 else 0
        return (wealth - 1) * 100

    def _collect_leaves(self, node: ScenarioNode) -> List[ScenarioNode]:
        """Collect all leaf nodes."""
        if node.is_leaf():
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self._collect_leaves(child))
        return leaves


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _shift_dist(dist: Dict, shift: float) -> Dict:
    """Shift a distribution {point, low, high} by a fixed amount."""
    if not isinstance(dist, dict):
        return {"point": float(dist) + shift, "low": float(dist) + shift * 0.5,
                "high": float(dist) + shift * 1.5}
    return {
        "point": dist.get("point", 0) + shift,
        "low": dist.get("low", 0) + shift * 0.5,
        "high": dist.get("high", 0) + shift * 1.5,
    }

def _scale_dist(dist: Dict, factor: float) -> Dict:
    """Scale a distribution by a factor."""
    if not isinstance(dist, dict):
        return {"point": float(dist) * factor, "low": float(dist) * factor * 0.9,
                "high": float(dist) * factor * 1.1}
    return {
        "point": dist.get("point", 0) * factor,
        "low": dist.get("low", 0) * factor,
        "high": dist.get("high", 0) * factor,
    }

def _summarize_outcome(outcome: Dict) -> Dict:
    """Short summary of an MC outcome."""
    if not outcome:
        return {}
    return {k: v for k, v in outcome.items()
            if k in ("irr_median", "prob_loss", "dscr_median", "var_5pct")}
