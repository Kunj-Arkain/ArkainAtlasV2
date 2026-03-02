"""
engine.capital_stack — Capital Stack Simulator
================================================
Simulates the full capital structure of a CRE deal:
  Senior debt → Mezzanine → Preferred equity → Common equity

For each tranche:
  - Compute cash flow waterfall (who gets paid first)
  - IRR sensitivity to NOI, rates, exit cap
  - Default probability under correlated stress
  - Equity dilution scenarios

This is what makes the engine investor-grade.
A risk engine says "P(loss) = 18%".
A capital allocator says "Under 200bp rate shock, the mezz tranche
defaults in 73% of scenarios and equity IRR drops from 18% to -4%."

Uses correlated Monte Carlo for stress testing.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from .determinism import get_rng
    from .correlated_mc import CorrelatedDrawEngine
except ImportError:
    from determinism import get_rng
    from correlated_mc import CorrelatedDrawEngine


# ═══════════════════════════════════════════════════════════════
# TRANCHE DEFINITIONS
# ═══════════════════════════════════════════════════════════════

@dataclass
class DebtTranche:
    """A single debt layer in the capital stack."""
    name: str                    # "Senior", "Mezz A", "Mezz B"
    amount: float                # Principal
    rate: float                  # Annual interest rate (decimal, e.g. 0.07)
    term_years: int = 25         # Amortization period
    io_years: int = 0            # Interest-only period
    priority: int = 1            # 1 = senior (paid first), 2 = mezz, etc.
    prepayment_penalty_pct: float = 0.0
    is_fixed: bool = True        # Fixed vs floating
    spread_over_base: float = 0.0  # For floating: spread over base rate

    @property
    def annual_debt_service(self) -> float:
        """Compute annual debt service (P&I or I-only)."""
        if self.rate <= 0 or self.amount <= 0:
            return 0
        monthly_rate = self.rate / 12
        if self.io_years > 0:
            return self.amount * self.rate  # Interest only
        months = self.term_years * 12
        monthly_pmt = self.amount * (monthly_rate * (1 + monthly_rate) ** months) / \
                      ((1 + monthly_rate) ** months - 1)
        return monthly_pmt * 12

    def remaining_balance(self, years_elapsed: int) -> float:
        """Remaining principal after n years of amortization."""
        if self.rate <= 0 or self.amount <= 0:
            return self.amount
        if years_elapsed <= self.io_years:
            return self.amount  # No principal paydown during IO
        amort_years = years_elapsed - self.io_years
        monthly_rate = self.rate / 12
        total_months = self.term_years * 12
        elapsed_months = amort_years * 12
        if elapsed_months >= total_months:
            return 0
        factor_total = (1 + monthly_rate) ** total_months
        factor_elapsed = (1 + monthly_rate) ** elapsed_months
        return self.amount * (factor_total - factor_elapsed) / (factor_total - 1)

    def dscr(self, noi: float) -> float:
        """Debt service coverage ratio for this tranche."""
        ds = self.annual_debt_service
        return noi / max(ds, 1)

    def ltv(self, property_value: float) -> float:
        return self.amount / max(property_value, 1)


@dataclass
class EquityTranche:
    """Equity layer in the capital stack."""
    name: str                    # "GP Equity", "LP Equity", "Preferred"
    amount: float
    preferred_return: float = 0.0  # Annual preferred return (decimal)
    promote_above: float = 0.0     # IRR hurdle for promote
    promote_split: float = 0.0     # GP share above hurdle (e.g. 0.20)
    priority: int = 10             # Equity is always behind debt


@dataclass
class CapitalStack:
    """Full capital structure of a deal."""
    purchase_price: float
    debt_tranches: List[DebtTranche] = field(default_factory=list)
    equity_tranches: List[EquityTranche] = field(default_factory=list)

    @property
    def total_debt(self) -> float:
        return sum(t.amount for t in self.debt_tranches)

    @property
    def total_equity(self) -> float:
        return sum(t.amount for t in self.equity_tranches)

    @property
    def total_capitalization(self) -> float:
        return self.total_debt + self.total_equity

    @property
    def overall_ltv(self) -> float:
        return self.total_debt / max(self.purchase_price, 1)

    @property
    def total_annual_debt_service(self) -> float:
        return sum(t.annual_debt_service for t in self.debt_tranches)

    def blended_rate(self) -> float:
        """Weighted average cost of debt."""
        if self.total_debt == 0:
            return 0
        return sum(t.amount * t.rate for t in self.debt_tranches) / self.total_debt

    def wacc(self, cost_of_equity: float = 0.15) -> float:
        """Weighted average cost of capital."""
        total = self.total_capitalization
        if total == 0:
            return 0
        debt_weight = self.total_debt / total
        equity_weight = self.total_equity / total
        return debt_weight * self.blended_rate() + equity_weight * cost_of_equity

    def summary(self) -> Dict:
        return {
            "purchase_price": self.purchase_price,
            "total_debt": self.total_debt,
            "total_equity": self.total_equity,
            "overall_ltv": round(self.overall_ltv * 100, 1),
            "blended_rate": round(self.blended_rate() * 100, 2),
            "total_debt_service": round(self.total_annual_debt_service),
            "tranches": [
                {"name": t.name, "amount": t.amount, "rate": round(t.rate * 100, 2),
                 "priority": t.priority, "annual_ds": round(t.annual_debt_service)}
                for t in sorted(self.debt_tranches, key=lambda x: x.priority)
            ] + [
                {"name": t.name, "amount": t.amount,
                 "pref_return": round(t.preferred_return * 100, 2),
                 "priority": t.priority}
                for t in self.equity_tranches
            ],
        }


# ═══════════════════════════════════════════════════════════════
# WATERFALL ENGINE
# ═══════════════════════════════════════════════════════════════

def compute_waterfall(stack: CapitalStack, noi: float,
                      exit_value: float, hold_years: int,
                      noi_growth: float = 0.02) -> Dict:
    """Compute the full cash flow waterfall.

    Returns per-tranche: cash flows, IRR, equity multiple, DSCR.
    """
    # Sort by priority (senior first)
    debt_sorted = sorted(stack.debt_tranches, key=lambda t: t.priority)
    equity_sorted = sorted(stack.equity_tranches, key=lambda t: t.priority)

    # Annual cash flows
    tranche_cfs = {t.name: [-t.amount] for t in debt_sorted + equity_sorted}
    # For equity, initial outflow is the equity amount
    for t in equity_sorted:
        tranche_cfs[t.name] = [-t.amount]
    # For debt, lender perspective: outflow = loan amount, inflow = debt service
    for t in debt_sorted:
        tranche_cfs[t.name] = [-t.amount]  # Lender puts money in

    # Operating period
    residual_cfs = []  # What's left for equity each year
    for yr in range(1, hold_years + 1):
        yr_noi = noi * (1 + noi_growth) ** (yr - 1)
        remaining = yr_noi

        for t in debt_sorted:
            ds = t.annual_debt_service
            paid = min(remaining, ds)
            tranche_cfs[t.name].append(paid)
            remaining -= paid

        # Remaining goes to equity (split by priority)
        for t in equity_sorted:
            pref = t.amount * t.preferred_return
            paid = min(remaining, pref) if remaining > 0 else 0
            tranche_cfs[t.name].append(paid)
            remaining -= paid

        residual_cfs.append(remaining)

    # Exit proceeds waterfall
    total_remaining_debt = sum(t.remaining_balance(hold_years) for t in debt_sorted)
    exit_proceeds = exit_value

    # Pay off debt (senior first)
    for t in debt_sorted:
        bal = t.remaining_balance(hold_years)
        prepay = bal * t.prepayment_penalty_pct
        payoff = bal + prepay
        paid = min(exit_proceeds, payoff)
        tranche_cfs[t.name][-1] += paid  # Add to last year
        exit_proceeds -= paid

    # Remaining to equity
    for t in equity_sorted:
        share = min(exit_proceeds, t.amount * 2)  # Cap at 2x for safety
        tranche_cfs[t.name][-1] += share
        exit_proceeds -= share

    # Compute per-tranche metrics
    results = {"tranches": {}, "residual_equity_cfs": residual_cfs}

    for t in debt_sorted + equity_sorted:
        cfs = tranche_cfs[t.name]
        irr = _compute_irr(cfs)
        total_in = sum(cf for cf in cfs[1:] if cf > 0)
        total_out = abs(cfs[0]) if cfs else 1

        results["tranches"][t.name] = {
            "cash_flows": [round(cf) for cf in cfs],
            "irr": round(irr * 100, 2) if irr else None,
            "equity_multiple": round(total_in / max(total_out, 1), 2),
            "total_distributions": round(total_in),
            "investment": round(total_out),
        }

    # Overall deal metrics
    deal_cfs = [-(stack.total_equity)]
    for yr_cf in residual_cfs:
        deal_cfs.append(yr_cf)
    deal_cfs[-1] += max(exit_proceeds, 0)  # Residual after all tranches

    deal_irr = _compute_irr(deal_cfs)
    results["deal"] = {
        "equity_irr": round(deal_irr * 100, 2) if deal_irr else None,
        "equity_multiple": round(sum(cf for cf in deal_cfs[1:]) / max(stack.total_equity, 1), 2),
        "overall_dscr": round(noi / max(stack.total_annual_debt_service, 1), 2),
        "breakeven_noi": round(stack.total_annual_debt_service),
        "debt_yield": round(noi / max(stack.total_debt, 1) * 100, 2),
    }

    return results


# ═══════════════════════════════════════════════════════════════
# STRESS TESTING (uses correlated MC)
# ═══════════════════════════════════════════════════════════════

def stress_test_stack(stack: CapitalStack, base_params: Dict,
                      num_simulations: int = 2000) -> Dict:
    """Run correlated Monte Carlo stress test on the full capital stack.

    Returns per-tranche: default probability, IRR distribution, loss severity.

    base_params:
      noi: {point, low, high}
      loan_rate: {point, low, high}  — for floating rate tranches
      exit_cap: {point, low, high}
      noi_growth: {point, low, high}
      hold_years: int
    """
    rng = get_rng()
    engine = CorrelatedDrawEngine()
    hold = base_params.get("hold_years", 5)

    # Per-tranche accumulators
    tranche_names = [t.name for t in stack.debt_tranches] + \
                    [t.name for t in stack.equity_tranches]
    irrs = {n: [] for n in tranche_names}
    defaults = {n: 0 for n in tranche_names}
    losses = {n: [] for n in tranche_names}

    equity_irrs = []
    deal_dscrs = []

    for _ in range(num_simulations):
        draws = engine.draw(base_params, rng)
        noi = draws["noi"]
        rate_shock = draws["loan_rate"] / 100
        exit_cap = draws["exit_cap"] / 100
        noi_growth = draws["noi_growth"] / 100

        # Apply rate shock to floating tranches
        stressed_stack = _apply_rate_shock(stack, rate_shock)

        # Exit value
        exit_noi = noi * (1 + noi_growth) ** hold
        exit_value = exit_noi / max(exit_cap, 0.01)

        # Run waterfall
        wf = compute_waterfall(stressed_stack, noi, exit_value, hold, noi_growth)

        # Collect results
        for name in tranche_names:
            if name in wf["tranches"]:
                t_data = wf["tranches"][name]
                t_irr = t_data["irr"]
                if t_irr is not None:
                    irrs[name].append(t_irr)
                # Default = didn't get full principal back
                if t_data["total_distributions"] < t_data["investment"] * 0.95:
                    defaults[name] += 1
                    loss = 1 - (t_data["total_distributions"] / max(t_data["investment"], 1))
                    losses[name].append(loss)

        if wf["deal"]["equity_irr"] is not None:
            equity_irrs.append(wf["deal"]["equity_irr"])
        deal_dscrs.append(wf["deal"]["overall_dscr"])

    # Compile results
    n = num_simulations
    results = {
        "simulations": n,
        "stack_summary": stack.summary(),
        "tranches": {},
        "deal": {
            "equity_irr_median": round(_median(equity_irrs), 2) if equity_irrs else None,
            "equity_irr_p5": round(_percentile(equity_irrs, 5), 2) if equity_irrs else None,
            "equity_irr_p95": round(_percentile(equity_irrs, 95), 2) if equity_irrs else None,
            "dscr_median": round(_median(deal_dscrs), 2),
            "dscr_p5": round(_percentile(deal_dscrs, 5), 2),
            "prob_dscr_below_1": round(sum(1 for d in deal_dscrs if d < 1.0) / n, 3),
        },
    }

    for name in tranche_names:
        t_irr_list = irrs[name]
        t_losses = losses[name]
        results["tranches"][name] = {
            "default_probability": round(defaults[name] / n, 3),
            "expected_loss_given_default": round(_mean(t_losses), 3) if t_losses else 0,
            "expected_loss": round(defaults[name] / n * (_mean(t_losses) if t_losses else 0), 4),
            "irr_median": round(_median(t_irr_list), 2) if t_irr_list else None,
            "irr_p5": round(_percentile(t_irr_list, 5), 2) if t_irr_list else None,
            "irr_p95": round(_percentile(t_irr_list, 95), 2) if t_irr_list else None,
        }

    return results


def _apply_rate_shock(stack: CapitalStack, market_rate: float) -> CapitalStack:
    """Create a copy of the stack with floating rates adjusted."""
    new_debt = []
    for t in stack.debt_tranches:
        if t.is_fixed:
            new_debt.append(t)
        else:
            new_t = DebtTranche(
                name=t.name, amount=t.amount,
                rate=market_rate + t.spread_over_base,
                term_years=t.term_years, io_years=t.io_years,
                priority=t.priority, is_fixed=False,
                spread_over_base=t.spread_over_base,
            )
            new_debt.append(new_t)
    return CapitalStack(
        purchase_price=stack.purchase_price,
        debt_tranches=new_debt,
        equity_tranches=stack.equity_tranches,
    )


# ═══════════════════════════════════════════════════════════════
# SCENARIO COMPARISON — Multi-stack analysis
# ═══════════════════════════════════════════════════════════════

def compare_structures(purchase_price: float, noi: float,
                       structures: List[Dict],
                       base_params: Dict,
                       num_simulations: int = 1000) -> Dict:
    """Compare multiple capital structures side-by-side.

    structures: list of dicts defining each structure, e.g.:
      [
        {"name": "Conservative", "senior_ltv": 0.65, "senior_rate": 0.065},
        {"name": "Aggressive", "senior_ltv": 0.75, "mezz_ltv": 0.10, "mezz_rate": 0.12},
      ]
    """
    results = {"structures": []}

    for struct in structures:
        stack = _build_stack_from_spec(purchase_price, struct)
        stress = stress_test_stack(stack, base_params, num_simulations)

        results["structures"].append({
            "name": struct["name"],
            "ltv": round(stack.overall_ltv * 100, 1),
            "equity_required": round(stack.total_equity),
            "annual_debt_service": round(stack.total_annual_debt_service),
            "base_dscr": round(noi / max(stack.total_annual_debt_service, 1), 2),
            "equity_irr_median": stress["deal"]["equity_irr_median"],
            "equity_irr_p5": stress["deal"]["equity_irr_p5"],
            "prob_dscr_below_1": stress["deal"]["prob_dscr_below_1"],
            "default_probs": {n: t["default_probability"]
                             for n, t in stress["tranches"].items()},
        })

    return results


def _build_stack_from_spec(price: float, spec: Dict) -> CapitalStack:
    """Build a CapitalStack from a simplified spec dict."""
    debt = []
    senior_amt = price * spec.get("senior_ltv", 0.75)
    debt.append(DebtTranche(
        name="Senior", amount=senior_amt,
        rate=spec.get("senior_rate", 0.07),
        term_years=spec.get("senior_term", 25),
        io_years=spec.get("senior_io", 0),
        priority=1, is_fixed=spec.get("senior_fixed", True),
        spread_over_base=spec.get("senior_spread", 0.0),
    ))

    if "mezz_ltv" in spec:
        mezz_amt = price * spec["mezz_ltv"]
        debt.append(DebtTranche(
            name="Mezzanine", amount=mezz_amt,
            rate=spec.get("mezz_rate", 0.12),
            term_years=spec.get("mezz_term", 10),
            io_years=spec.get("mezz_io", 2),
            priority=2,
        ))

    total_debt = sum(t.amount for t in debt)
    equity_amt = price - total_debt

    equity = [EquityTranche(
        name=spec.get("equity_name", "Equity"),
        amount=equity_amt,
        preferred_return=spec.get("pref_return", 0.08),
        priority=10,
    )]

    return CapitalStack(purchase_price=price, debt_tranches=debt, equity_tranches=equity)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _compute_irr(cashflows: List[float], max_iter: int = 100) -> Optional[float]:
    """Newton's method IRR."""
    if not cashflows or len(cashflows) < 2:
        return None
    r = 0.10
    for _ in range(max_iter):
        npv = sum(cf / (1 + r) ** t for t, cf in enumerate(cashflows))
        dnpv = sum(-t * cf / (1 + r) ** (t + 1) for t, cf in enumerate(cashflows))
        if abs(dnpv) < 1e-12:
            break
        r_new = r - npv / dnpv
        if abs(r_new - r) < 1e-8:
            return r_new
        r = max(min(r_new, 5.0), -0.99)
    return r


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)

def _median(xs: List[float]) -> float:
    if not xs:
        return 0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

def _percentile(xs: List[float], pct: float) -> float:
    if not xs:
        return 0
    s = sorted(xs)
    idx = int(len(s) * pct / 100)
    return s[min(idx, len(s) - 1)]
