"""
engine.realestate.stages — Pipeline Evaluation Stages
=========================================================
Phase 6A: Seven-stage deal evaluation pipeline.

Stages:
  1. Intake — validate property data + apply template defaults
  2. Feasibility — zoning, environmental, preliminary screen
  3. Market — comp analysis, demand drivers, market score
  4. Cost — renovation/build estimates, capex schedule
  5. Finance — capital structure, debt terms, IRR, DSCR
  6. Risk — Monte Carlo + downside scenario analysis
  7. Decision — weighted score → GO / HOLD / NO-GO

Each stage is a pure function: f(inputs, params) → stage_result dict.
No DB dependencies — the pipeline orchestrator handles persistence.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from engine.realestate.state_config import (
    get_state, get_business_context, is_gaming_legal,
    GAMING_LEGAL, GAMING_PENDING, GAMING_CASINO_ONLY, GAMING_TRIBAL_ONLY,
)


# ═══════════════════════════════════════════════════════════════
# STAGE 1 — INTAKE
# ═══════════════════════════════════════════════════════════════

def stage_intake(inputs: Dict, defaults: Dict) -> Dict:
    """Validate and normalize property inputs.

    Required inputs: purchase_price, address, property_type
    Optional: noi, sqft, lot_size, year_built, num_units
    Fills missing values from template defaults.
    """
    errors = []
    if not inputs.get("purchase_price"):
        errors.append("purchase_price is required")
    if not inputs.get("address"):
        errors.append("address is required")

    # Merge defaults for missing optional fields
    params = dict(defaults)
    params.update({k: v for k, v in inputs.items() if v is not None})

    # Enrich with state data
    state = params.get("state", "")
    property_type = params.get("property_type", "retail_strip")
    if state:
        state_ctx = get_business_context(state, property_type)
        params["_state_context"] = state_ctx

        # Apply state-specific defaults if not already set
        if not params.get("property_tax_rate") and state_ctx.get("property_tax_rate"):
            params["property_tax_rate"] = state_ctx["property_tax_rate"]
        if not params.get("closing_cost_pct") and state_ctx.get("closing_cost_pct"):
            params["closing_cost_pct"] = state_ctx["closing_cost_pct"]

        # Auto-set gaming eligibility based on state law
        if params.get("gaming_eligible") and not state_ctx.get("gaming_legal"):
            gaming_status = state_ctx.get("gaming_status", "none")
            if gaming_status not in (GAMING_LEGAL, GAMING_PENDING):
                params["_gaming_warning"] = f"Gaming not legal in {state} (status: {gaming_status})"

    # Derive NOI if not provided
    if not params.get("noi") and params.get("purchase_price"):
        # Use state-specific avg cap rate if available
        default_cap = 0.07
        if state and "_state_context" in params:
            default_cap = params["_state_context"].get("avg_cap_rate", 0.07)
        cap = params.get("cap_rate", default_cap)
        params["noi"] = round(params["purchase_price"] * cap, 2)
        params["noi_source"] = "derived_from_cap_rate"

    if errors:
        return {"status": "fail", "errors": errors, "params": params}

    return {"status": "pass", "errors": [], "params": params}


# ═══════════════════════════════════════════════════════════════
# STAGE 2 — FEASIBILITY
# ═══════════════════════════════════════════════════════════════

def stage_feasibility(params: Dict) -> Dict:
    """Preliminary feasibility screen.

    Checks: price floor, cap rate floor, gaming legality, state-specific risks.
    """
    score = 1.0
    flags = []

    price = params.get("purchase_price", 0)
    noi = params.get("noi", 0)
    cap_rate = noi / price if price > 0 else 0
    state = params.get("state", "")
    state_ctx = params.get("_state_context", {})

    # Price sanity
    if price < 50000:
        flags.append("Price below $50K — micro deal")
        score -= 0.2
    if price > 20000000:
        flags.append("Price above $20M — institutional deal")
        score -= 0.1

    # Cap rate floor
    min_cap = params.get("min_cap_rate", 0.04)
    if cap_rate < min_cap:
        flags.append(f"Going-in cap rate {cap_rate:.2%} below minimum {min_cap:.2%}")
        score -= 0.3

    # State-specific cap rate comparison
    avg_cap = state_ctx.get("avg_cap_rate")
    if avg_cap and cap_rate > 0:
        if cap_rate > avg_cap * 1.3:
            flags.append(f"Cap rate {cap_rate:.1%} significantly above state avg {avg_cap:.1%} — verify NOI")
        elif cap_rate < avg_cap * 0.7:
            flags.append(f"Cap rate {cap_rate:.1%} below state avg {avg_cap:.1%} — may be overpriced")
            score -= 0.1

    # Gaming eligibility — state-aware
    gaming = params.get("gaming_eligible", False)
    if gaming:
        if is_gaming_legal(state):
            flags.append(f"Gaming legal in {state} ✓")
            max_terms = state_ctx.get("gaming_max_terminals")
            requested_terms = params.get("terminal_count", 0)
            if max_terms and requested_terms and requested_terms > max_terms:
                flags.append(f"Requested {requested_terms} terminals exceeds {state} max of {max_terms}")
                score -= 0.1
            # Check liquor license requirement
            if state_ctx.get("gaming_liquor_required"):
                flags.append(f"{state} requires liquor license for gaming")
        else:
            gaming_status = state_ctx.get("gaming_status", "none")
            if gaming_status == GAMING_PENDING:
                flags.append(f"Gaming legislation pending in {state} — speculative")
                score -= 0.1
            elif gaming_status in (GAMING_CASINO_ONLY, GAMING_TRIBAL_ONLY):
                flags.append(f"Only casino/tribal gaming in {state} — no VGTs allowed")
                score -= 0.2
            else:
                flags.append(f"No gaming in {state}")
                score -= 0.2

    # High property tax warning
    prop_tax = state_ctx.get("property_tax_rate", 0)
    if prop_tax > 0.015:
        flags.append(f"High property tax state ({prop_tax:.2%} avg) — factor into NOI")

    # Transfer tax warning
    transfer_tax = state_ctx.get("transfer_tax", 0)
    if transfer_tax > 0:
        flags.append(f"Transfer tax: {transfer_tax:.1%} — adds to closing costs")

    # Gaming warning from intake
    if params.get("_gaming_warning"):
        flags.append(params["_gaming_warning"])

    score = max(0.0, min(1.0, score))

    result = {
        "status": "pass" if score >= 0.4 else "fail",
        "score": round(score, 4),
        "going_in_cap_rate": round(cap_rate, 6),
        "flags": flags,
        "gaming_eligible": gaming and is_gaming_legal(state),
    }

    # Include state context summary
    if state_ctx:
        result["state_context"] = {
            "state": state,
            "state_name": state_ctx.get("state_name", ""),
            "gaming_status": state_ctx.get("gaming_status", ""),
            "gaming_tax_rate": state_ctx.get("gaming_tax_rate"),
            "property_tax_rate": state_ctx.get("property_tax_rate"),
            "avg_cap_rate": avg_cap,
        }

    return result


# ═══════════════════════════════════════════════════════════════
# STAGE 3 — MARKET
# ═══════════════════════════════════════════════════════════════

def stage_market(params: Dict, market_context: Dict = None) -> Dict:
    """Market analysis and scoring.

    market_context can include: comp_cap_rates, vacancy_rates,
    rent_growth, population_growth, gaming_location_count, etc.
    """
    ctx = market_context or {}
    score = 0.7  # Neutral baseline
    state_ctx = params.get("_state_context", {})

    rent_growth = params.get("rent_growth_annual", 0.02)
    vacancy = params.get("vacancy_rate", 0.08)

    # Adjust for market conditions
    if rent_growth >= 0.03:
        score += 0.15
    elif rent_growth <= 0.01:
        score -= 0.10

    if vacancy <= 0.05:
        score += 0.10
    elif vacancy >= 0.12:
        score -= 0.15

    # Gaming market density (from Phase 3/4 data)
    gaming_locations = ctx.get("gaming_location_count", 0)
    if gaming_locations > 100:
        score -= 0.05  # Saturated
    elif 20 < gaming_locations < 60:
        score += 0.05  # Healthy market

    avg_nti = ctx.get("avg_monthly_nti", 0)

    # State-level market adjustment
    if state_ctx:
        region = state_ctx.get("region", "")
        if region in ("west", "northeast"):
            # Higher-cost markets tend to have tighter cap rates
            score += 0.05
        # Gaming market NTI from state data if not in context
        if not avg_nti and state_ctx.get("gaming_avg_nti"):
            avg_nti = state_ctx["gaming_avg_nti"]

    score = max(0.0, min(1.0, score))

    return {
        "status": "pass",
        "score": round(score, 4),
        "rent_growth": rent_growth,
        "vacancy_rate": vacancy,
        "gaming_market": {
            "location_count": gaming_locations,
            "avg_monthly_nti": avg_nti,
        },
    }


# ═══════════════════════════════════════════════════════════════
# STAGE 4 — COST
# ═══════════════════════════════════════════════════════════════

def stage_cost(params: Dict) -> Dict:
    """Renovation and capex estimation."""
    price = params.get("purchase_price", 0)
    state_ctx = params.get("_state_context", {})

    reno_pct = params.get("renovation_pct_of_price", 0.10)

    # Use explicit renovation budget if provided
    if params.get("renovation_budget"):
        renovation_cost = float(params["renovation_budget"])
        reno_pct = renovation_cost / price if price > 0 else 0
    else:
        renovation_cost = round(price * reno_pct, 2)

    # Closing costs — use state-specific if available
    closing_pct = state_ctx.get("closing_cost_pct", 0.025)
    transfer_tax = state_ctx.get("transfer_tax", 0)
    closing_costs = round(price * (closing_pct + transfer_tax), 2)

    # Total basis
    total_basis = round(price + renovation_cost + closing_costs, 2)

    score = 1.0
    if reno_pct > 0.20:
        score -= 0.2
    if reno_pct > 0.30:
        score -= 0.2

    return {
        "status": "pass",
        "score": round(max(0.0, score), 4),
        "purchase_price": price,
        "renovation_cost": renovation_cost,
        "closing_costs": closing_costs,
        "total_basis": total_basis,
    }


# ═══════════════════════════════════════════════════════════════
# STAGE 5 — FINANCE
# ═══════════════════════════════════════════════════════════════

def stage_finance(params: Dict) -> Dict:
    """Capital structure, debt terms, IRR, DSCR, cash-on-cash.

    Uses Phase 2 financial math (inlined here for self-containment).
    """
    price = params.get("purchase_price", 0)
    noi = params.get("noi", 0)
    state_ctx = params.get("_state_context", {})

    reno_pct = params.get("renovation_pct_of_price", 0.10)
    closing_pct = state_ctx.get("closing_cost_pct", 0.025) + state_ctx.get("transfer_tax", 0)
    total_basis = price * (1 + reno_pct + closing_pct)

    ltv = params.get("ltv_max", 0.70)
    loan_amount = round(total_basis * ltv, 2)
    equity = round(total_basis - loan_amount, 2)

    loan_rate = params.get("loan_rate", 0.065)
    amort_years = params.get("loan_amort_years", 30)
    monthly_rate = loan_rate / 12
    n_payments = amort_years * 12

    # Monthly debt service
    if monthly_rate > 0 and n_payments > 0:
        ds_monthly = loan_amount * (
            monthly_rate * (1 + monthly_rate) ** n_payments
        ) / ((1 + monthly_rate) ** n_payments - 1)
    else:
        ds_monthly = loan_amount / max(n_payments, 1)

    annual_ds = round(ds_monthly * 12, 2)
    dscr = round(noi / annual_ds, 4) if annual_ds > 0 else 0

    # Cash-on-cash return
    cf_after_debt = noi - annual_ds
    cash_on_cash = round(cf_after_debt / max(equity, 1), 4)

    # Simplified IRR (unlevered going-in)
    cap_rate = noi / price if price > 0 else 0

    # Hold period IRR estimate
    hold_years = params.get("hold_period_years", 7)
    exit_cap = params.get("exit_cap_rate", cap_rate + 0.005)
    rent_growth = params.get("rent_growth_annual", 0.02)

    # Project NOI growth
    cf_series = [-equity]
    current_noi = noi
    for y in range(1, hold_years + 1):
        current_noi *= (1 + rent_growth)
        annual_cf = current_noi - annual_ds
        cf_series.append(annual_cf)

    # Add exit proceeds in final year
    exit_noi = current_noi * (1 + rent_growth)
    exit_value = exit_noi / exit_cap if exit_cap > 0 else 0
    remaining_balance = loan_amount * 0.85
    exit_proceeds = exit_value - remaining_balance
    cf_series[-1] += exit_proceeds

    irr = _solve_irr_annual(cf_series)

    # Scoring
    target_irr = params.get("target_irr", 0.15)
    min_dscr = params.get("min_dscr", 1.25)

    score = 0.5
    if irr >= target_irr:
        score += 0.3
    elif irr >= target_irr * 0.75:
        score += 0.15

    if dscr >= min_dscr:
        score += 0.2
    elif dscr >= 1.0:
        score += 0.1
    else:
        score -= 0.2

    return {
        "status": "pass" if dscr >= 1.0 else "fail",
        "score": round(max(0.0, min(1.0, score)), 4),
        "total_basis": round(total_basis, 2),
        "loan_amount": loan_amount,
        "equity_required": equity,
        "annual_debt_service": annual_ds,
        "monthly_debt_service": round(ds_monthly, 2),
        "dscr": dscr,
        "cash_on_cash": cash_on_cash,
        "irr_estimate": round(irr, 4),
        "cap_rate": round(cap_rate, 6),
        "exit_cap_rate": exit_cap,
        "hold_period_years": hold_years,
    }


# ═══════════════════════════════════════════════════════════════
# STAGE 6 — RISK
# ═══════════════════════════════════════════════════════════════

def stage_risk(params: Dict, finance_result: Dict) -> Dict:
    """Risk scoring with scenario analysis.

    Runs downside scenarios: vacancy spike, rate increase, NOI decline.
    """
    noi = params.get("noi", 0)
    annual_ds = finance_result.get("annual_debt_service", 0)
    equity = finance_result.get("equity_required", 0)
    dscr = finance_result.get("dscr", 0)
    state_ctx = params.get("_state_context", {})

    scenarios = {}

    # Scenario 1: Vacancy spikes +10%
    vacancy_noi = noi * 0.90
    scenarios["vacancy_spike"] = {
        "noi": round(vacancy_noi, 2),
        "dscr": round(vacancy_noi / max(annual_ds, 1), 4),
        "cash_on_cash": round((vacancy_noi - annual_ds) / max(equity, 1), 4),
    }

    # Scenario 2: Rate increase +2%
    new_rate = params.get("loan_rate", 0.065) + 0.02
    loan_amount = finance_result.get("loan_amount", 0)
    new_ds = _simple_annual_ds(loan_amount, new_rate, 30)
    scenarios["rate_increase"] = {
        "new_rate": new_rate,
        "annual_ds": round(new_ds, 2),
        "dscr": round(noi / max(new_ds, 1), 4),
    }

    # Scenario 3: NOI decline 20%
    decline_noi = noi * 0.80
    scenarios["noi_decline_20pct"] = {
        "noi": round(decline_noi, 2),
        "dscr": round(decline_noi / max(annual_ds, 1), 4),
        "cash_on_cash": round((decline_noi - annual_ds) / max(equity, 1), 4),
    }

    # Gaming revenue risk (if applicable)
    if params.get("gaming_eligible") or params.get("_state_context", {}).get("gaming_legal"):
        gaming_nw = params.get("expected_gaming_net_win_monthly", 0) * 12
        # Use state-specific avg NTI if no explicit value
        if gaming_nw == 0 and state_ctx.get("gaming_avg_nti"):
            terminal_count = params.get("terminal_count", 5)
            gaming_nw = state_ctx["gaming_avg_nti"] * terminal_count * 12
        # Apply gaming tax rate from state config
        gaming_tax = state_ctx.get("gaming_tax_rate", 0.34)
        operator_share = 1.0 - gaming_tax
        # Operator split (location owner's share of post-tax revenue)
        location_share_pct = params.get("operator_split", 35) / 100.0
        gaming_income = gaming_nw * operator_share * location_share_pct

        noi_with_gaming = noi + gaming_income
        scenarios["gaming_upside"] = {
            "gross_gaming_revenue": round(gaming_nw, 2),
            "gaming_tax_rate": gaming_tax,
            "location_income_annual": round(gaming_income, 2),
            "noi_with_gaming": round(noi_with_gaming, 2),
            "dscr_with_gaming": round(noi_with_gaming / max(annual_ds, 1), 4),
        }

    # Property tax stress (state-aware)
    prop_tax = state_ctx.get("property_tax_rate", 0)
    if prop_tax > 0.015:
        # Simulate 10% property tax increase
        price = params.get("purchase_price", 0)
        tax_increase = price * prop_tax * 0.10
        stressed_noi = noi - tax_increase
        scenarios["property_tax_increase"] = {
            "tax_increase": round(tax_increase, 2),
            "stressed_noi": round(stressed_noi, 2),
            "dscr": round(stressed_noi / max(annual_ds, 1), 4),
        }

    # Aggregate risk score
    score = 0.7
    worst_dscr = min(
        scenarios["vacancy_spike"]["dscr"],
        scenarios["noi_decline_20pct"]["dscr"],
    )
    if worst_dscr >= 1.0:
        score += 0.2
    elif worst_dscr >= 0.8:
        score += 0.0
    else:
        score -= 0.3

    if scenarios["rate_increase"]["dscr"] < 1.0:
        score -= 0.1

    return {
        "status": "pass",
        "score": round(max(0.0, min(1.0, score)), 4),
        "scenarios": scenarios,
        "worst_case_dscr": round(worst_dscr, 4),
    }


# ═══════════════════════════════════════════════════════════════
# STAGE 7 — DECISION SCORING
# ═══════════════════════════════════════════════════════════════

def stage_decision(
    stage_scores: Dict[str, float],
    weights: Dict[str, float],
    finance_result: Dict,
    risk_result: Dict,
) -> Dict:
    """Compute weighted decision score → GO / HOLD / NO-GO.

    stage_scores maps dimension → score (0.0–1.0):
      financial_return, market_strength, feasibility,
      cost_risk, gaming_upside, debt_coverage

    weights maps the same dimensions → float (should sum to ~1.0).
    """
    weighted_sum = 0.0
    weight_total = 0.0
    score_detail = {}

    for dim, weight in weights.items():
        s = stage_scores.get(dim, 0.5)
        weighted_sum += s * weight
        weight_total += weight
        score_detail[dim] = {"score": s, "weight": weight, "contribution": round(s * weight, 4)}

    composite = round(weighted_sum / max(weight_total, 0.01), 4)

    # Decision thresholds
    if composite >= 0.70:
        decision = "GO"
    elif composite >= 0.50:
        decision = "HOLD"
    else:
        decision = "NO_GO"

    # Build rationale
    rationale_parts = []
    irr = finance_result.get("irr_estimate", 0)
    dscr = finance_result.get("dscr", 0)
    worst_dscr = risk_result.get("worst_case_dscr", 0)

    if decision == "GO":
        rationale_parts.append(f"Deal scores {composite:.0%} — above GO threshold")
    elif decision == "HOLD":
        rationale_parts.append(f"Deal scores {composite:.0%} — review required")
    else:
        rationale_parts.append(f"Deal scores {composite:.0%} — below minimum threshold")

    rationale_parts.append(f"Estimated IRR: {irr:.1%}")
    rationale_parts.append(f"DSCR: {dscr:.2f} (worst case: {worst_dscr:.2f})")

    # Flag concerns
    if dscr < 1.25:
        rationale_parts.append("⚠ DSCR below 1.25x minimum")
    if worst_dscr < 1.0:
        rationale_parts.append("⚠ Worst-case DSCR below 1.0x — debt coverage at risk")
    if irr < 0.10:
        rationale_parts.append("⚠ IRR below 10% minimum")

    return {
        "composite_score": composite,
        "decision": decision,
        "rationale": "; ".join(rationale_parts),
        "score_detail": score_detail,
    }


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _solve_irr_annual(cash_flows: list, max_iter: int = 100) -> float:
    """Newton's method IRR solver for annual cash flows."""
    if not cash_flows or all(cf == 0 for cf in cash_flows):
        return 0.0

    rate = 0.10
    for _ in range(max_iter):
        npv = sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))
        dnpv = sum(-t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cash_flows))
        if abs(dnpv) < 1e-12:
            break
        new_rate = rate - npv / dnpv
        if abs(new_rate - rate) < 1e-8:
            rate = new_rate
            break
        rate = max(-0.99, min(10.0, new_rate))

    return rate


def _simple_annual_ds(loan_amount: float, annual_rate: float, amort_years: int) -> float:
    """Simple annual debt service calculation."""
    monthly_rate = annual_rate / 12
    n = amort_years * 12
    if monthly_rate <= 0:
        return loan_amount / max(n, 1) * 12
    payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** n) / (
        (1 + monthly_rate) ** n - 1
    )
    return payment * 12
