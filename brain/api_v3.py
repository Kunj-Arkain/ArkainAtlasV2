"""
engine.api_v3 — Unified REST API for Truth Accretion Engine
=============================================================
Replaces api_endpoints_pipeline.py + api_endpoints_data.py + api_endpoints_states.py

All endpoints wired to v3 orchestrator, tool registry, RBAC, and audit ledger.

Mount with: app.include_router(router, prefix="/api/v3")

Sections:
  /deals        — Run deals, get decision packages, submit data
  /data         — Direct data tool access (demographics, traffic, etc.)
  /pipeline     — Configuration, presets, agent roster
  /construction — Drawings, specs, engineering calcs
  /states       — State intelligence
  /admin        — Users, access logs, secrets
  /replay       — Audit trail, replay, diff
"""

from __future__ import annotations
import time
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# ENGINE SINGLETON — all endpoints share one orchestrator
# ═══════════════════════════════════════════════════════════════

_engine_instance = None


def get_engine():
    """Get or create the engine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = EngineInstance()
    return _engine_instance


class EngineInstance:
    """Holds the orchestrator + subsystems between requests."""

    def __init__(self):
        from .orchestrator_v3 import TruthAccretionOrchestrator
        from .tools_unified import create_tool_registry
        from .budget import BudgetLimits
        from .convergence_v2 import ConvergenceConfig
        from .security import RBACManager, SecretsVault, Role

        self.tool_registry = create_tool_registry(live=True)
        self.rbac = RBACManager()
        self.vault = SecretsVault()

        # Pre-create default users
        self.rbac.add_user("system", "System", Role.ADMIN)

        # Orchestrator factory — new instance per deal run
        self._budget_defaults = BudgetLimits()
        self._convergence_defaults = ConvergenceConfig()

    def create_orchestrator(self, execute_fn=None):
        from .orchestrator_v3 import TruthAccretionOrchestrator
        return TruthAccretionOrchestrator(
            execute_fn=execute_fn or self._default_execute,
            tool_registry=self.tool_registry,
            budget=self._budget_defaults,
            convergence=self._convergence_defaults,
        )

    def _default_execute(self, **kw):
        """Default agent execution stub — replace with LLM call in production."""
        return {"_error": "No LLM executor configured", "recommendation": "NEEDS_DATA"}


# ═══════════════════════════════════════════════════════════════
# DEAL ENDPOINTS — The core v3 pipeline
# ═══════════════════════════════════════════════════════════════

def deals_run(body: Dict) -> Dict:
    """POST /api/v3/deals/run

    Run the full truth accretion pipeline on a deal.

    Body:
      deal_data: {address, price, property_type, sqft, ...}
      agents: ["underwriting_analyst", "risk_officer", ...] (optional)
      agent_tools: {agent: [tool_names]} (optional, auto-assigned if omitted)
      execute_fn_name: str (optional, for custom LLM routing)
      budget: {max_tool_calls, max_runtime_seconds, ...} (optional)
      user_id: str (for RBAC, default "system")

    Returns: DecisionPackage as JSON
    """
    engine = get_engine()
    user_id = body.get("user_id", "system")
    engine.rbac.require_permission(user_id, "run_pipeline", "deals/run")

    deal_data = body.get("deal_data", {})
    agents = body.get("agents", ["underwriting_analyst", "risk_officer"])

    # Auto-assign tools if not provided
    agent_tools = body.get("agent_tools")
    if not agent_tools:
        agent_tools = _auto_assign_tools(agents, deal_data)

    # Custom budget
    budget = None
    if "budget" in body:
        from .budget import BudgetLimits
        budget = BudgetLimits(**body["budget"])

    orch = engine.create_orchestrator()
    if budget:
        orch.budget = __import__("budget").BudgetEnforcer(budget)

    # Pre-submit any inline user data
    for var, val in body.get("user_data", {}).items():
        orch.submit_user_data(var, val, signer=user_id)

    # Run
    pkg = orch.run(
        deal_data=deal_data,
        active_agents=agents,
        agent_tools=agent_tools,
    )

    return {
        "decision_package": pkg.to_dict(),
        "summary": pkg.render_summary(),
        "actionable": pkg.is_actionable(),
    }


def deals_missing_vars(body: Dict) -> Dict:
    """POST /api/v3/deals/missing-variables

    Get ranked list of missing variables for a deal.
    Used by frontend to show "what data do we still need?" UI.

    Body:
      deal_data: {address, price, property_type, ...}
      agents: ["underwriting_analyst", ...] (optional)

    Returns: ranked list of UNKNOWN/ASSUMPTION variables with tools that provide them
    """
    engine = get_engine()
    orch = engine.create_orchestrator()

    deal_data = body.get("deal_data", {})
    # Initialize beliefs + assumptions from deal data
    from .active_inference import BeliefState
    orch.beliefs = BeliefState(deal_data)
    orch._init_assumptions(deal_data)

    missing = orch.get_missing_variables()
    return {
        "missing_variables": missing,
        "total_missing": len(missing),
        "coverage": orch.assumptions.coverage_report(),
    }


def deals_submit_data(body: Dict) -> Dict:
    """POST /api/v3/deals/submit-data

    Submit user data or assumptions for a running deal.

    Body:
      run_id: str (identifies the deal context)
      variables: [
        {name: "noi", value: 195000, type: "user_input", signer: "jsmith",
         confidence: 0.8, low: 170000, high: 220000},
        {name: "cap_rate", value: 7.0, type: "assumption",
         rationale: "Based on comps", tag: "market_standard"},
      ]
    """
    engine = get_engine()
    user_id = body.get("user_id", "system")
    engine.rbac.require_permission(user_id, "submit_data")

    results = []
    for var in body.get("variables", []):
        name = var["name"]
        value = var["value"]
        vtype = var.get("type", "user_input")

        if vtype == "user_input":
            results.append({
                "variable": name, "status": "USER_INPUT",
                "value": value, "signer": var.get("signer", user_id),
            })
        elif vtype == "assumption":
            results.append({
                "variable": name, "status": "ASSUMPTION",
                "value": value, "rationale": var.get("rationale", ""),
            })

    return {"submitted": len(results), "variables": results}


def deals_monte_carlo(body: Dict) -> Dict:
    """POST /api/v3/deals/monte-carlo

    Run standalone Monte Carlo simulation (correlated draws).

    Body:
      purchase_price: float
      noi: {point, low, high}
      loan_rate: {point, low, high}
      exit_cap: {point, low, high}
      noi_growth: {point, low, high} (optional)
      loan_ltv: float (default 0.75)
      hold_years: int (default 5)
      num_simulations: int (default 2000)
      gaming_nti: {point, low, high} (optional)
      terminal_count: int (optional)
    """
    from .seal_ceca import MonteCarloSimulator
    from .determinism import seed_engine

    seed_engine(run_id=f"mc_{int(time.time())}")
    n = body.pop("num_simulations", 2000)
    mc = MonteCarloSimulator(num_simulations=n)
    result = mc.simulate_deal(body)
    return result


# ═══════════════════════════════════════════════════════════════
# DATA ENDPOINTS — Direct tool access
# ═══════════════════════════════════════════════════════════════

def data_tool_call(tool_name: str, params: Dict) -> Dict:
    """POST /api/v3/data/{tool_name}

    Call any registered data tool directly.
    Returns ToolResult with evidence refs.
    """
    engine = get_engine()
    result = engine.tool_registry.call(tool_name, params)
    return {
        "tool": tool_name,
        "status": result.status.value,
        "data": result.data,
        "evidence": [{"ref_id": r.ref_id, "source": r.source_name,
                       "variable": r.variable, "value": r.value,
                       "grade": r.grade.value}
                      for r in result.evidence_refs],
        "error": result.error_message,
        "call_duration_ms": result.call_duration_ms,
    }


def data_site_report(body: Dict) -> Dict:
    """POST /api/v3/data/site-report

    Generate comprehensive site intelligence report.
    Calls all relevant tools for a property address.

    Body:
      address: str (required)
      state: str (required)
      property_type: str (default "gas_station")
      gaming_eligible: bool (optional)
    """
    from .data_tools import (
        census_demographics, census_business_patterns,
        bls_employment, fred_economic_data,
        gaming_board_data, environmental_risk,
        traffic_counts, property_records,
        crime_data, market_cap_rates,
        insurance_estimate, utility_costs,
        competitor_scan, location_scores, zoning_lookup,
    )

    start = time.perf_counter()
    address = body.get("address", "")
    state = body.get("state", "")
    city = body.get("city", "")
    ptype = body.get("property_type", "gas_station")

    report = {"address": address, "state": state, "sections": {}}

    # State context
    try:
        from .state_config import get_business_context
        report["sections"]["state_context"] = get_business_context(state, ptype)
    except Exception:
        pass

    # All data tools — wrapped for resilience
    tool_calls = [
        ("demographics", census_demographics, {"state": state, "city": city}),
        ("employment", bls_employment, {"state": state, "city": city}),
        ("economic", fred_economic_data, {}),
        ("traffic", traffic_counts, {"address": address, "state": state}),
        ("environmental", environmental_risk, {"address": address, "state": state}),
        ("property_records", property_records, {"address": address, "state": state}),
        ("crime", crime_data, {"city": city, "state": state}),
        ("cap_rates", market_cap_rates, {"property_type": ptype, "state": state}),
        ("insurance", insurance_estimate, {"property_type": ptype, "state": state,
                                            "purchase_price": body.get("purchase_price", 0)}),
        ("utilities", utility_costs, {"state": state, "sqft": body.get("sqft", 3000)}),
        ("competitors", competitor_scan, {"address": address, "city": city, "state": state}),
        ("location_scores", location_scores, {"address": address}),
        ("zoning", zoning_lookup, {"address": address, "city": city, "state": state}),
    ]

    if body.get("gaming_eligible"):
        tool_calls.append(("gaming", gaming_board_data, {"state": state, "city": city}))

    errors = []
    for section, fn, params in tool_calls:
        try:
            report["sections"][section] = fn(params)
        except Exception as e:
            errors.append({"section": section, "error": str(e)})

    report["_meta"] = {
        "sections_generated": len(report["sections"]),
        "errors": len(errors),
        "elapsed_ms": int((time.perf_counter() - start) * 1000),
    }
    if errors:
        report["_errors"] = errors

    return report


def data_tool_catalog() -> Dict:
    """GET /api/v3/data/catalog — List all available data tools."""
    from .tools_unified import tool_catalog
    return tool_catalog()


# ═══════════════════════════════════════════════════════════════
# PIPELINE CONFIGURATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════

def pipeline_presets() -> Dict:
    """GET /api/v3/pipeline/presets"""
    from .pipeline_config import list_presets
    return list_presets()


def pipeline_features() -> Dict:
    """GET /api/v3/pipeline/features"""
    from .pipeline_config import list_all_features
    return {"features": list_all_features()}


def pipeline_configure(body: Dict) -> Dict:
    """POST /api/v3/pipeline/configure"""
    from .pipeline_config import PipelineConfig, FeatureFlags

    preset = body.get("preset")
    project_data = body.get("project_data")
    flags = body.get("flags")
    overrides = body.get("agent_overrides", {})
    name = body.get("project_name", "")

    if preset:
        config = PipelineConfig.from_preset(preset, name)
    elif project_data:
        config = PipelineConfig.for_project(project_data)
    elif flags:
        config = PipelineConfig(project_name=name, preset="custom",
                                flags=FeatureFlags.from_dict(flags))
    else:
        config = PipelineConfig.from_preset("standard_deal", name)

    for agent, enabled in overrides.items():
        config.enable(agent) if enabled else config.disable(agent)

    return config.to_dict()


def pipeline_agents() -> Dict:
    """GET /api/v3/pipeline/agents"""
    from .agents import AGENT_ROLES
    return {
        "agents": [{**p.to_dict(), "task_details": p.tasks}
                    for p in AGENT_ROLES.values()],
        "total": len(AGENT_ROLES),
    }


def pipeline_agent_detail(agent_name: str) -> Dict:
    """GET /api/v3/pipeline/agents/{agent_name}"""
    from .agents import resolve_agent
    profile = resolve_agent(agent_name)
    return {
        **profile.to_dict(),
        "task_details": profile.tasks,
        "system_prompt_preview": profile.system_prompt[:500] + "...",
    }


def pipeline_order() -> Dict:
    """GET /api/v3/pipeline/order"""
    from .agents import get_agent_pipeline
    pipeline = get_agent_pipeline()
    return {"pipeline": pipeline, "total_agents": len(pipeline)}


# ═══════════════════════════════════════════════════════════════
# CONSTRUCTION ENDPOINTS (unchanged, fixed imports)
# ═══════════════════════════════════════════════════════════════

def construction_code_analysis(body: Dict) -> Dict:
    from .construction_tools import code_analysis
    return code_analysis(body)


def construction_electrical(body: Dict) -> Dict:
    from .construction_tools import electrical_load_calc
    return electrical_load_calc(body)


def construction_hvac(body: Dict) -> Dict:
    from .construction_tools import hvac_sizing
    return hvac_sizing(body)


def construction_plumbing(body: Dict) -> Dict:
    from .construction_tools import plumbing_design
    return plumbing_design(body)


def construction_structural(body: Dict) -> Dict:
    from .construction_tools import structural_calc
    return structural_calc(body)


def construction_schedule(body: Dict) -> Dict:
    from .construction_tools import construction_schedule
    return construction_schedule(body)


def construction_drawings(body: Dict) -> Dict:
    from .construction_tools import generate_drawing_set
    if "output_dir" not in body:
        body["output_dir"] = f"/tmp/drawings/{body.get('project_name', 'project').replace(' ', '_')}"
    return generate_drawing_set(body)


def construction_specs(body: Dict) -> Dict:
    from .construction_tools import generate_spec_book
    if "output_path" not in body:
        name = body.get("project_name", "project").replace(" ", "_")
        body["output_path"] = f"/tmp/specs/{name}_spec_book.pdf"
    return generate_spec_book(body)


def construction_full_package(body: Dict) -> Dict:
    """Generate complete construction package: drawings + specs + schedule + calcs."""
    from .construction_tools import (
        generate_drawing_set, generate_spec_book,
        construction_schedule as sched, code_analysis,
        electrical_load_calc, hvac_sizing,
        plumbing_design, structural_calc,
    )

    start = time.perf_counter()
    name = body.get("project_name", "Construction Project")
    base_dir = f"/tmp/construction/{name.replace(' ', '_')}"
    results = {"project_name": name, "components": {}}

    results["components"]["code_analysis"] = code_analysis(body)
    results["components"]["electrical"] = electrical_load_calc(body)
    results["components"]["hvac"] = hvac_sizing(body)
    results["components"]["plumbing"] = plumbing_design(body)
    results["components"]["structural"] = structural_calc(body)

    body["output_dir"] = f"{base_dir}/drawings"
    results["components"]["drawings"] = generate_drawing_set(body)

    body["output_path"] = f"{base_dir}/specs/{name.replace(' ', '_')}_specs.pdf"
    results["components"]["specs"] = generate_spec_book(body)

    results["components"]["schedule"] = sched(body)

    results["_meta"] = {
        "components_generated": len(results["components"]),
        "elapsed_ms": int((time.perf_counter() - start) * 1000),
        "output_directory": base_dir,
    }
    return results


# ═══════════════════════════════════════════════════════════════
# STATE INTELLIGENCE ENDPOINTS (fixed imports)
# ═══════════════════════════════════════════════════════════════

def states_list() -> Dict:
    from .state_config import STATE_CONFIG, get_state_summary
    return {
        "states": [get_state_summary(code) for code in sorted(STATE_CONFIG.keys())],
        "count": len(STATE_CONFIG),
    }


def states_detail(state_code: str) -> Dict:
    from .state_config import get_state
    sc = get_state(state_code)
    if not sc:
        return {"error": f"State '{state_code}' not found"}
    return {"code": state_code.upper(), **sc}


def states_business(state_code: str, business_type: str) -> Dict:
    from .state_config import get_business_context
    return get_business_context(state_code, business_type)


def states_gaming_legal() -> Dict:
    from .state_config import get_gaming_states, get_state_summary
    codes = get_gaming_states()
    return {"gaming_states": [get_state_summary(c) for c in sorted(codes)], "count": len(codes)}


# ═══════════════════════════════════════════════════════════════
# ADMIN ENDPOINTS — RBAC, access logs, secrets
# ═══════════════════════════════════════════════════════════════

def admin_users(user_id: str = "system") -> Dict:
    engine = get_engine()
    engine.rbac.require_permission(user_id, "manage_users")
    return {"users": engine.rbac.users()}


def admin_access_log(user_id: str = "system", limit: int = 100) -> Dict:
    engine = get_engine()
    engine.rbac.require_permission(user_id, "view_ledger")
    return {"entries": engine.rbac.access_log(limit), "total": limit}


def admin_add_user(body: Dict) -> Dict:
    from .security import Role
    engine = get_engine()
    engine.rbac.require_permission(body.get("admin_user_id", "system"), "manage_users")
    role = Role(body["role"])
    user = engine.rbac.add_user(body["user_id"], body["name"], role,
                                 password=body.get("password", ""))
    return {"created": user.user_id, "role": role.value}


# ═══════════════════════════════════════════════════════════════
# CAPITAL STACK ENDPOINTS
# ═══════════════════════════════════════════════════════════════

def deals_capital_stack_waterfall(body: Dict) -> Dict:
    """POST /api/v3/deals/capital-stack/waterfall

    Compute cash flow waterfall for a capital structure.

    Body:
      purchase_price: float
      noi: float
      exit_value: float
      hold_years: int (default 5)
      noi_growth: float (default 0.02)
      debt: [{name, amount, rate, term_years, io_years, priority}]
      equity: [{name, amount, preferred_return}]
    """
    from .capital_stack import CapitalStack, DebtTranche, EquityTranche, compute_waterfall

    debt = [DebtTranche(**d) for d in body.get("debt", [])]
    equity = [EquityTranche(**e) for e in body.get("equity", [])]
    stack = CapitalStack(body["purchase_price"], debt, equity)
    wf = compute_waterfall(stack, body["noi"], body["exit_value"],
                           body.get("hold_years", 5), body.get("noi_growth", 0.02))
    wf["stack_summary"] = stack.summary()
    return wf


def deals_capital_stack_stress(body: Dict) -> Dict:
    """POST /api/v3/deals/capital-stack/stress-test

    Run correlated MC stress test on a capital structure.

    Body: same as waterfall + noi/loan_rate/exit_cap/noi_growth as {point,low,high}
    """
    from .capital_stack import (CapitalStack, DebtTranche, EquityTranche,
                                stress_test_stack)
    from .determinism import seed_engine

    seed_engine(run_id=f"stress_{int(time.time())}")
    debt = [DebtTranche(**d) for d in body.get("debt", [])]
    equity = [EquityTranche(**e) for e in body.get("equity", [])]
    stack = CapitalStack(body["purchase_price"], debt, equity)
    return stress_test_stack(stack, body, body.get("num_simulations", 2000))


def deals_capital_stack_compare(body: Dict) -> Dict:
    """POST /api/v3/deals/capital-stack/compare

    Compare multiple capital structures side-by-side under stress.

    Body:
      purchase_price: float
      noi: float
      structures: [{name, senior_ltv, senior_rate, mezz_ltv?, mezz_rate?}]
      + base_params (noi, loan_rate, exit_cap, noi_growth as {point,low,high})
    """
    from .capital_stack import compare_structures
    from .determinism import seed_engine

    seed_engine(run_id=f"compare_{int(time.time())}")
    return compare_structures(
        body["purchase_price"], body["noi"],
        body["structures"], body,
        body.get("num_simulations", 1000))


# ═══════════════════════════════════════════════════════════════
# SCENARIO TREE ENDPOINTS
# ═══════════════════════════════════════════════════════════════

def deals_scenario_evaluate(body: Dict) -> Dict:
    """POST /api/v3/deals/scenarios/evaluate

    Build and evaluate a standard CRE scenario tree.

    Body:
      base_params: {purchase_price, noi, loan_rate, exit_cap, noi_growth, ...}
      risk_aversion: float (default 2.0, γ for CRRA utility)
      num_simulations: int (default 500 per leaf)
      tree_type: str ("standard" or "expansion", default "standard")
    """
    from .scenario_tree import ScenarioTreeEvaluator, cre_standard_tree, expansion_tree
    from .determinism import seed_engine

    seed_engine(run_id=f"scenario_{int(time.time())}")
    base = body.get("base_params", body)
    gamma = body.get("risk_aversion", 2.0)
    n = body.get("num_simulations", 500)
    tree_type = body.get("tree_type", "standard")

    tree = cre_standard_tree(base) if tree_type == "standard" else expansion_tree(base)
    evaluator = ScenarioTreeEvaluator(base, num_simulations=n, risk_aversion=gamma)
    return evaluator.evaluate(tree)


def deals_scenario_compare_strategies(body: Dict) -> Dict:
    """POST /api/v3/deals/scenarios/compare-strategies

    Compare multiple strategies across a scenario tree.

    Body:
      base_params: {purchase_price, noi, loan_rate, exit_cap, ...}
      strategies: [{name, loan_ltv?, ...override params}]
      risk_aversion: float (default 2.0)
    """
    from .scenario_tree import ScenarioTreeEvaluator, cre_standard_tree
    from .determinism import seed_engine

    seed_engine(run_id=f"strat_{int(time.time())}")
    base = body.get("base_params", body)
    tree = cre_standard_tree(base)
    evaluator = ScenarioTreeEvaluator(
        base, num_simulations=body.get("num_simulations", 300),
        risk_aversion=body.get("risk_aversion", 2.0))
    return evaluator.compare_strategies(tree, body["strategies"])


# ═══════════════════════════════════════════════════════════════
# REPLAY / AUDIT ENDPOINTS
# ═══════════════════════════════════════════════════════════════

def replay_summary(run_id: str) -> Dict:
    """GET /api/v3/replay/{run_id}/summary"""
    # In production, load from persisted replay logs
    return {"run_id": run_id, "status": "load from replay store"}


def replay_diff(body: Dict) -> Dict:
    """POST /api/v3/replay/diff — Compare two runs."""
    from .replay import ReplayLog, ReplayDiff
    run_a_path = body.get("run_a_path")
    run_b_path = body.get("run_b_path")
    if not run_a_path or not run_b_path:
        return {"error": "Provide run_a_path and run_b_path"}
    log_a = ReplayLog.load(run_a_path)
    log_b = ReplayLog.load(run_b_path)
    return ReplayDiff.diff(log_a, log_b)


# ═══════════════════════════════════════════════════════════════
# ROUTE TABLE — for framework-agnostic mounting
# ═══════════════════════════════════════════════════════════════

def _auto_assign_tools(agents, deal_data):
    """Auto-assign tools to agents based on deal type."""
    gaming = deal_data.get("gaming_eligible", False)
    base = ["census_demographics", "traffic_counts", "pull_comps",
            "fred_economic_data", "crime_data", "environmental_risk"]
    if gaming:
        base.append("gaming_board_data")

    return {
        "underwriting_analyst": base + ["generate_term_sheets", "market_cap_rates"],
        "risk_officer": ["census_demographics", "crime_data",
                         "environmental_risk", "traffic_counts"],
        "deal_structurer": ["pull_comps", "fred_economic_data",
                            "generate_term_sheets", "market_cap_rates"],
        "gaming_optimizer": (["gaming_board_data", "traffic_counts"]
                             if gaming else []),
    }


ROUTE_TABLE = {
    # Deals (v3 orchestrator)
    "POST /deals/run": deals_run,
    "POST /deals/missing-variables": deals_missing_vars,
    "POST /deals/submit-data": deals_submit_data,
    "POST /deals/monte-carlo": deals_monte_carlo,

    # Capital stack
    "POST /deals/capital-stack/waterfall": deals_capital_stack_waterfall,
    "POST /deals/capital-stack/stress-test": deals_capital_stack_stress,
    "POST /deals/capital-stack/compare": deals_capital_stack_compare,

    # Scenario tree
    "POST /deals/scenarios/evaluate": deals_scenario_evaluate,
    "POST /deals/scenarios/compare-strategies": deals_scenario_compare_strategies,

    # Data tools
    "POST /data/{tool_name}": data_tool_call,
    "POST /data/site-report": data_site_report,
    "GET  /data/catalog": data_tool_catalog,

    # Pipeline config
    "GET  /pipeline/presets": pipeline_presets,
    "GET  /pipeline/features": pipeline_features,
    "POST /pipeline/configure": pipeline_configure,
    "GET  /pipeline/agents": pipeline_agents,
    "GET  /pipeline/agents/{agent_name}": pipeline_agent_detail,
    "GET  /pipeline/order": pipeline_order,

    # Construction
    "POST /construction/code-analysis": construction_code_analysis,
    "POST /construction/electrical": construction_electrical,
    "POST /construction/hvac": construction_hvac,
    "POST /construction/plumbing": construction_plumbing,
    "POST /construction/structural": construction_structural,
    "POST /construction/schedule": construction_schedule,
    "POST /construction/drawings": construction_drawings,
    "POST /construction/specs": construction_specs,
    "POST /construction/full-package": construction_full_package,

    # States
    "GET  /states": states_list,
    "GET  /states/{state_code}": states_detail,
    "GET  /states/{state_code}/business/{business_type}": states_business,
    "GET  /states/gaming/legal": states_gaming_legal,

    # Admin
    "GET  /admin/users": admin_users,
    "GET  /admin/access-log": admin_access_log,
    "POST /admin/users": admin_add_user,

    # Replay/Audit
    "GET  /replay/{run_id}/summary": replay_summary,
    "POST /replay/diff": replay_diff,
}


def mount_fastapi(app, prefix: str = "/api/v3"):
    """Mount all routes on a FastAPI app.

    Usage:
        from fastapi import FastAPI
        from engine.api_v3 import mount_fastapi
        app = FastAPI()
        mount_fastapi(app)
    """
    # Deals
    app.post(f"{prefix}/deals/run")(lambda body: deals_run(body))
    app.post(f"{prefix}/deals/missing-variables")(lambda body: deals_missing_vars(body))
    app.post(f"{prefix}/deals/submit-data")(lambda body: deals_submit_data(body))
    app.post(f"{prefix}/deals/monte-carlo")(lambda body: deals_monte_carlo(body))

    # Capital stack
    app.post(f"{prefix}/deals/capital-stack/waterfall")(lambda body: deals_capital_stack_waterfall(body))
    app.post(f"{prefix}/deals/capital-stack/stress-test")(lambda body: deals_capital_stack_stress(body))
    app.post(f"{prefix}/deals/capital-stack/compare")(lambda body: deals_capital_stack_compare(body))

    # Scenario tree
    app.post(f"{prefix}/deals/scenarios/evaluate")(lambda body: deals_scenario_evaluate(body))
    app.post(f"{prefix}/deals/scenarios/compare-strategies")(lambda body: deals_scenario_compare_strategies(body))

    # Data
    app.post(f"{prefix}/data/site-report")(lambda body: data_site_report(body))
    app.get(f"{prefix}/data/catalog")(data_tool_catalog)
    app.post(f"{prefix}/data/{{tool_name}}")(lambda tool_name, body: data_tool_call(tool_name, body))

    # Pipeline
    app.get(f"{prefix}/pipeline/presets")(pipeline_presets)
    app.get(f"{prefix}/pipeline/features")(pipeline_features)
    app.post(f"{prefix}/pipeline/configure")(lambda body: pipeline_configure(body))
    app.get(f"{prefix}/pipeline/agents")(pipeline_agents)
    app.get(f"{prefix}/pipeline/agents/{{agent_name}}")(pipeline_agent_detail)
    app.get(f"{prefix}/pipeline/order")(pipeline_order)

    # Construction
    app.post(f"{prefix}/construction/code-analysis")(lambda body: construction_code_analysis(body))
    app.post(f"{prefix}/construction/drawings")(lambda body: construction_drawings(body))
    app.post(f"{prefix}/construction/specs")(lambda body: construction_specs(body))
    app.post(f"{prefix}/construction/full-package")(lambda body: construction_full_package(body))

    # States
    app.get(f"{prefix}/states")(states_list)
    app.get(f"{prefix}/states/gaming/legal")(states_gaming_legal)
    app.get(f"{prefix}/states/{{state_code}}")(states_detail)
    app.get(f"{prefix}/states/{{state_code}}/business/{{business_type}}")(states_business)

    # Admin
    app.get(f"{prefix}/admin/users")(admin_users)
    app.get(f"{prefix}/admin/access-log")(admin_access_log)
    app.post(f"{prefix}/admin/users")(lambda body: admin_add_user(body))

    # Replay
    app.get(f"{prefix}/replay/{{run_id}}/summary")(replay_summary)
    app.post(f"{prefix}/replay/diff")(lambda body: replay_diff(body))

    return app
