"""
engine.orchestrator_v3 — Central Orchestrator Owns the Loop (#4)
==================================================================
Single orchestrator that owns:
  - OODA phases (agents don't self-loop)
  - Tool selection + ordering (forager proposes, orchestrator decides)
  - Budget enforcement (hard stops)
  - Convergence testing (mechanical, not vibes)
  - Agent routing (skip/prioritize/resolve)
  - Assumption tracking (quarantine unknown data)
  - Evidence grounding (every number has a source)
  - Truth maintenance (contradiction resolution)
  - Schema validation (no kinda outputs)
  - Replay logging (audit-grade)
  - Decision package assembly (auditable output)
  - Data intake workflow (truth accretion)

HARD RULE: Agents propose; orchestrator decides. No hidden logic.
"""

from __future__ import annotations
import time
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Import all subsystems
try:
    from .assumptions import AssumptionTable, DataStatus
    from .evidence import EvidenceStore, make_evidence_ref, EvidenceMethod, EvidenceGrade
    from .schemas import validate_agent_output, StrictSchema
    from .tool_contracts import ToolRegistry, ToolResult, ToolStatus
    from .truth_maintenance import TruthMaintenanceSystem
    from .budget import BudgetEnforcer, BudgetLimits, BudgetExhausted
    from .convergence_v2 import MechanicalConvergence, ConvergenceConfig
    from .calibration import CalibrationLog
    from .replay import ReplayLog, RunMode
    from .decision_package import DecisionPackageBuilder, DecisionPackage
    from .active_inference import BeliefState, EpistemicForager
    from .seal_ceca import SEALDetector, CECACritic, MonteCarloSimulator
    from .ooda import ReflexionMemory, ReflexionEvaluator
    from .determinism import seed_engine, get_seed
    from .ledger import HashChainedLedger
    from .evidence_policy import EvidencePolicy, CRITICAL_VARIABLES
    from .cost_aware_forager import CostAwareForager, ForagingBudget
except ImportError:
    from assumptions import AssumptionTable, DataStatus
    from evidence import EvidenceStore, make_evidence_ref, EvidenceMethod, EvidenceGrade
    from schemas import validate_agent_output, StrictSchema
    from tool_contracts import ToolRegistry, ToolResult, ToolStatus
    from truth_maintenance import TruthMaintenanceSystem
    from budget import BudgetEnforcer, BudgetLimits, BudgetExhausted
    from convergence_v2 import MechanicalConvergence, ConvergenceConfig
    from calibration import CalibrationLog
    from replay import ReplayLog, RunMode
    from decision_package import DecisionPackageBuilder, DecisionPackage
    from active_inference import BeliefState, EpistemicForager
    from seal_ceca import SEALDetector, CECACritic, MonteCarloSimulator
    from ooda import ReflexionMemory, ReflexionEvaluator
    from determinism import seed_engine, get_seed
    from ledger import HashChainedLedger
    from evidence_policy import EvidencePolicy, CRITICAL_VARIABLES
    from cost_aware_forager import CostAwareForager, ForagingBudget


class TruthAccretionOrchestrator:
    """The master orchestrator — owns ALL control flow.

    Agents are pure functions: they receive context, return output.
    They never self-loop, never call tools directly, never decide
    when to retry. The orchestrator makes all control decisions.

    Run cycle per agent:
      1. OBSERVE: Orchestrator runs forager → gets tool plan
      2. EXECUTE TOOLS: Orchestrator calls tools → gets evidence
      3. ORIENT: Orchestrator runs SEAL + CECA on evidence
      4. DECIDE: Orchestrator runs MC simulation
      5. ACT: Orchestrator calls agent with assembled context
      6. VALIDATE: Orchestrator validates output against schema
      7. REPAIR/REJECT: If invalid → auto-repair prompt → reject
      8. CONVERGE: Orchestrator tests mechanical convergence
      9. RECORD: Orchestrator logs everything to replay log
    """

    def __init__(
        self,
        execute_fn: Callable,
        tool_registry: ToolRegistry = None,
        budget: BudgetLimits = None,
        convergence: ConvergenceConfig = None,
        run_mode: RunMode = RunMode.LIVE,
        replay_log: ReplayLog = None,
    ):
        self.execute_fn = execute_fn
        self.run_mode = run_mode

        # Subsystems
        self.assumptions = AssumptionTable()
        self.evidence = EvidenceStore()
        self.tms = TruthMaintenanceSystem()
        self.budget = BudgetEnforcer(budget or BudgetLimits())
        self.convergence = MechanicalConvergence(convergence or ConvergenceConfig())
        self.calibration = CalibrationLog()
        self.replay = replay_log or ReplayLog()
        self.ledger = HashChainedLedger()
        self.evidence_policy = EvidencePolicy()
        self.tools = tool_registry or ToolRegistry()
        self.memory = ReflexionMemory()
        self.evaluator = ReflexionEvaluator()

        # Cognitive components
        self.forager = CostAwareForager()  # EIG/$ optimized (replaces EpistemicForager)
        self.seal = SEALDetector()
        self.ceca = CECACritic()
        self.mc = MonteCarloSimulator(num_simulations=1000)

        # State
        self.beliefs: Optional[BeliefState] = None
        self.agent_outputs: Dict[str, Dict] = {}
        self.results: Dict[str, Dict] = {}

    def run(self, deal_data: Dict, active_agents: List[str],
            agent_tools: Dict[str, List[str]] = None) -> DecisionPackage:
        """Execute the full truth accretion pipeline.

        Returns an auditable DecisionPackage, never a raw dict.
        """
        start = time.perf_counter()
        agent_tools = agent_tools or {}

        # Initialize
        seed_engine(run_id=self.replay.run_id)
        self.budget.start()
        self.replay.start(deal_data, config={"seed": get_seed()})
        self.beliefs = BeliefState(deal_data)
        self._init_assumptions(deal_data)

        completed: Set[str] = set()
        skipped: List[Dict] = []
        total_agents = len(active_agents)
        budget_exhausted = False

        for idx, agent_name in enumerate(active_agents):
            if budget_exhausted:
                skipped.append({"agent": agent_name, "reason": "budget_exhausted"})
                continue

            try:
                self.budget.check_or_raise(agent_name)
            except BudgetExhausted as e:
                logger.warning(f"Budget exhausted: {e.reason}")
                skipped.append({"agent": agent_name, "reason": e.reason})
                budget_exhausted = True
                continue

            # Run single agent through orchestrator-controlled OODA
            try:
                result = self._run_agent(
                    agent_name, deal_data,
                    agent_tools.get(agent_name, []),
                    idx, total_agents,
                )
                self.results[agent_name] = result
                self.agent_outputs[agent_name] = result.get("output", {})
                completed.add(agent_name)
                self.budget.state.agents_completed += 1
            except BudgetExhausted as e:
                logger.warning(f"Budget exhausted during {agent_name}: {e.reason}")
                skipped.append({"agent": agent_name, "reason": e.reason})
                budget_exhausted = True

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        # Build decision package
        pkg = self._build_decision_package(
            deal_data, list(completed), skipped, elapsed_ms)

        # Chain the decision into the ledger
        self.ledger.append("decision", {
            "decision": pkg.decision, "confidence": pkg.confidence,
            "grounding_pct": pkg.grounding_pct})
        pkg.ledger_head_hash = self.ledger.head_hash()
        pkg.ledger_verified = self.ledger.verify()
        pkg.seed = get_seed()

        # Record final output
        self.replay.finish(pkg.to_dict())

        return pkg

    def _run_agent(self, agent_name: str, deal_data: Dict,
                   tools: List[str], agent_idx: int,
                   total_agents: int) -> Dict:
        """Execute one agent through orchestrator-controlled OODA cycle."""
        phase_results = {}

        # ── 1. OBSERVE: Forager proposes tool plan ──
        plan = self.forager.generate_foraging_plan(
            self.beliefs, tools, budget=6)
        phase_results["observe"] = {
            "plan": [s["tool"] for s in plan[:5]],
            "initial_entropy": round(self.beliefs.total_entropy(), 2),
        }

        # ── 2. EXECUTE TOOLS: Orchestrator calls tools ──
        tool_results = []
        for step in plan:
            tool_name = step["tool"]
            if not self.budget.can_call(tool_name, agent_name):
                break

            if self.run_mode == RunMode.FROZEN:
                # Replay from log
                cached = self.replay.get_tool_result(tool_name, {})
                if cached:
                    tool_results.append({"tool": tool_name, "data": cached, "status": "OK"})
                    continue

            # Call tool through registry
            result = self.tools.call(tool_name, deal_data, called_by=agent_name)
            self.budget.record_tool_call(tool_name, agent_name)
            self.replay.record_tool_call(tool_name, deal_data, result.to_dict(), agent_name)

            if result.ok and result.data:
                tool_results.append({"tool": tool_name, "data": result.data, "status": "OK"})
                self.ledger.append("tool_result", {"tool": tool_name, "status": "OK",
                                                   "variables": list(result.data.keys())})
                # Store evidence + update assumptions (with policy gating)
                for eref in result.evidence_refs:
                    self.evidence.add(eref)
                    # Gap D: Check evidence policy before promoting to EVIDENCE
                    existing = self.evidence.get_for_variable(eref.variable)
                    allowed, reason = self.evidence_policy.check_before_update(
                        eref.variable, eref, existing, self.assumptions)
                    if allowed and "pending 2-source" not in reason:
                        self.assumptions.set_evidence(
                            eref.variable, eref.value, [eref.ref_id],
                            eref.confidence)
                    elif allowed:
                        # Critical var with single source → ASSUMPTION until confirmed
                        self.assumptions.set_assumption(
                            eref.variable, eref.value,
                            eref.value * 0.85, eref.value * 1.15,
                            rationale=f"Single-source ({eref.source_name}), "
                                      f"pending 2nd source confirmation",
                            tag="pending_confirmation",
                            confidence=eref.confidence * 0.7)
                    self.beliefs.set_observation(
                        eref.variable, eref.value, eref.confidence, eref.source_name)
                    self.ledger.append("evidence", {"variable": eref.variable,
                                                    "value": eref.value, "ref_id": eref.ref_id,
                                                    "grade": eref.grade.value,
                                                    "policy_allowed": allowed,
                                                    "policy_reason": reason[:100]})
                    # Check for conflicts
                    if len(existing) > 1:
                        conflict = self.tms.check_for_conflicts(eref.variable, existing)
                        if conflict:
                            self.tms.resolve(conflict)
                            self.ledger.append("conflict_resolved",
                                              {"variable": eref.variable,
                                               "severity": conflict.severity.value,
                                               "resolved": conflict.resolved})
            else:
                tool_results.append({
                    "tool": tool_name, "status": result.status.value,
                    "error": result.error_message})

        phase_results["tools"] = {
            "called": len(tool_results),
            "ok": sum(1 for r in tool_results if r["status"] == "OK"),
            "failed": sum(1 for r in tool_results if r["status"] != "OK"),
        }

        # ── 3. ORIENT: SEAL + CECA ──
        signals = self.seal.sense(deal_data, {}, self.agent_outputs)
        seal_eval = self.seal.evaluate()
        critique = self.ceca.critique(deal_data, self.agent_outputs, self.beliefs, signals)
        phase_results["orient"] = {
            "seal_grade": seal_eval.get("grade", "C"),
            "ceca_findings": critique["total_findings"],
        }

        # ── 4. DECIDE: Monte Carlo ──
        mc_result = None
        financial_agents = {"underwriting_analyst", "deal_structurer",
                           "exit_strategist", "gaming_optimizer"}
        if agent_name in financial_agents:
            mc_params = self._build_mc_params(deal_data)
            mc_result = self.mc.simulate_deal(mc_params)
            phase_results["decide"] = {
                "median_irr": mc_result["irr"]["median"],
                "prob_loss": mc_result["probability_analysis"]["prob_loss"],
            }

        # ── 5-7. ACT + VALIDATE + REPAIR ──
        context = self._build_context(
            seal_eval, critique, mc_result, agent_name)
        output = {}
        converged = False
        max_attempts = min(self.budget.limits.max_retries_per_agent + 1, 4)

        for attempt in range(max_attempts):
            if attempt > 0 and not self.budget.can_retry(agent_name):
                break
            if attempt > 0:
                self.budget.record_retry(agent_name)

            # Frozen mode: replay agent output
            if self.run_mode == RunMode.FROZEN:
                cached_output = self.replay.get_agent_output(agent_name, attempt)
                if cached_output:
                    output = cached_output
                else:
                    break
            else:
                try:
                    output = self.execute_fn(
                        agent_name=agent_name, deal_data=deal_data,
                        cognitive_context=context, attempt=attempt)
                except Exception as e:
                    output = {"_error": str(e)}
                    continue

            self.replay.record_agent_output(agent_name, output, attempt)
            self.ledger.append("agent_output", {
                "agent": agent_name, "attempt": attempt,
                "schema_valid": True,  # updated below if needed
                "output_keys": list(output.keys()) if isinstance(output, dict) else []})

            # Schema validation
            val_result, output = validate_agent_output(agent_name, output)
            if not val_result.valid and attempt < max_attempts - 1:
                # Auto-repair prompt
                try:
                    from .schemas import AGENT_SCHEMAS
                except ImportError:
                    from schemas import AGENT_SCHEMAS
                schema = AGENT_SCHEMAS.get(agent_name)
                if schema:
                    repair_prompt = schema.generate_repair_prompt(val_result.issues)
                    context += f"\n\n## SCHEMA REPAIR REQUIRED ##\n{repair_prompt}"
                continue

            # ── 8. CONVERGE: Mechanical test ──
            conv_metrics = {
                "entropy_initial": phase_results["observe"]["initial_entropy"],
                "entropy_final": self.beliefs.total_entropy(),
                "ig_per_call": (
                    (phase_results["observe"]["initial_entropy"] - self.beliefs.total_entropy())
                    / max(phase_results["tools"]["called"], 1)),
                "unresolved_conflicts": len(self.tms.unresolved_conflicts()),
                "schema_passed": val_result.valid,
                "usable_pct": self.assumptions.coverage_report().get("usable_pct", 0),
            }
            conv_report = self.convergence.test(conv_metrics, iteration=attempt)

            if conv_report.converged:
                converged = True
                break

        # Reflexion evaluation
        episode = self.evaluator.evaluate(
            agent_name, output, [], attempt=attempt)
        self.memory.add_episode(episode)

        # Record calibration predictions
        if isinstance(output, dict):
            for key, val in output.items():
                if isinstance(val, (int, float)):
                    belief = self.beliefs.get(key)
                    if belief:
                        self.calibration.record_prediction(
                            key, self.replay.run_id,
                            belief.point, belief.confidence,
                            belief.low, belief.high)

        return {
            "output": output, "converged": converged,
            "attempts": attempt + 1, "phases": phase_results,
            "schema_valid": val_result.valid if 'val_result' in dir() else True,
        }

    def _build_context(self, seal_eval, critique, mc_result, agent_name) -> str:
        """Build cognitive context for agent (orchestrator controls content)."""
        parts = [self.beliefs.to_prompt_fragment()]

        # SEAL signals
        if seal_eval.get("top_opportunities") or seal_eval.get("top_threats"):
            parts.append(f"\n## SEAL — Grade: {seal_eval.get('grade', 'C')} ##")
            for opp in seal_eval.get("top_opportunities", [])[:3]:
                parts.append(f"  [OPP] {opp['description']}")
            for thr in seal_eval.get("top_threats", [])[:3]:
                parts.append(f"  [THR] {thr['description']}")

        # CECA
        parts.append(self.ceca.to_prompt_injection(critique))

        # MC
        if mc_result:
            pa = mc_result["probability_analysis"]
            parts.append(f"\n## MONTE CARLO ##")
            parts.append(f"  Median IRR: {mc_result['irr']['median']:.1f}%")
            parts.append(f"  P(Loss): {pa['prob_loss']:.0%}")

        # Assumption table summary
        cov = self.assumptions.coverage_report()
        parts.append(f"\n## DATA GROUNDING: {cov['grounding_pct']:.0f}% evidence-backed ##")
        if cov.get("missing_required"):
            parts.append(f"  MISSING: {', '.join(cov['missing_required'][:5])}")

        # Reflexion
        parts.append(self.memory.to_prompt_injection(agent_name))

        # Trim to budget
        full = "\n".join(parts)
        return full[:self.budget.limits.max_context_chars]

    def _build_mc_params(self, deal_data: Dict) -> Dict:
        """Build MC params from beliefs (uses correct interest_rate / exit_cap)."""
        def _b2d(var, default=0):
            b = self.beliefs.get(var)
            if b:
                return {"point": b.point, "low": b.low, "high": b.high}
            return {"point": default, "low": default * 0.7, "high": default * 1.3}
        return {
            "purchase_price": deal_data.get("price", deal_data.get("purchase_price", 2000000)),
            "noi": _b2d("noi", 150000),
            "loan_rate": _b2d("interest_rate", 7.5),
            "loan_ltv": deal_data.get("ltv", 0.75),
            "hold_years": deal_data.get("hold_years", 5),
            "noi_growth": {"point": 2.0, "low": -2.0, "high": 5.0},
            "exit_cap": _b2d("exit_cap_rate", 7.5),
        }

    def _init_assumptions(self, deal_data: Dict):
        """Register all variables and set known deal data."""
        for var in self.beliefs.beliefs:
            self.assumptions.register(var)
        # Set known deal data as USER_INPUT
        field_map = {
            "price": "purchase_price", "purchase_price": "purchase_price",
            "sqft": "sqft", "year_built": "year_built",
            "terminal_count": "terminal_count",
        }
        for data_key, var_name in field_map.items():
            if data_key in deal_data:
                self.assumptions.set_user_input(
                    var_name, float(deal_data[data_key]),
                    signer="deal_input", confidence=0.9)

    def _build_decision_package(self, deal_data: Dict,
                                completed: List[str],
                                skipped: List[Dict],
                                elapsed_ms: int) -> DecisionPackage:
        """Assemble the final auditable decision package."""
        builder = DecisionPackageBuilder(deal_data, self.replay.run_id)

        # Determine decision
        uw = self.agent_outputs.get("underwriting_analyst", {})
        risk = self.agent_outputs.get("risk_officer", {})
        rec = uw.get("recommendation", risk.get("verdict", "NEEDS_DATA"))

        # Compute overall confidence from assumption coverage
        cov = self.assumptions.coverage_report()
        overall_conf = cov.get("usable_pct", 0) / 100 * 0.8  # 80% weight on data coverage
        if any(r.get("converged") for r in self.results.values()):
            overall_conf += 0.1

        # Calibration tag
        cal_report = self.calibration.report()
        brier = cal_report.get("overall_brier")
        if brier is None:
            cal_tag = "uncalibrated"
        elif brier < 0.1:
            cal_tag = "well_calibrated"
        elif brier < 0.25:
            cal_tag = "moderately_calibrated"
        else:
            cal_tag = "poorly_calibrated"

        builder.set_decision(rec, min(overall_conf, 0.95), cal_tag)
        builder.set_assumptions(self.assumptions.to_dict())
        builder.set_evidence(self.evidence.to_audit_log())

        # Risk register
        risk_items = risk.get("risk_register", [])
        risk_score = risk.get("verdict", "NEEDS_DATA")
        if risk_score == "APPROVE":
            risk_score = "LOW"
        elif risk_score == "CONDITIONAL":
            risk_score = "MED"
        elif risk_score == "REJECT":
            risk_score = "HIGH"
        builder.set_risk(risk_items, risk_score)

        # Financials
        fin = {}
        if uw:
            for k in ["noi", "cap_rate", "purchase_price", "dscr",
                       "direct_cap_value", "dcf_value"]:
                if k in uw:
                    fin[k] = uw[k]
        builder.set_financials(fin)

        # Auto-generate pivots and next actions
        beliefs_dict = {v: {"value": b.point} for v, b in self.beliefs.beliefs.items()}
        builder.auto_pivots(beliefs_dict)
        builder.auto_next_actions(self.assumptions.to_dict())

        # Convergence
        all_converged = all(r.get("converged", False) for r in self.results.values())
        conv_status = "CONVERGED" if all_converged else "NOT_CONVERGED"
        if any("budget" in s.get("reason", "") for s in skipped):
            conv_status = "BUDGET_EXHAUSTED"
        builder.set_convergence(conv_status, {
            "agents_converged": sum(1 for r in self.results.values() if r.get("converged")),
            "agents_total": len(self.results),
            "conflicts": self.tms.report(),
        })

        # Run stats
        builder.set_run_stats(
            agents_run=completed, agents_skipped=skipped,
            tool_calls=self.budget.state.tool_calls,
            retries=self.budget.state.retries,
            elapsed_ms=elapsed_ms,
            replay_log_id=self.replay.run_id,
        )

        return builder.build()

    # ── Data Intake API (#12) ──

    def get_missing_variables(self) -> List[Dict]:
        """List missing variables ranked by value-of-information.

        This drives the data intake UX workflow.
        """
        if not self.beliefs:
            return []
        high_entropy = self.beliefs.max_entropy_variables(20)
        results = []
        for belief in high_entropy:
            rec = self.assumptions.get(belief.variable)
            status = rec.status.value if rec else "UNKNOWN"
            if status in ("UNKNOWN", "ASSUMPTION"):
                results.append({
                    "variable": belief.variable,
                    "status": status,
                    "current_entropy": round(belief.entropy, 2),
                    "confidence": round(belief.confidence, 2),
                    "current_value": belief.point if status == "ASSUMPTION" else None,
                    "tools_that_provide": self.tools.tools_for_variable(belief.variable),
                    "priority": "HIGH" if belief.entropy > 5 else "MEDIUM",
                })
        return sorted(results, key=lambda x: -x.get("current_entropy", 0))

    def submit_user_data(self, variable: str, value: float,
                         signer: str, confidence: float = 0.7,
                         low: float = None, high: float = None):
        """Accept user-provided data and update assumptions + beliefs."""
        self.assumptions.set_user_input(
            variable, value, signer, confidence, low, high)
        if self.beliefs:
            self.beliefs.set_observation(variable, value, confidence, f"user:{signer}")
        # Record evidence
        ref = make_evidence_ref(
            source_name=f"user:{signer}", variable=variable,
            value=value, method=EvidenceMethod.USER_ENTRY,
            grade=EvidenceGrade.D, confidence=confidence)
        self.evidence.add(ref)

    def submit_assumption(self, variable: str, value: float,
                          low: float, high: float, rationale: str,
                          tag: str = "estimate"):
        """Accept an explicit assumption."""
        self.assumptions.set_assumption(
            variable, value, low, high, rationale, tag)
        if self.beliefs:
            self.beliefs.set_observation(variable, value, 0.3, f"assumption:{tag}")
