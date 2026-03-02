"""
engine.brain.ooda — OODA Cognitive Orchestrator + Reflexion (v2 integrated)
==============================================================================
v2 enhancements integrated:
  - TokenBudget + ContextCompressor for adaptive context window management
  - ToolCallCache for cross-agent tool deduplication
  - CalibrationTracker for predicted vs actual accuracy
  - Blackboard for shared working memory across agents
  - OrchestratorAgent for skip/prioritize/conflict resolution
  - BenchmarkSuite for quantitative metrics + ablation
  - EpisodicMemoryStore for vector-similarity episode retrieval
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════
# REFLEXION: EPISODIC MEMORY
# ═════════════════════════════════════════

@dataclass
class Episode:
    agent: str
    attempt: int
    timestamp: float
    tasks_completed: List[str]
    tools_called: List[str]
    output_summary: str
    self_score: float
    validator_score: float
    convergence_passed: bool
    issues_found: List[str]
    reflection: str
    lessons: List[str]

    def to_dict(self) -> Dict:
        return {
            "agent": self.agent, "attempt": self.attempt,
            "self_score": self.self_score, "validator_score": self.validator_score,
            "convergence_passed": self.convergence_passed,
            "issues": self.issues_found, "reflection": self.reflection,
            "lessons": self.lessons,
        }


class ReflexionMemory:
    def __init__(self):
        self.episodes: Dict[str, List[Episode]] = {}
        self.global_lessons: List[str] = []

    def add_episode(self, episode: Episode):
        if episode.agent not in self.episodes:
            self.episodes[episode.agent] = []
        self.episodes[episode.agent].append(episode)
        if not episode.convergence_passed:
            for lesson in episode.lessons:
                if lesson not in self.global_lessons:
                    self.global_lessons.append(lesson)

    def get_agent_history(self, agent: str) -> List[Episode]:
        return self.episodes.get(agent, [])

    def get_last_reflection(self, agent: str) -> Optional[str]:
        episodes = self.get_agent_history(agent)
        return episodes[-1].reflection if episodes else None

    def get_agent_lessons(self, agent: str) -> List[str]:
        lessons = []
        for ep in self.get_agent_history(agent):
            lessons.extend(ep.lessons)
        return list(set(lessons))

    def to_prompt_injection(self, agent: str) -> str:
        episodes = self.get_agent_history(agent)
        if not episodes:
            if self.global_lessons:
                lines = ["\n## LESSONS FROM OTHER AGENTS ##"]
                for lesson in self.global_lessons[-5:]:
                    lines.append(f"  * {lesson}")
                return "\n".join(lines)
            return ""
        last = episodes[-1]
        lines = [f"\n## REFLEXION -- ATTEMPT {last.attempt + 1} ##"]
        lines.append(f"Previous attempt scored {last.self_score:.0%} (self) / {last.validator_score:.0%} (validator)")
        if last.issues_found:
            lines.append(f"\nISSUES FROM LAST ATTEMPT ({len(last.issues_found)}):")
            for issue in last.issues_found[:5]:
                lines.append(f"  x {issue}")
        if last.reflection:
            lines.append(f"\nYOUR OWN REFLECTION:")
            lines.append(f'  "{last.reflection}"')
        if last.lessons:
            lines.append(f"\nLESSONS LEARNED:")
            for lesson in last.lessons:
                lines.append(f"  * {lesson}")
        lines.append("\nApply these lessons. Do NOT repeat the same mistakes.\n")
        return "\n".join(lines)

    def summary(self) -> Dict:
        return {
            "agents_with_episodes": list(self.episodes.keys()),
            "total_episodes": sum(len(eps) for eps in self.episodes.values()),
            "global_lessons": self.global_lessons,
            "per_agent": {
                agent: {
                    "attempts": len(eps),
                    "best_score": max(e.validator_score for e in eps) if eps else 0,
                    "converged": any(e.convergence_passed for e in eps),
                }
                for agent, eps in self.episodes.items()
            },
        }


# ═════════════════════════════════════════
# REFLEXION: SELF-EVALUATOR
# ═════════════════════════════════════════

class ReflexionEvaluator:
    SELF_EVAL_PROMPT = """You just completed an analysis. Evaluate your own work:
OUTPUT: {output_summary}
VALIDATION: {validation_results}
Answer as JSON: {{"completeness": N, "accuracy": N, "quality": N, "reflection": "...", "lessons": ["..."]}}"""

    def evaluate(self, agent_name: str, output: Dict,
                 validation_issues: List[Dict],
                 evaluate_fn: Callable = None,
                 attempt: int = None) -> Episode:
        issues_text = [i.get("message", str(i)) for i in validation_issues]
        has_critical = any(i.get("severity") == "critical" for i in validation_issues)
        error_count = sum(1 for i in validation_issues if i.get("severity") in ("error", "critical"))

        base_score = max(0, min(1.0, 1.0 - error_count * 0.15 - has_critical * 0.3))
        validator_score = max(0, 1.0 - error_count * 0.2)

        if evaluate_fn:
            try:
                eval_result = evaluate_fn(prompt=self.SELF_EVAL_PROMPT.format(
                    output_summary=json.dumps(output)[:2000],
                    validation_results=json.dumps(validation_issues)[:1000]))
                if isinstance(eval_result, dict):
                    base_score = (eval_result.get("completeness", 5) + eval_result.get("accuracy", 5) + eval_result.get("quality", 5)) / 30
                    reflection = eval_result.get("reflection", "")
                    lessons = eval_result.get("lessons", [])
                else:
                    reflection, lessons = self._heuristic_reflection(issues_text)
            except Exception:
                reflection, lessons = self._heuristic_reflection(issues_text)
        else:
            reflection, lessons = self._heuristic_reflection(issues_text)

        # Use caller-provided attempt number, or track internally
        if attempt is None:
            attempt = len(self._prev_attempts.get(agent_name, []))

        # Update internal tracking so subsequent calls auto-increment
        if agent_name not in self._prev_attempts:
            self._prev_attempts[agent_name] = []
        self._prev_attempts[agent_name].append(attempt)

        return Episode(
            agent=agent_name, attempt=attempt,
            timestamp=time.time(),
            tasks_completed=list(output.keys()) if isinstance(output, dict) else [],
            tools_called=output.get("_tools_called", []) if isinstance(output, dict) else [],
            output_summary=json.dumps(output)[:500],
            self_score=round(base_score, 2), validator_score=round(validator_score, 2),
            convergence_passed=not has_critical and error_count == 0,
            issues_found=issues_text[:10], reflection=reflection, lessons=lessons,
        )

    def _heuristic_reflection(self, issues: List[str]) -> Tuple[str, List[str]]:
        if not issues:
            return "Output passed all checks.", ["Tool-backed numbers pass validation."]
        lessons, parts = [], []
        for issue in issues[:5]:
            lower = issue.lower()
            if "missing" in lower:
                f = issue.split("Missing ")[-1].split(" --")[0] if "Missing " in issue else "a required field"
                parts.append(f"Include {f}")
                lessons.append(f"Always produce {f}.")
            elif "match" in lower or "convergence" in lower:
                parts.append("Use ledger numbers")
                lessons.append("Check ledger values BEFORE calculating.")
            else:
                parts.append(f"Fix: {issue[:80]}")
        return "If I redo this: " + "; ".join(parts), lessons[:3]

    _prev_attempts: Dict = {}


# ═════════════════════════════════════════
# OODA RESULT + LOOP
# ═════════════════════════════════════════

class OODAPhase(Enum):
    OBSERVE = "observe"
    ORIENT = "orient"
    DECIDE = "decide"
    ACT = "act"
    REFLECT = "reflect"


@dataclass
class OODAResult:
    agent_name: str
    output: Dict
    phase_results: Dict[str, Dict]
    converged: bool
    attempts: int
    total_time_ms: int
    initial_entropy: float = 0
    final_entropy: float = 0
    entropy_reduction: float = 0
    tools_planned: int = 0
    tools_executed: int = 0
    signals_detected: int = 0
    opportunity_score: float = 0
    critique_findings: int = 0
    bias_detections: int = 0
    adjusted_confidence: float = 1.0
    simulation_ran: bool = False
    prob_positive_irr: float = 0
    prob_loss: float = 0
    self_score: float = 0
    reflection: str = ""

    def to_dict(self) -> Dict:
        return {
            "agent": self.agent_name, "converged": self.converged,
            "attempts": self.attempts, "total_time_ms": self.total_time_ms,
            "active_inference": {
                "initial_entropy": round(self.initial_entropy, 2),
                "final_entropy": round(self.final_entropy, 2),
                "entropy_reduction": round(self.entropy_reduction, 2),
                "info_gain_pct": round((1 - self.final_entropy / max(self.initial_entropy, 1)) * 100, 1),
                "tools_planned": self.tools_planned, "tools_executed": self.tools_executed,
            },
            "seal": {"signals_detected": self.signals_detected, "opportunity_score": round(self.opportunity_score, 3)},
            "ceca": {"findings": self.critique_findings, "biases_detected": self.bias_detections, "adjusted_confidence": round(self.adjusted_confidence, 2)},
            "monte_carlo": {"ran": self.simulation_ran, "prob_positive_irr": round(self.prob_positive_irr, 3), "prob_loss": round(self.prob_loss, 3)},
            "reflexion": {"self_score": round(self.self_score, 2), "reflection": self.reflection[:200]},
            "phases": self.phase_results,
        }


class OODALoop:
    """Single agent OODA + Reflexion cycle with v2 context compression."""

    def __init__(self, execute_fn: Callable, max_retries: int = 3):
        self.execute_fn = execute_fn
        self.max_retries = max_retries

        try:
            from .active_inference import BeliefState, FreeEnergyCalc, EpistemicForager
            from .seal_ceca import SEALDetector, CECACritic, MonteCarloSimulator
            from .convergence import OutputValidator, ConvergenceChecker
        except ImportError:
            from active_inference import BeliefState, FreeEnergyCalc, EpistemicForager
            from seal_ceca import SEALDetector, CECACritic, MonteCarloSimulator
            from convergence import OutputValidator, ConvergenceChecker

        self.BeliefState = BeliefState
        self.FreeEnergyCalc = FreeEnergyCalc
        self.forager = EpistemicForager()
        self.seal = SEALDetector()
        self.ceca = CECACritic()
        self.mc = MonteCarloSimulator(num_simulations=1000)
        self.validator = OutputValidator()
        self.convergence = ConvergenceChecker()
        self.evaluator = ReflexionEvaluator()

        # v2 components (graceful fallback)
        self.token_budget = None
        self.compressor = None
        self.tool_cache = None
        self.calibration = None
        try:
            try:
                from .v2_fixes import (TokenBudget, ContextCompressor, ToolCallCache, CalibrationTracker)
            except ImportError:
                from v2_fixes import (TokenBudget, ContextCompressor, ToolCallCache, CalibrationTracker)
            self.token_budget = TokenBudget(total_budget=3500)
            self.compressor = ContextCompressor()
            self.tool_cache = ToolCallCache()
            self.calibration = CalibrationTracker()
        except ImportError:
            pass

    def run(self, agent_name: str, deal_data: Dict, agent_tools: List[str],
            ledger: Any, memory: ReflexionMemory, agent_outputs: Dict = None,
            agent_index: int = 0, total_agents: int = 1) -> OODAResult:
        start = time.perf_counter()
        agent_outputs = agent_outputs or {}
        phase_results = {}

        # OBSERVE
        beliefs = self.BeliefState(deal_data)
        if hasattr(ledger, 'all_entries'):
            for fn, entry in ledger.all_entries().items():
                beliefs.set_observation(fn, entry["value"], 0.8, f"ledger:{entry['source']}")
        initial_entropy = beliefs.total_entropy()
        foraging_plan = self.forager.generate_foraging_plan(beliefs, agent_tools, budget=8)
        phase_results["observe"] = {"initial_entropy": round(initial_entropy, 2), "foraging_plan": foraging_plan[:5]}

        # ORIENT
        signals = self.seal.sense(deal_data, {}, agent_outputs)
        seal_eval = self.seal.evaluate()
        critique = self.ceca.critique(deal_data, agent_outputs, beliefs, signals)
        phase_results["orient"] = {
            "seal_score": seal_eval.get("score", 0.5), "seal_grade": seal_eval.get("grade", "C"),
            "signals": len(signals), "ceca_findings": critique["total_findings"],
            "ceca_recommendation": critique["proceed_recommendation"],
            "adjusted_confidence": critique["adjusted_confidence"],
        }

        # DECIDE
        mc_result = None
        financial_agents = {"underwriting_analyst", "deal_structurer", "exit_strategist", "gaming_optimizer"}
        if agent_name in financial_agents:
            mc_params = self._build_mc_params(deal_data, beliefs, agent_outputs)
            mc_result = self.mc.simulate_deal(mc_params)
            phase_results["decide"] = {"monte_carlo": {
                "prob_positive_irr": mc_result["probability_analysis"]["prob_positive_irr"],
                "prob_loss": mc_result["probability_analysis"]["prob_loss"],
                "median_irr": mc_result["irr"]["median"],
            }}
        else:
            phase_results["decide"] = {"monte_carlo": None}

        # ACT (with Reflexion retry)
        output, converged, final_issues, episode = {}, False, [], None
        for attempt in range(self.max_retries + 1):
            cognitive_prompt = self._build_cognitive_prompt(
                beliefs, foraging_plan, seal_eval, critique,
                mc_result, ledger, memory, agent_name, attempt,
                agent_index=agent_index, total_agents=total_agents)
            try:
                output = self.execute_fn(agent_name=agent_name, deal_data=deal_data,
                                         cognitive_context=cognitive_prompt, attempt=attempt)
            except Exception as e:
                output = {"_error": str(e)}
                continue

            val_issues = self.validator.validate(agent_name, output)
            conv_issues = self.convergence.check(agent_name, output, ledger)
            all_issues = val_issues + conv_issues
            final_issues = all_issues

            episode = self.evaluator.evaluate(
                agent_name, output,
                [i.to_dict() if hasattr(i, 'to_dict') else {"message": str(i)} for i in all_issues],
                attempt=attempt)
            memory.add_episode(episode)

            critical = [i for i in all_issues if hasattr(i, 'severity') and i.severity.value == "critical"]
            errors = [i for i in all_issues if hasattr(i, 'severity') and i.severity.value == "error"]
            if not critical and not errors:
                converged = True
                break

        # REFLECT — update beliefs with output
        if isinstance(output, dict):
            for key, val in output.items():
                if isinstance(val, (int, float)) and key in beliefs.beliefs:
                    beliefs.set_observation(key, float(val), 0.7, f"agent:{agent_name}")
        final_entropy = beliefs.total_entropy()

        phase_results["act"] = {"attempts": attempt + 1, "converged": converged, "issues": len(final_issues)}
        phase_results["reflect"] = {
            "self_score": episode.self_score if episode else 0,
            "reflection": episode.reflection if episode else "",
        }

        elapsed = int((time.perf_counter() - start) * 1000)
        return OODAResult(
            agent_name=agent_name, output=output, phase_results=phase_results,
            converged=converged, attempts=attempt + 1, total_time_ms=elapsed,
            initial_entropy=initial_entropy, final_entropy=final_entropy,
            entropy_reduction=initial_entropy - final_entropy,
            tools_planned=len(foraging_plan),
            tools_executed=len(output.get("_tools_called", [])) if isinstance(output, dict) else 0,
            signals_detected=len(signals), opportunity_score=seal_eval.get("score", 0.5),
            critique_findings=critique["total_findings"],
            bias_detections=len(critique.get("bias_detections", [])),
            adjusted_confidence=critique["adjusted_confidence"],
            simulation_ran=mc_result is not None,
            prob_positive_irr=mc_result["probability_analysis"]["prob_positive_irr"] if mc_result else 0,
            prob_loss=mc_result["probability_analysis"]["prob_loss"] if mc_result else 0,
            self_score=episode.self_score if episode else 0,
            reflection=episode.reflection if episode else "",
        )

    def _build_cognitive_prompt(self, beliefs, foraging_plan, seal_eval, critique,
                                mc_result, ledger, memory, agent_name, attempt,
                                agent_index=0, total_agents=1) -> str:
        sections = {}
        belief_text = beliefs.to_prompt_fragment()
        if foraging_plan:
            plan_lines = ["\n## INFORMATION FORAGING PLAN ##"]
            for step in foraging_plan[:6]:
                plan_lines.append(f"  {step['step']}. {step['tool']} -- {step['rationale']}")
            belief_text += "\n".join(plan_lines) + "\n"
        sections["beliefs"] = belief_text

        seal_lines = []
        if seal_eval.get("top_opportunities") or seal_eval.get("top_threats"):
            seal_lines.append(f"\n## SEAL SIGNALS -- Grade: {seal_eval.get('grade', 'C')} ##")
            for opp in seal_eval.get("top_opportunities", [])[:3]:
                seal_lines.append(f"  [OPP] {opp['description']}")
            for thr in seal_eval.get("top_threats", [])[:3]:
                seal_lines.append(f"  [THR] {thr['description']}")
        sections["seal"] = "\n".join(seal_lines)

        sections["ceca"] = self.ceca.to_prompt_injection(critique)

        mc_text = ""
        if mc_result:
            pa = mc_result["probability_analysis"]
            mc_text = (f"\n## MONTE CARLO ##\n"
                       f"  Median IRR: {mc_result['irr']['median']:.1f}%\n"
                       f"  P(IRR > 15%): {pa['prob_irr_above_15']:.0%}\n"
                       f"  P(Loss): {pa['prob_loss']:.0%}\n"
                       f"  5% VaR: ${pa['value_at_risk_5pct']:,.0f}\n")
        sections["monte_carlo"] = mc_text
        sections["ledger"] = ledger.inject_into_prompt() if hasattr(ledger, 'inject_into_prompt') else ""
        sections["reflexion"] = memory.to_prompt_injection(agent_name)

        # v2: Apply context compression + token budget
        if self.compressor and self.token_budget:
            level = self.compressor.auto_level(agent_index, total_agents, attempt)
            return self.compressor.compress(sections, level=level, budget=self.token_budget)
        return "\n".join(v for v in sections.values() if v)

    def _build_mc_params(self, deal_data, beliefs, agent_outputs):
        def _b2d(var, default=0):
            b = beliefs.get(var)
            return {"point": b.point, "low": b.low, "high": b.high} if b else {"point": default, "low": default * 0.7, "high": default * 1.3}
        params = {
            "purchase_price": deal_data.get("price", deal_data.get("purchase_price", 2000000)),
            "noi": _b2d("noi", 150000), "loan_rate": _b2d("interest_rate", 7.5),
            "loan_ltv": deal_data.get("ltv", 0.75), "hold_years": deal_data.get("hold_years", 5),
            "noi_growth": {"point": 2.0, "low": -2.0, "high": 5.0}, "exit_cap": _b2d("exit_cap_rate", 7.5),
        }
        if deal_data.get("gaming_eligible"):
            params["gaming_nti"] = _b2d("nti_per_terminal", 200)
            params["terminal_count"] = deal_data.get("terminal_count", 5)
        return params


# ═════════════════════════════════════════
# COGNITIVE ORCHESTRATOR (v2 integrated)
# ═════════════════════════════════════════

class CognitiveOrchestrator:
    """Full pipeline with v2: Blackboard, OrchestratorAgent, BenchmarkSuite, EpisodicMemoryStore."""

    def __init__(self, execute_fn: Callable, max_retries: int = 3):
        try:
            from .convergence import ReconciliationLedger
        except ImportError:
            from convergence import ReconciliationLedger
        self.execute_fn = execute_fn
        self.max_retries = max_retries
        self.ledger = ReconciliationLedger()
        self.memory = ReflexionMemory()
        self.results: Dict[str, OODAResult] = {}
        self.ooda = OODALoop(execute_fn=execute_fn, max_retries=max_retries)

        self.blackboard = self.orch_agent = self.benchmark = self.episodic_store = None
        try:
            try:
                from .v2_fixes import Blackboard, OrchestratorAgent, BenchmarkSuite, EpisodicMemoryStore
            except ImportError:
                from v2_fixes import Blackboard, OrchestratorAgent, BenchmarkSuite, EpisodicMemoryStore
            self.blackboard = Blackboard()
            self.orch_agent = OrchestratorAgent(self.blackboard)
            self.benchmark = BenchmarkSuite()
            self.episodic_store = EpisodicMemoryStore()
        except ImportError:
            pass

    def run(self, deal_data: Dict, active_agents: List[str],
            agent_tools: Dict[str, List[str]] = None) -> Dict:
        start = time.perf_counter()
        agent_tools = agent_tools or {}
        agent_outputs, completed, skipped = {}, set(), []
        total_agents = len(active_agents)

        for idx, agent_name in enumerate(active_agents):
            if self.orch_agent:
                decision = self.orch_agent.should_run(agent_name, deal_data, completed, self.ledger)
                if not decision["run"]:
                    skipped.append({"agent": agent_name, "reason": decision["reason"]})
                    logger.info(f"SKIP {agent_name}: {decision['reason']}")
                    continue

            result = self.ooda.run(
                agent_name=agent_name, deal_data=deal_data,
                agent_tools=agent_tools.get(agent_name, []),
                ledger=self.ledger, memory=self.memory, agent_outputs=agent_outputs,
                agent_index=idx, total_agents=total_agents)

            self.results[agent_name] = result
            agent_outputs[agent_name] = result.output
            completed.add(agent_name)

            if self.blackboard and isinstance(result.output, dict):
                for key, val in result.output.items():
                    if isinstance(val, (int, float)):
                        self.blackboard.write(agent_name, f"financials.{key}", val)

            if self.episodic_store:
                self.episodic_store.add({"agent": agent_name, "converged": result.converged,
                                         "attempts": result.attempts, "entropy_reduction": result.entropy_reduction})

            if self.benchmark:
                self.benchmark.record_agent_run({
                    "agent": agent_name, "attempts": result.attempts, "total_time_ms": result.total_time_ms,
                    "active_inference": {"tools_executed": result.tools_planned,
                                         "entropy_reduction": result.entropy_reduction,
                                         "info_gain_pct": round(result.entropy_reduction / max(result.initial_entropy, 1) * 100, 1)},
                    "seal": {"signals_detected": result.signals_detected, "opportunity_score": result.opportunity_score},
                    "ceca": {"findings": result.critique_findings, "biases_detected": result.bias_detections,
                             "adjusted_confidence": result.adjusted_confidence},
                    "reflexion": {"self_score": result.self_score},
                })

            status = "OK" if result.converged else "FAIL"
            logger.info(f"{status} {agent_name}: entropy {result.initial_entropy:.0f}->{result.final_entropy:.0f}, "
                        f"SEAL={result.opportunity_score:.2f}, CECA={result.critique_findings} findings, "
                        f"{result.attempts} attempts, {result.total_time_ms}ms")

        elapsed = int((time.perf_counter() - start) * 1000)
        return self._generate_report(elapsed, skipped)

    def _generate_report(self, elapsed_ms: int, skipped: List[Dict] = None) -> Dict:
        converged = sum(1 for r in self.results.values() if r.converged)
        total = len(self.results)
        te_i = sum(r.initial_entropy for r in self.results.values())
        te_f = sum(r.final_entropy for r in self.results.values())
        total_attempts = sum(r.attempts for r in self.results.values())

        report = {
            "pipeline_health": "CONVERGED" if converged == total else ("PARTIAL" if converged > total * 0.7 else "FAILED"),
            "elapsed_ms": elapsed_ms, "agents_run": total, "agents_converged": converged,
            "agents_skipped": skipped or [], "total_attempts": total_attempts, "retries": total_attempts - total,
            "active_inference": {
                "initial_total_entropy": round(te_i, 1), "final_total_entropy": round(te_f, 1),
                "entropy_reduction_pct": round((1 - te_f / max(te_i, 1)) * 100, 1),
            },
            "seal": {"total_signals": sum(r.signals_detected for r in self.results.values()),
                     "avg_opportunity_score": round(sum(r.opportunity_score for r in self.results.values()) / max(total, 1), 3)},
            "ceca": {"total_findings": sum(r.critique_findings for r in self.results.values()),
                     "total_biases_detected": sum(r.bias_detections for r in self.results.values()),
                     "avg_adjusted_confidence": round(sum(r.adjusted_confidence for r in self.results.values()) / max(total, 1), 2)},
            "monte_carlo": {
                "agents_simulated": sum(1 for r in self.results.values() if r.simulation_ran),
                "avg_prob_positive_irr": round(
                    sum(r.prob_positive_irr for r in self.results.values() if r.simulation_ran) /
                    max(sum(1 for r in self.results.values() if r.simulation_ran), 1), 3),
            },
            "reflexion": {"total_episodes": sum(len(eps) for eps in self.memory.episodes.values()),
                          "global_lessons_learned": len(self.memory.global_lessons), "lessons": self.memory.global_lessons[:10]},
            "reconciliation_ledger": self.ledger.report() if hasattr(self.ledger, 'report') else {},
            "agent_results": {name: result.to_dict() for name, result in self.results.items()},
        }

        # v2 report sections
        if self.orch_agent:
            report["orchestrator"] = self.orch_agent.report()
        if self.blackboard:
            snap = self.blackboard.snapshot()
            report["blackboard"] = {"entries": snap["entries"], "flags": snap["flags"],
                                    "hypotheses_count": len(snap["hypotheses"]), "questions_count": len(snap["questions"])}
        if self.benchmark:
            report["benchmark"] = self.benchmark.full_report().get("aggregates", {})
        if self.ooda.tool_cache:
            report["tool_cache"] = self.ooda.tool_cache.stats()
        if self.ooda.calibration:
            cal = self.ooda.calibration.calibration_report()
            report["calibration"] = {"predictions": cal.get("total_predictions", 0), "quality": cal.get("calibration_quality", "unknown")}

        return report
