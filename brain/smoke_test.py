#!/usr/bin/env python3
"""
engine.smoke_test — 60-Second Smoke Test
==========================================
Run after every change. If this breaks, you shipped a broken upgrade.

Tests:
  1. All modules import cleanly
  2. Engine instantiates (orchestrator + all subsystems)
  3. Minimal pipeline with stubbed tools → decision package
  4. Determinism: same seed → identical outputs
  5. Correlated MC produces results
  6. Capital stack waterfall computes
  7. Schema rejects extra fields
  8. Ledger verifies + detects tamper
  9. RBAC blocks unauthorized access
  10. Cost-aware forager ranks tools

Usage:
  python -m engine.smoke_test
  # or standalone:
  cd engine && python smoke_test.py

Exit code 0 = all pass, 1 = failure
"""

from __future__ import annotations
import sys
import time
import json

# ── Bootstrap path for standalone execution ──
if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def smoke_test() -> bool:
    """Run all smoke tests. Returns True if all pass."""
    start = time.perf_counter()
    results = []

    def check(name: str, fn):
        t0 = time.perf_counter()
        try:
            fn()
            ms = int((time.perf_counter() - t0) * 1000)
            results.append((name, True, ms, ""))
            print(f"  ✅ {name} ({ms}ms)")
        except Exception as e:
            ms = int((time.perf_counter() - t0) * 1000)
            results.append((name, False, ms, str(e)))
            print(f"  ❌ {name} ({ms}ms): {e}")

    print("=" * 60)
    print("SMOKE TEST — Truth Accretion Engine")
    print("=" * 60)

    # ── 1. Module imports ──
    def test_imports():
        modules = [
            "assumptions", "evidence", "schemas", "tool_contracts",
            "truth_maintenance", "budget", "convergence_v2", "replay",
            "decision_package", "active_inference", "seal_ceca", "ooda",
            "determinism", "ledger", "evidence_policy", "security",
            "tools_unified", "correlated_mc", "capital_stack",
            "scenario_tree", "cost_aware_forager", "orchestrator_v3",
            "calibration", "data_tools", "agents", "pipeline_config",
        ]
        for mod in modules:
            __import__(mod)
    check("1. Module imports (26 modules)", test_imports)

    # ── 2. Engine instantiation ──
    def test_instantiation():
        from orchestrator_v3 import TruthAccretionOrchestrator
        from tools_unified import create_tool_registry
        from tool_contracts import ToolContract
        from evidence import EvidenceGrade

        registry = create_tool_registry(live=True)
        assert len(registry.available_tools()) >= 16

        orch = TruthAccretionOrchestrator(
            execute_fn=lambda **kw: {"recommendation": "GO"},
            tool_registry=registry,
        )
        assert orch.forager is not None
        assert orch.beliefs is None  # Not yet initialized
        assert type(orch.forager).__name__ == "CostAwareForager"
    check("2. Engine instantiation", test_instantiation)

    # ── 3. Minimal pipeline → decision package ──
    def test_pipeline():
        from determinism import seed_engine
        from orchestrator_v3 import TruthAccretionOrchestrator
        from tools_unified import create_tool_registry
        from tool_contracts import ToolContract
        from evidence import EvidenceGrade

        seed_engine(run_id="smoke_test_pipeline")
        registry = create_tool_registry(live=True)

        # Register mock tools
        for name, outputs, grade, impl in [
            ("census_demographics", ["population", "median_income"], EvidenceGrade.B,
             lambda **kw: {"population": 47000, "median_income": 61500}),
            ("traffic_counts", ["traffic_count"], EvidenceGrade.B,
             lambda **kw: {"traffic_count": 22000}),
            ("pull_comps", ["cap_rate", "noi"], EvidenceGrade.C,
             lambda **kw: {"cap_rate": 7.2, "noi": 188000}),
            ("fred_economic_data", ["interest_rate"], EvidenceGrade.A,
             lambda **kw: {"interest_rate": 7.25}),
            ("crime_data", ["crime_rate"], EvidenceGrade.B,
             lambda **kw: {"crime_rate": 3.8}),
        ]:
            try:
                registry.register(ToolContract(
                    name=name, description=name,
                    required_params=[], output_variables=outputs,
                    evidence_grade=grade), implementation=impl)
            except Exception:
                pass

        orch = TruthAccretionOrchestrator(
            execute_fn=lambda **kw: {"recommendation": "GO", "noi": 190000},
            tool_registry=registry,
        )

        pkg = orch.run(
            deal_data={"address": "123 Test St", "price": 2500000,
                       "property_type": "gas_station"},
            active_agents=["underwriting_analyst", "risk_officer"],
            agent_tools={
                "underwriting_analyst": ["census_demographics", "pull_comps",
                                         "fred_economic_data", "traffic_counts"],
                "risk_officer": ["crime_data", "census_demographics"],
            },
        )

        assert pkg.decision in ("GO", "NO_GO", "CONDITIONAL", "NEEDS_DATA")
        assert pkg.seed is not None
        assert pkg.ledger_verified is True
        assert pkg.total_tool_calls > 0
    check("3. Pipeline → decision package", test_pipeline)

    # ── 4. Determinism ──
    def test_determinism():
        from determinism import seed_engine
        from seal_ceca import MonteCarloSimulator

        params = {
            "purchase_price": 2500000,
            "noi": {"point": 195000, "low": 155000, "high": 235000},
            "loan_rate": {"point": 7.0, "low": 5.5, "high": 9.0},
            "exit_cap": {"point": 7.5, "low": 5.5, "high": 10.0},
        }

        seed_engine(run_id="determinism_check")
        r1 = MonteCarloSimulator(100).simulate_deal(params)

        seed_engine(run_id="determinism_check")
        r2 = MonteCarloSimulator(100).simulate_deal(params)

        assert r1["irr"]["median"] == r2["irr"]["median"], \
            f"IRR mismatch: {r1['irr']['median']} != {r2['irr']['median']}"
        assert r1["probability_analysis"]["prob_loss"] == r2["probability_analysis"]["prob_loss"]
    check("4. Determinism (same seed = same output)", test_determinism)

    # ── 5. Correlated MC ──
    def test_correlated_mc():
        from determinism import seed_engine
        from seal_ceca import MonteCarloSimulator

        seed_engine(run_id="corr_mc_smoke")
        mc = MonteCarloSimulator(500)
        assert mc._correlated_engine is not None, "Correlated engine not initialized"

        result = mc.simulate_deal({
            "purchase_price": 2500000,
            "noi": {"point": 195000, "low": 155000, "high": 235000},
            "loan_rate": {"point": 7.0, "low": 5.5, "high": 9.0},
            "exit_cap": {"point": 7.5, "low": 5.5, "high": 10.0},
        })
        assert result["correlated_draws"] is True
        assert result["simulations"] == 500
        assert result["irr"]["median"] != 0
    check("5. Correlated Monte Carlo", test_correlated_mc)

    # ── 6. Capital stack waterfall ──
    def test_capital_stack():
        from capital_stack import CapitalStack, DebtTranche, EquityTranche, compute_waterfall

        stack = CapitalStack(
            purchase_price=2_500_000,
            debt_tranches=[DebtTranche("Senior", 1_875_000, rate=0.07, term_years=25, priority=1)],
            equity_tranches=[EquityTranche("Equity", 625_000, preferred_return=0.08)],
        )
        wf = compute_waterfall(stack, noi=195_000, exit_value=2_800_000, hold_years=5)
        assert wf["deal"]["overall_dscr"] > 1.0
        assert wf["deal"]["equity_irr"] is not None
        assert "Senior" in wf["tranches"]
    check("6. Capital stack waterfall", test_capital_stack)

    # ── 7. Schema extra field rejection ──
    def test_schema_rejection():
        from schemas import validate_agent_output
        output = {
            "recommendation": "GO", "confidence": 0.8,
            "secret_override": "HACK", "shadow_data": 42,
        }
        result, cleaned = validate_agent_output("underwriting_analyst", output)
        assert "secret_override" not in cleaned
        assert "shadow_data" not in cleaned
    check("7. Schema extra field rejection", test_schema_rejection)

    # ── 8. Ledger integrity ──
    def test_ledger():
        from ledger import HashChainedLedger
        ledger = HashChainedLedger()
        ledger.append("tool_result", {"tool": "census"})
        ledger.append("evidence", {"var": "population", "value": 47000})
        ledger.append("decision", {"decision": "GO"})
        assert ledger.verify() is True

        # Tamper and detect
        ledger._entries[1].content = "TAMPERED"
        assert ledger.verify() is False
        assert ledger.find_tamper() == 1
    check("8. Ledger verify + tamper detect", test_ledger)

    # ── 9. RBAC ──
    def test_rbac():
        from security import RBACManager, Role, AccessDenied
        rbac = RBACManager()
        rbac.add_user("admin", "Admin", Role.ADMIN, password="pw")
        rbac.add_user("viewer", "Viewer", Role.AUDITOR, password="pw")

        assert rbac.authenticate("admin", "pw") is not None
        assert rbac.authenticate("admin", "wrong") is None

        # Admin can manage users, auditor cannot
        rbac.check_permission("admin", "manage_users")
        try:
            rbac.require_permission("viewer", "manage_users")
            assert False, "Should have raised"
        except AccessDenied:
            pass
    check("9. RBAC permissions", test_rbac)

    # ── 10. Cost-aware forager ──
    def test_forager():
        from cost_aware_forager import CostAwareForager, ForagingBudget
        from active_inference import BeliefState

        beliefs = BeliefState({"address": "123 Main", "price": 2500000})
        forager = CostAwareForager(budget=ForagingBudget(max_cost_usd=5.0))
        ranked = forager.rank_actions_costed(
            beliefs,
            ["census_demographics", "traffic_counts", "pull_comps", "fred_economic_data"],
        )
        assert len(ranked) > 0
        # Should have cost, EIG, and combined score
        assert "eig_per_dollar" in ranked[0]
        assert "combined_score" in ranked[0]
        assert ranked[0]["combined_score"] >= ranked[-1]["combined_score"]
    check("10. Cost-aware forager ranking", test_forager)

    # ── Summary ──
    elapsed = time.perf_counter() - start
    passed = sum(1 for _, ok, _, _ in results if ok)
    failed = sum(1 for _, ok, _, _ in results if not ok)

    print(f"\n{'=' * 60}")
    print(f"  {passed}/{len(results)} passed, {failed} failed  ({elapsed:.1f}s)")
    print(f"{'=' * 60}")

    if failed:
        print("\nFAILURES:")
        for name, ok, ms, err in results:
            if not ok:
                print(f"  {name}: {err}")
        return False

    return True


if __name__ == "__main__":
    ok = smoke_test()
    sys.exit(0 if ok else 1)
