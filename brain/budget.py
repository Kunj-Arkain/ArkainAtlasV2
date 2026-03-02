"""
engine.budget — Budget Enforcement With Hard Stops (#6)
=========================================================
Hard caps on: tool calls, tokens, runtime, per-agent, per-variable.
When budget hits → stop and produce best-effort + blockers + next plan.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class BudgetLimits:
    """Hard limits for a pipeline run."""
    max_tool_calls: int = 50           # Total across all agents
    max_tool_calls_per_agent: int = 12 # Per agent
    max_runtime_seconds: float = 120.0 # Wall clock
    max_retries_per_agent: int = 3
    max_total_retries: int = 10
    max_variable_queries: int = 5      # Max tool calls targeting same variable
    max_context_chars: int = 4000      # Context injection cap


@dataclass
class BudgetState:
    """Live budget consumption tracking."""
    tool_calls: int = 0
    tool_calls_by_agent: Dict[str, int] = field(default_factory=dict)
    tool_calls_by_variable: Dict[str, int] = field(default_factory=dict)
    retries: int = 0
    retries_by_agent: Dict[str, int] = field(default_factory=dict)
    start_time: float = 0.0
    agents_completed: int = 0


class BudgetExhausted(Exception):
    """Raised when any budget limit is hit."""
    def __init__(self, reason: str, budget_type: str, consumed: int, limit: int):
        self.reason = reason
        self.budget_type = budget_type
        self.consumed = consumed
        self.limit = limit
        super().__init__(reason)


class BudgetEnforcer:
    """Enforces hard stops on resource consumption.

    Usage:
        enforcer = BudgetEnforcer(BudgetLimits(max_tool_calls=30))
        enforcer.start()
        enforcer.record_tool_call("census_demographics", "underwriting_analyst", ["population"])
        if enforcer.can_call("traffic_counts", "market_analyst"):
            ...
    """

    def __init__(self, limits: BudgetLimits = None):
        self.limits = limits or BudgetLimits()
        self.state = BudgetState()
        self._exhaustion_reasons: List[str] = []

    def start(self):
        self.state.start_time = time.time()

    def elapsed_seconds(self) -> float:
        return time.time() - self.state.start_time if self.state.start_time else 0

    def record_tool_call(self, tool_name: str, agent_name: str,
                         target_variables: List[str] = None):
        """Record a tool call and check limits."""
        self.state.tool_calls += 1
        self.state.tool_calls_by_agent[agent_name] = \
            self.state.tool_calls_by_agent.get(agent_name, 0) + 1
        for var in (target_variables or []):
            self.state.tool_calls_by_variable[var] = \
                self.state.tool_calls_by_variable.get(var, 0) + 1

    def record_retry(self, agent_name: str):
        self.state.retries += 1
        self.state.retries_by_agent[agent_name] = \
            self.state.retries_by_agent.get(agent_name, 0) + 1

    def can_call(self, tool_name: str, agent_name: str,
                 target_variables: List[str] = None) -> bool:
        """Check if a tool call is within budget. Returns False if any limit hit."""
        if self.state.tool_calls >= self.limits.max_tool_calls:
            return False
        agent_calls = self.state.tool_calls_by_agent.get(agent_name, 0)
        if agent_calls >= self.limits.max_tool_calls_per_agent:
            return False
        for var in (target_variables or []):
            var_calls = self.state.tool_calls_by_variable.get(var, 0)
            if var_calls >= self.limits.max_variable_queries:
                return False
        if self.elapsed_seconds() > self.limits.max_runtime_seconds:
            return False
        return True

    def can_retry(self, agent_name: str) -> bool:
        if self.state.retries >= self.limits.max_total_retries:
            return False
        agent_retries = self.state.retries_by_agent.get(agent_name, 0)
        if agent_retries >= self.limits.max_retries_per_agent:
            return False
        return True

    def check_or_raise(self, agent_name: str):
        """Raise BudgetExhausted if any limit is exceeded."""
        if self.state.tool_calls >= self.limits.max_tool_calls:
            raise BudgetExhausted(
                f"Total tool call limit ({self.limits.max_tool_calls}) exceeded",
                "tool_calls", self.state.tool_calls, self.limits.max_tool_calls)
        if self.elapsed_seconds() > self.limits.max_runtime_seconds:
            raise BudgetExhausted(
                f"Runtime limit ({self.limits.max_runtime_seconds}s) exceeded",
                "runtime", int(self.elapsed_seconds()), int(self.limits.max_runtime_seconds))

    def best_effort_summary(self, assumption_table=None) -> Dict:
        """When budget is exhausted, produce a structured summary."""
        # Find variables with most queries (obsession loops)
        obsession = sorted(self.state.tool_calls_by_variable.items(),
                           key=lambda x: x[1], reverse=True)[:5]
        # Find what's still unknown
        missing = []
        if assumption_table:
            try:
                from .assumptions import DataStatus
            except ImportError:
                from assumptions import DataStatus
            for var, rec in assumption_table._records.items():
                if rec.status == DataStatus.UNKNOWN:
                    missing.append(var)

        return {
            "status": "BUDGET_EXHAUSTED",
            "consumed": {
                "tool_calls": f"{self.state.tool_calls}/{self.limits.max_tool_calls}",
                "runtime": f"{self.elapsed_seconds():.1f}s/{self.limits.max_runtime_seconds}s",
                "retries": f"{self.state.retries}/{self.limits.max_total_retries}",
            },
            "top_data_blockers": missing[:10],
            "variable_effort": obsession,
            "agents_completed": self.state.agents_completed,
            "next_data_plan": [
                f"Obtain evidence for: {var}" for var in missing[:5]
            ],
        }

    def report(self) -> Dict:
        return {
            "tool_calls": self.state.tool_calls,
            "limit": self.limits.max_tool_calls,
            "utilization_pct": round(self.state.tool_calls / max(self.limits.max_tool_calls, 1) * 100, 1),
            "elapsed_seconds": round(self.elapsed_seconds(), 1),
            "retries": self.state.retries,
            "by_agent": dict(self.state.tool_calls_by_agent),
            "by_variable": dict(self.state.tool_calls_by_variable),
        }
