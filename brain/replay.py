"""
engine.replay — Replay Engine / Frozen Runs (#7)
===================================================
Run modes:
  LIVE:   call tools for fresh evidence
  FROZEN: replay from recorded tool + model outputs
  DIFF:   compare run A vs run B and show what changed

Every LIVE run auto-records a ReplayLog. Every FROZEN run replays
from a saved log. DIFF mode aligns two logs and reports deltas.
This is the "audit grade" backbone.
"""

from __future__ import annotations
import json
import time
import hashlib
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum


class RunMode(Enum):
    LIVE = "live"       # Call real tools, record everything
    FROZEN = "frozen"   # Replay from recorded outputs
    DIFF = "diff"       # Compare two runs


@dataclass
class ReplayEntry:
    """Single recorded event in a replay log."""
    sequence: int                # Monotonic event counter
    timestamp: float
    event_type: str              # "tool_call", "tool_result", "agent_call", "agent_output",
                                 # "belief_update", "convergence_check", "decision"
    agent_name: str = ""
    tool_name: str = ""
    parameters: Dict = field(default_factory=dict)
    result: Any = None
    metadata: Dict = field(default_factory=dict)

    def content_hash(self) -> str:
        """Hash of the meaningful content (for diff comparison)."""
        content = json.dumps({
            "event_type": self.event_type, "agent_name": self.agent_name,
            "tool_name": self.tool_name, "result": self.result,
        }, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        return {
            "seq": self.sequence, "ts": self.timestamp,
            "type": self.event_type, "agent": self.agent_name,
            "tool": self.tool_name, "params": self.parameters,
            "result": self.result, "meta": self.metadata,
            "hash": self.content_hash(),
        }


class ReplayLog:
    """Records every action in a pipeline run for replay/audit."""

    def __init__(self, run_id: str = None):
        self.run_id = run_id or f"run_{int(time.time())}"
        self.mode: RunMode = RunMode.LIVE
        self.entries: List[ReplayEntry] = []
        self._seq = 0
        self.started_at: float = 0.0
        self.finished_at: float = 0.0
        self.deal_data: Dict = {}
        self.config: Dict = {}

    def start(self, deal_data: Dict, config: Dict = None):
        self.started_at = time.time()
        self.deal_data = copy.deepcopy(deal_data)
        self.config = config or {}

    def record(self, event_type: str, agent_name: str = "",
               tool_name: str = "", parameters: Dict = None,
               result: Any = None, metadata: Dict = None) -> ReplayEntry:
        """Record an event."""
        self._seq += 1
        entry = ReplayEntry(
            sequence=self._seq, timestamp=time.time(),
            event_type=event_type, agent_name=agent_name,
            tool_name=tool_name, parameters=parameters or {},
            result=result, metadata=metadata or {},
        )
        self.entries.append(entry)
        return entry

    def record_tool_call(self, tool_name: str, params: Dict,
                         result: Any, agent_name: str = ""):
        self.record("tool_call", agent_name=agent_name,
                    tool_name=tool_name, parameters=params)
        self.record("tool_result", agent_name=agent_name,
                    tool_name=tool_name, result=result)

    def record_agent_output(self, agent_name: str, output: Dict,
                            attempt: int = 0):
        self.record("agent_output", agent_name=agent_name,
                    result=output, metadata={"attempt": attempt})

    def finish(self, final_output: Dict = None):
        self.finished_at = time.time()
        if final_output:
            self.record("decision", result=final_output)

    # ── Lookup for frozen replay ──

    def get_tool_result(self, tool_name: str, params: Dict) -> Optional[Any]:
        """Find a recorded tool result matching this call (for FROZEN mode)."""
        for i, entry in enumerate(self.entries):
            if entry.event_type == "tool_call" and entry.tool_name == tool_name:
                if i + 1 < len(self.entries) and self.entries[i+1].event_type == "tool_result":
                    return self.entries[i+1].result
        return None

    def get_agent_output(self, agent_name: str, attempt: int = 0) -> Optional[Dict]:
        """Find a recorded agent output (for FROZEN mode)."""
        for entry in self.entries:
            if (entry.event_type == "agent_output" and
                entry.agent_name == agent_name and
                entry.metadata.get("attempt") == attempt):
                return entry.result
        return None

    # ── Serialization ──

    def save(self, path: str):
        data = {
            "run_id": self.run_id, "mode": self.mode.value,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "deal_data": self.deal_data, "config": self.config,
            "entry_count": len(self.entries),
            "entries": [e.to_dict() for e in self.entries],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "ReplayLog":
        with open(path) as f:
            data = json.load(f)
        log = cls(run_id=data["run_id"])
        log.mode = RunMode(data.get("mode", "frozen"))
        log.started_at = data["started_at"]
        log.finished_at = data["finished_at"]
        log.deal_data = data["deal_data"]
        log.config = data.get("config", {})
        for ed in data["entries"]:
            log._seq += 1
            log.entries.append(ReplayEntry(
                sequence=ed["seq"], timestamp=ed["ts"],
                event_type=ed["type"], agent_name=ed.get("agent", ""),
                tool_name=ed.get("tool", ""),
                parameters=ed.get("params", {}),
                result=ed.get("result"), metadata=ed.get("meta", {}),
            ))
        return log

    def summary(self) -> Dict:
        tool_calls = sum(1 for e in self.entries if e.event_type == "tool_call")
        agent_outputs = sum(1 for e in self.entries if e.event_type == "agent_output")
        agents = list(set(e.agent_name for e in self.entries if e.agent_name))
        return {
            "run_id": self.run_id, "mode": self.mode.value,
            "duration_sec": round(self.finished_at - self.started_at, 2) if self.finished_at else 0,
            "total_events": len(self.entries),
            "tool_calls": tool_calls, "agent_outputs": agent_outputs,
            "agents": agents,
        }


class ReplayDiff:
    """Compare two replay logs and show what changed."""

    @staticmethod
    def diff(log_a: ReplayLog, log_b: ReplayLog) -> Dict:
        """Compare run A vs run B. Returns structured diff."""
        diffs = {
            "run_a": log_a.run_id, "run_b": log_b.run_id,
            "deal_data_changed": log_a.deal_data != log_b.deal_data,
            "tool_call_diffs": [],
            "agent_output_diffs": [],
            "decision_diff": None,
        }

        # Compare tool results by tool name
        tools_a = {e.tool_name: e for e in log_a.entries if e.event_type == "tool_result"}
        tools_b = {e.tool_name: e for e in log_b.entries if e.event_type == "tool_result"}
        all_tools = set(tools_a.keys()) | set(tools_b.keys())
        for tool in sorted(all_tools):
            ea, eb = tools_a.get(tool), tools_b.get(tool)
            if ea and eb:
                if ea.content_hash() != eb.content_hash():
                    diffs["tool_call_diffs"].append({
                        "tool": tool, "change": "modified",
                        "a_hash": ea.content_hash(), "b_hash": eb.content_hash(),
                        "a_result": ea.result, "b_result": eb.result,
                    })
            elif ea:
                diffs["tool_call_diffs"].append({"tool": tool, "change": "removed_in_b"})
            else:
                diffs["tool_call_diffs"].append({"tool": tool, "change": "added_in_b"})

        # Compare agent outputs
        agents_a = {e.agent_name: e for e in log_a.entries if e.event_type == "agent_output"}
        agents_b = {e.agent_name: e for e in log_b.entries if e.event_type == "agent_output"}
        all_agents = set(agents_a.keys()) | set(agents_b.keys())
        for agent in sorted(all_agents):
            ea, eb = agents_a.get(agent), agents_b.get(agent)
            if ea and eb:
                # Find numerical differences
                if isinstance(ea.result, dict) and isinstance(eb.result, dict):
                    field_diffs = {}
                    all_fields = set(ea.result.keys()) | set(eb.result.keys())
                    for fld in all_fields:
                        va, vb = ea.result.get(fld), eb.result.get(fld)
                        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                            if va != vb:
                                pct = abs(va - vb) / max(abs(va), 1) * 100
                                field_diffs[fld] = {"a": va, "b": vb, "delta_pct": round(pct, 1)}
                        elif va != vb:
                            field_diffs[fld] = {"a": va, "b": vb}
                    if field_diffs:
                        diffs["agent_output_diffs"].append({"agent": agent, "fields": field_diffs})

        # Compare final decisions
        dec_a = [e for e in log_a.entries if e.event_type == "decision"]
        dec_b = [e for e in log_b.entries if e.event_type == "decision"]
        if dec_a and dec_b:
            da, db = dec_a[-1].result, dec_b[-1].result
            if da != db:
                diffs["decision_diff"] = {"a": da, "b": db}

        diffs["total_changes"] = (
            len(diffs["tool_call_diffs"]) +
            len(diffs["agent_output_diffs"]) +
            (1 if diffs["decision_diff"] else 0)
        )
        return diffs
