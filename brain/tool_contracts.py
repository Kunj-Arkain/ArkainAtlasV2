"""
engine.tool_contracts — Tool Layer With Real Contracts (#8)
============================================================
Tools are real functions that return structured results.
They NEVER fabricate. They return explicit failure states.

ToolOutcome: OK | NOT_FOUND | FORBIDDEN | RATE_LIMITED | NEEDS_AUTH | UPSTREAM_ERROR
ToolContract: defines inputs/outputs/evidence grade per tool
ToolRegistry: central registry of all tools + their contracts
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

try:
    from .evidence import EvidenceRef, EvidenceGrade, EvidenceMethod, make_evidence_ref
except ImportError:
    from evidence import EvidenceRef, EvidenceGrade, EvidenceMethod, make_evidence_ref


class ToolStatus(Enum):
    OK = "OK"
    NOT_FOUND = "NOT_FOUND"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMITED = "RATE_LIMITED"
    NEEDS_AUTH = "NEEDS_AUTH"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    TIMEOUT = "TIMEOUT"
    NO_DATA = "NO_DATA"            # Source exists but has no data for query
    PARTIAL = "PARTIAL"            # Some fields returned, some missing


@dataclass
class ToolResult:
    """Structured result from any tool call. Never ambiguous."""
    tool_name: str
    status: ToolStatus
    data: Optional[Dict] = None
    error_message: str = ""
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    call_duration_ms: int = 0
    called_at: float = 0.0
    parameters: Dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in (ToolStatus.OK, ToolStatus.PARTIAL)

    def to_dict(self) -> Dict:
        d = {
            "tool": self.tool_name, "status": self.status.value,
            "ok": self.ok, "duration_ms": self.call_duration_ms,
            "called_at": self.called_at,
        }
        if self.data:
            d["data"] = self.data
        if self.error_message:
            d["error"] = self.error_message
        if self.evidence_refs:
            d["evidence_count"] = len(self.evidence_refs)
        return d


@dataclass
class ToolContract:
    """Defines what a tool accepts, returns, and how reliable it is."""
    name: str
    description: str
    required_params: List[str]           # Must be provided
    optional_params: List[str] = field(default_factory=list)
    output_variables: List[str] = field(default_factory=list)  # What vars it informs
    evidence_grade: EvidenceGrade = EvidenceGrade.C
    evidence_method: EvidenceMethod = EvidenceMethod.API_CALL
    max_latency_ms: int = 5000
    requires_auth: bool = False
    is_live: bool = True                 # False = offline/mock


class ToolRegistry:
    """Central registry of all tools with their contracts.

    Tools register here. The orchestrator queries the registry
    to know what's available, what each tool provides, and
    what failure modes to expect.
    """

    def __init__(self):
        self._contracts: Dict[str, ToolContract] = {}
        self._implementations: Dict[str, Callable] = {}
        self._call_log: List[Dict] = []

    def register(self, contract: ToolContract, implementation: Callable = None):
        """Register a tool with its contract."""
        self._contracts[contract.name] = contract
        if implementation:
            self._implementations[contract.name] = implementation

    def get_contract(self, tool_name: str) -> Optional[ToolContract]:
        return self._contracts.get(tool_name)

    def available_tools(self) -> List[str]:
        return list(self._contracts.keys())

    def tools_for_variable(self, variable: str) -> List[str]:
        """Which tools can provide data for this variable?"""
        return [name for name, c in self._contracts.items()
                if variable in c.output_variables]

    def call(self, tool_name: str, params: Dict,
             called_by: str = "orchestrator") -> ToolResult:
        """Call a tool through its contract. Returns structured ToolResult."""
        start = time.time()
        contract = self._contracts.get(tool_name)
        if not contract:
            return ToolResult(tool_name=tool_name, status=ToolStatus.NOT_FOUND,
                              error_message=f"Tool '{tool_name}' not registered",
                              called_at=start, parameters=params)

        # Validate required params
        missing = [p for p in contract.required_params if p not in params]
        if missing:
            return ToolResult(tool_name=tool_name, status=ToolStatus.UPSTREAM_ERROR,
                              error_message=f"Missing required params: {missing}",
                              called_at=start, parameters=params)

        # Execute
        impl = self._implementations.get(tool_name)
        if not impl:
            return ToolResult(tool_name=tool_name, status=ToolStatus.NOT_FOUND,
                              error_message=f"No implementation for '{tool_name}'",
                              called_at=start, parameters=params)

        try:
            raw_result = impl(**params)
            elapsed = int((time.time() - start) * 1000)

            if raw_result is None:
                return ToolResult(tool_name=tool_name, status=ToolStatus.NO_DATA,
                                  call_duration_ms=elapsed, called_at=start,
                                  parameters=params)

            # Build evidence refs for each output variable
            evidence_refs = []
            data = raw_result if isinstance(raw_result, dict) else {"value": raw_result}
            for var in contract.output_variables:
                if var in data:
                    ref = make_evidence_ref(
                        source_name=tool_name, variable=var,
                        value=data[var], method=contract.evidence_method,
                        grade=contract.evidence_grade,
                        confidence=0.7 if contract.evidence_grade.value in ("A", "B") else 0.5,
                        raw_payload=json.dumps(data)[:500],
                    )
                    evidence_refs.append(ref)

            return ToolResult(
                tool_name=tool_name, status=ToolStatus.OK,
                data=data, evidence_refs=evidence_refs,
                call_duration_ms=elapsed, called_at=start,
                parameters=params,
            )

        except PermissionError:
            return ToolResult(tool_name=tool_name, status=ToolStatus.FORBIDDEN,
                              error_message="Permission denied",
                              call_duration_ms=int((time.time()-start)*1000),
                              called_at=start, parameters=params)
        except TimeoutError:
            return ToolResult(tool_name=tool_name, status=ToolStatus.TIMEOUT,
                              error_message="Tool timed out",
                              call_duration_ms=int((time.time()-start)*1000),
                              called_at=start, parameters=params)
        except Exception as e:
            return ToolResult(tool_name=tool_name, status=ToolStatus.UPSTREAM_ERROR,
                              error_message=str(e)[:500],
                              call_duration_ms=int((time.time()-start)*1000),
                              called_at=start, parameters=params)
        finally:
            self._call_log.append({
                "tool": tool_name, "called_by": called_by,
                "params": params, "timestamp": start,
            })

    def call_log(self) -> List[Dict]:
        return list(self._call_log)
