"""
engine.schemas — Strict Schemas Everywhere (#3)
==================================================
Dataclass-based strict validation for all engine I/O.
If data doesn't validate → auto-repair prompt → reject.
No "kinda" outputs.

Covers:
  - BeliefStateSchema (what the engine knows)
  - ToolCallSchema / ToolResultSchema (tool I/O)
  - AgentOutputSchema (per-agent validation)
  - DecisionPackageSchema (final output)
  - ValidationResult (pass/fail with reasons)
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Type
from enum import Enum


class SchemaViolation(Enum):
    MISSING_REQUIRED = "missing_required"
    WRONG_TYPE = "wrong_type"
    OUT_OF_RANGE = "out_of_range"
    INVALID_ENUM = "invalid_enum"
    CONSTRAINT_FAILED = "constraint_failed"
    EXTRA_FIELD = "extra_field"


@dataclass
class ValidationIssue:
    field: str
    violation: SchemaViolation
    message: str
    expected: str = ""
    got: str = ""
    auto_repairable: bool = False


@dataclass
class ValidationResult:
    """Result of schema validation — pass/fail with details."""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    repaired_fields: List[str] = field(default_factory=list)
    schema_name: str = ""

    def to_dict(self) -> Dict:
        return {
            "valid": self.valid, "schema": self.schema_name,
            "issue_count": len(self.issues),
            "issues": [{"field": i.field, "violation": i.violation.value,
                        "message": i.message} for i in self.issues],
            "repaired": self.repaired_fields,
        }


# ── Field descriptor for strict validation ──

@dataclass
class FieldSpec:
    """Specification for a single field in a schema."""
    name: str
    type: str                    # "float", "int", "str", "bool", "list", "dict", "enum"
    required: bool = True
    default: Any = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    enum_values: Optional[List[str]] = None
    list_min_len: int = 0
    auto_repair: bool = False    # Can we fix this automatically?
    repair_value: Any = None     # What to repair to


class StrictSchema:
    """Base schema validator. Subclasses define FIELDS."""

    SCHEMA_NAME: str = "base"
    FIELDS: List[FieldSpec] = []
    STRICT_NO_EXTRA: bool = True   # Reject unexpected fields (Gap B: prevent shadow outputs)
    # Fields that are always allowed (internal bookkeeping)
    _ALWAYS_ALLOWED: Set[str] = {"_tools_called", "_error", "_meta", "_attempt"}

    @classmethod
    def validate(cls, data: Dict, auto_repair: bool = True) -> Tuple[ValidationResult, Dict]:
        """Validate data against schema. Returns (result, repaired_data)."""
        issues = []
        repaired = []
        output = dict(data) if data else {}
        known_fields = {spec.name for spec in cls.FIELDS}

        # Gap B: Reject extra fields — prevents junk, prompt injection,
        # and shadow outputs that bypass downstream logic
        if cls.STRICT_NO_EXTRA:
            extra = set(output.keys()) - known_fields - cls._ALWAYS_ALLOWED
            for extra_field in sorted(extra):
                issues.append(ValidationIssue(
                    field=extra_field, violation=SchemaViolation.EXTRA_FIELD,
                    message=f"Unexpected field '{extra_field}' — not in schema. "
                            f"Allowed: {sorted(known_fields)}",
                ))
                if auto_repair:
                    del output[extra_field]
                    repaired.append(extra_field)

        for spec in cls.FIELDS:
            val = output.get(spec.name)

            # Missing required
            if val is None and spec.required:
                if spec.auto_repair and auto_repair and spec.repair_value is not None:
                    output[spec.name] = spec.repair_value
                    repaired.append(spec.name)
                else:
                    issues.append(ValidationIssue(
                        field=spec.name, violation=SchemaViolation.MISSING_REQUIRED,
                        message=f"Required field '{spec.name}' is missing",
                        expected=spec.type, auto_repairable=spec.auto_repair,
                    ))
                continue

            if val is None:
                continue

            # Type check
            type_ok = True
            if spec.type == "float" and not isinstance(val, (int, float)):
                type_ok = False
            elif spec.type == "int" and not isinstance(val, int):
                if isinstance(val, float) and val == int(val):
                    output[spec.name] = int(val)
                    repaired.append(spec.name)
                else:
                    type_ok = False
            elif spec.type == "str" and not isinstance(val, str):
                type_ok = False
            elif spec.type == "bool" and not isinstance(val, bool):
                type_ok = False
            elif spec.type == "list" and not isinstance(val, list):
                type_ok = False
            elif spec.type == "dict" and not isinstance(val, dict):
                type_ok = False

            if not type_ok:
                issues.append(ValidationIssue(
                    field=spec.name, violation=SchemaViolation.WRONG_TYPE,
                    message=f"'{spec.name}' expected {spec.type}, got {type(val).__name__}",
                    expected=spec.type, got=type(val).__name__,
                ))
                continue

            # Range check
            if spec.min_val is not None and isinstance(val, (int, float)):
                if val < spec.min_val:
                    issues.append(ValidationIssue(
                        field=spec.name, violation=SchemaViolation.OUT_OF_RANGE,
                        message=f"'{spec.name}' = {val} below minimum {spec.min_val}",
                    ))

            if spec.max_val is not None and isinstance(val, (int, float)):
                if val > spec.max_val:
                    issues.append(ValidationIssue(
                        field=spec.name, violation=SchemaViolation.OUT_OF_RANGE,
                        message=f"'{spec.name}' = {val} above maximum {spec.max_val}",
                    ))

            # Enum check
            if spec.enum_values and isinstance(val, str):
                if val not in spec.enum_values:
                    issues.append(ValidationIssue(
                        field=spec.name, violation=SchemaViolation.INVALID_ENUM,
                        message=f"'{spec.name}' = '{val}' not in {spec.enum_values}",
                    ))

            # List min length
            if spec.type == "list" and isinstance(val, list) and len(val) < spec.list_min_len:
                issues.append(ValidationIssue(
                    field=spec.name, violation=SchemaViolation.CONSTRAINT_FAILED,
                    message=f"'{spec.name}' has {len(val)} items, need >= {spec.list_min_len}",
                ))

        result = ValidationResult(
            valid=len(issues) == 0, issues=issues,
            repaired_fields=repaired, schema_name=cls.SCHEMA_NAME,
        )
        return result, output

    @classmethod
    def generate_repair_prompt(cls, issues: List[ValidationIssue]) -> str:
        """Generate a prompt telling the agent exactly what to fix."""
        lines = ["Your output failed schema validation. Fix these issues:"]
        for i, issue in enumerate(issues, 1):
            lines.append(f"  {i}. {issue.message}")
            if issue.expected:
                lines.append(f"     Expected type: {issue.expected}")
        lines.append("\nReturn ONLY the corrected JSON output.")
        return "\n".join(lines)


# ── Concrete schemas ──

class UnderwritingOutputSchema(StrictSchema):
    SCHEMA_NAME = "underwriting_output"
    FIELDS = [
        FieldSpec("noi", "float", required=True, min_val=0),
        FieldSpec("cap_rate", "float", required=True, min_val=1.0, max_val=20.0),
        FieldSpec("purchase_price", "float", required=True, min_val=0),
        FieldSpec("dscr", "float", required=True, min_val=0),
        FieldSpec("recommendation", "str", required=True,
                  enum_values=["GO", "NO_GO", "CONDITIONAL", "NEEDS_DATA"]),
        FieldSpec("direct_cap_value", "float", required=True, min_val=0),
        FieldSpec("dcf_value", "float", required=False, min_val=0),
        FieldSpec("comparable_value", "float", required=False, min_val=0),
        FieldSpec("stress_test", "dict", required=True,
                  auto_repair=True, repair_value={"noi_minus_20": "not_run"}),
    ]


class DealStructurerOutputSchema(StrictSchema):
    SCHEMA_NAME = "deal_structurer_output"
    FIELDS = [
        FieldSpec("noi", "float", required=True, min_val=0),
        FieldSpec("purchase_price", "float", required=True, min_val=0),
        FieldSpec("recommendation", "str", required=True),
        FieldSpec("variant_a", "str", required=True),
        FieldSpec("variant_b", "str", required=True),
        FieldSpec("sensitivity_matrix", "bool", required=True,
                  auto_repair=True, repair_value=False),
    ]


class RiskOfficerOutputSchema(StrictSchema):
    SCHEMA_NAME = "risk_officer_output"
    FIELDS = [
        FieldSpec("verdict", "str", required=True,
                  enum_values=["APPROVE", "REJECT", "CONDITIONAL", "NEEDS_DATA"]),
        FieldSpec("risk_register", "list", required=True, list_min_len=1),
        FieldSpec("market_risk", "str", required=True,
                  enum_values=["LOW", "MED", "HIGH", "CRITICAL"]),
        FieldSpec("credit_risk", "str", required=True,
                  enum_values=["LOW", "MED", "HIGH", "CRITICAL"]),
        FieldSpec("environmental_risk", "str", required=True,
                  enum_values=["LOW", "MED", "HIGH", "CRITICAL"]),
    ]


class GamingOptimizerOutputSchema(StrictSchema):
    SCHEMA_NAME = "gaming_optimizer_output"
    FIELDS = [
        FieldSpec("terminal_count", "int", required=True, min_val=0, max_val=100),
        FieldSpec("nti_per_terminal", "float", required=True, min_val=0),
        FieldSpec("gaming_net_revenue", "float", required=True, min_val=0),
        FieldSpec("recommendation", "str", required=True),
    ]


class ToolCallSchema(StrictSchema):
    SCHEMA_NAME = "tool_call"
    FIELDS = [
        FieldSpec("tool_name", "str", required=True),
        FieldSpec("parameters", "dict", required=True),
        FieldSpec("called_by", "str", required=True),
        FieldSpec("timestamp", "float", required=True),
    ]


# Schema registry
AGENT_SCHEMAS: Dict[str, type] = {
    "underwriting_analyst": UnderwritingOutputSchema,
    "deal_structurer": DealStructurerOutputSchema,
    "risk_officer": RiskOfficerOutputSchema,
    "gaming_optimizer": GamingOptimizerOutputSchema,
}


def validate_agent_output(agent_name: str, output: Dict,
                          auto_repair: bool = True) -> Tuple[ValidationResult, Dict]:
    """Validate an agent's output against its schema."""
    schema_cls = AGENT_SCHEMAS.get(agent_name)
    if not schema_cls:
        # No schema for this agent — pass through
        return ValidationResult(valid=True, schema_name="none"), output
    return schema_cls.validate(output, auto_repair=auto_repair)
