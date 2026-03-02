"""
engine.brain.pipeline_config — Project Pipeline Configuration
================================================================
Toggle agents and features on/off per project. Not every deal
needs all 17 agents — a simple NNN dollar store doesn't need
an MEP engineer or gaming optimizer.

Usage:
    config = PipelineConfig.for_project({
        "property_type": "gas_station",
        "gaming_eligible": True,
        "renovation_planned": True,
        "construction_drawings": False,
    })

    # Check if an agent should run
    if config.is_enabled("gaming_optimizer"):
        run_agent("gaming_optimizer", deal_data)

    # Get list of active agents for this project
    active = config.active_agents()

    # Get active agent pipeline in order
    pipeline = config.active_pipeline()

Presets:
    "quick_screen"   — Scout + Market + Quick Underwrite (3 agents)
    "standard_deal"  — Full financial analysis, no construction (9 agents)
    "gaming_deal"    — Standard + gaming optimization (11 agents)
    "full_build"     — Everything including construction drawings (17 agents)
    "disposition"    — Exit-focused subset (5 agents)
    "custom"         — User picks exactly which agents to run
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ═══════════════════════════════════════════════════════════════
# FEATURE FLAGS
# ═══════════════════════════════════════════════════════════════

@dataclass
class FeatureFlags:
    """Individual feature toggles within the pipeline."""
    # Core analysis
    deal_screening: bool = True
    market_research: bool = True
    underwriting: bool = True
    deal_structuring: bool = True

    # Gaming
    gaming_analysis: bool = False

    # Risk & DD
    risk_assessment: bool = True
    due_diligence: bool = True

    # Legal
    contract_review: bool = False
    tax_planning: bool = False

    # Construction
    renovation_planning: bool = False
    construction_drawings: bool = False
    architectural_set: bool = False
    mep_drawings: bool = False
    structural_drawings: bool = False
    spec_book: bool = False

    # Compliance & Exit
    compliance_docs: bool = False
    exit_planning: bool = False

    # Output formats
    generate_pdf_report: bool = True
    generate_excel_model: bool = False
    generate_cad_files: bool = False
    generate_spec_book: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict) -> "FeatureFlags":
        flags = cls()
        for k, v in d.items():
            if hasattr(flags, k):
                setattr(flags, k, bool(v))
        return flags


# ═══════════════════════════════════════════════════════════════
# AGENT → FEATURE MAPPING
# ═══════════════════════════════════════════════════════════════

# Maps each agent to the feature flag(s) that must be True for it to run
AGENT_FEATURE_MAP = {
    "acquisition_scout":     ["deal_screening"],
    "site_selector":         ["deal_screening", "market_research"],
    "market_analyst":        ["market_research"],
    "underwriting_analyst":  ["underwriting"],
    "deal_structurer":       ["deal_structuring"],
    "gaming_optimizer":      ["gaming_analysis"],
    "risk_officer":          ["risk_assessment"],
    "due_diligence":         ["due_diligence"],
    "contract_redliner":     ["contract_review"],
    "tax_strategist":        ["tax_planning"],
    "renovation_planner":    ["renovation_planning"],
    "compliance_writer":     ["compliance_docs"],
    "exit_strategist":       ["exit_planning"],
    # Construction agents (new)
    "architect":             ["construction_drawings", "architectural_set"],
    "mep_engineer":          ["construction_drawings", "mep_drawings"],
    "structural_engineer":   ["construction_drawings", "structural_drawings"],
    "spec_writer":           ["construction_drawings", "spec_book"],
}

# Canonical execution order
PIPELINE_ORDER = [
    "acquisition_scout",
    "site_selector",
    "market_analyst",
    "underwriting_analyst",
    "deal_structurer",
    "gaming_optimizer",
    "risk_officer",
    "due_diligence",
    "contract_redliner",
    "tax_strategist",
    "renovation_planner",
    "architect",
    "structural_engineer",
    "mep_engineer",
    "spec_writer",
    "compliance_writer",
    "exit_strategist",
]


# ═══════════════════════════════════════════════════════════════
# PRESETS
# ═══════════════════════════════════════════════════════════════

PRESETS: Dict[str, Dict[str, bool]] = {
    "quick_screen": {
        "deal_screening": True,
        "market_research": True,
        "underwriting": True,
    },

    "standard_deal": {
        "deal_screening": True,
        "market_research": True,
        "underwriting": True,
        "deal_structuring": True,
        "risk_assessment": True,
        "due_diligence": True,
        "contract_review": True,
        "tax_planning": True,
        "renovation_planning": True,
        "generate_pdf_report": True,
    },

    "gaming_deal": {
        "deal_screening": True,
        "market_research": True,
        "underwriting": True,
        "deal_structuring": True,
        "gaming_analysis": True,
        "risk_assessment": True,
        "due_diligence": True,
        "contract_review": True,
        "tax_planning": True,
        "renovation_planning": True,
        "compliance_docs": True,
        "generate_pdf_report": True,
    },

    "full_build": {
        "deal_screening": True,
        "market_research": True,
        "underwriting": True,
        "deal_structuring": True,
        "gaming_analysis": True,
        "risk_assessment": True,
        "due_diligence": True,
        "contract_review": True,
        "tax_planning": True,
        "renovation_planning": True,
        "construction_drawings": True,
        "architectural_set": True,
        "mep_drawings": True,
        "structural_drawings": True,
        "spec_book": True,
        "compliance_docs": True,
        "exit_planning": True,
        "generate_pdf_report": True,
        "generate_excel_model": True,
        "generate_cad_files": True,
        "generate_spec_book": True,
    },

    "disposition": {
        "market_research": True,
        "underwriting": True,
        "risk_assessment": True,
        "tax_planning": True,
        "exit_planning": True,
        "generate_pdf_report": True,
    },

    "construction_only": {
        "renovation_planning": True,
        "construction_drawings": True,
        "architectural_set": True,
        "mep_drawings": True,
        "structural_drawings": True,
        "spec_book": True,
        "generate_pdf_report": True,
        "generate_cad_files": True,
        "generate_spec_book": True,
    },
}


# ═══════════════════════════════════════════════════════════════
# PIPELINE CONFIG
# ═══════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Project-level pipeline configuration with agent toggles."""

    project_name: str = ""
    preset: str = "standard_deal"
    flags: FeatureFlags = field(default_factory=FeatureFlags)
    agent_overrides: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_enabled(self, agent_name: str) -> bool:
        """Check if a specific agent should run for this project."""
        # Explicit override takes priority
        if agent_name in self.agent_overrides:
            return self.agent_overrides[agent_name]

        # Check feature flags
        required_features = AGENT_FEATURE_MAP.get(agent_name, [])
        if not required_features:
            return False

        # Agent runs if ANY of its required features are enabled
        return any(getattr(self.flags, feat, False) for feat in required_features)

    def active_agents(self) -> List[str]:
        """Get list of enabled agent names."""
        return [name for name in PIPELINE_ORDER if self.is_enabled(name)]

    def active_pipeline(self) -> List[Dict]:
        """Get ordered list of active agents with their config."""
        from engine.brain.agents import AGENT_ROLES
        result = []
        for i, name in enumerate(PIPELINE_ORDER):
            if self.is_enabled(name) and name in AGENT_ROLES:
                agent = AGENT_ROLES[name]
                result.append({
                    "order": i + 1,
                    "name": name,
                    "role": agent.role,
                    "description": agent.description,
                    "task_count": len(agent.tasks),
                    "tool_count": len(agent.tools),
                    "enabled": True,
                })
        return result

    def disabled_agents(self) -> List[str]:
        """Get list of disabled agent names."""
        return [name for name in PIPELINE_ORDER if not self.is_enabled(name)]

    def enable(self, *agent_names: str) -> "PipelineConfig":
        """Enable specific agents (override)."""
        for name in agent_names:
            self.agent_overrides[name] = True
        return self

    def disable(self, *agent_names: str) -> "PipelineConfig":
        """Disable specific agents (override)."""
        for name in agent_names:
            self.agent_overrides[name] = False
        return self

    def set_feature(self, feature: str, enabled: bool) -> "PipelineConfig":
        """Toggle a feature flag."""
        if hasattr(self.flags, feature):
            setattr(self.flags, feature, enabled)
        return self

    def to_dict(self) -> Dict:
        return {
            "project_name": self.project_name,
            "preset": self.preset,
            "flags": self.flags.to_dict(),
            "agent_overrides": self.agent_overrides,
            "active_agents": self.active_agents(),
            "active_count": len(self.active_agents()),
            "disabled_agents": self.disabled_agents(),
            "disabled_count": len(self.disabled_agents()),
        }

    # ── Factory methods ───────────────────────────────────

    @classmethod
    def from_preset(cls, preset_name: str, project_name: str = "") -> "PipelineConfig":
        """Create config from a named preset."""
        preset_flags = PRESETS.get(preset_name, PRESETS["standard_deal"])
        flags = FeatureFlags.from_dict(preset_flags)
        return cls(project_name=project_name, preset=preset_name, flags=flags)

    @classmethod
    def for_project(cls, project_data: Dict) -> "PipelineConfig":
        """Auto-configure based on project parameters.

        Reads property_type, gaming_eligible, renovation_planned,
        construction_drawings, etc. and enables the right agents.
        """
        ptype = project_data.get("property_type", "")
        gaming = project_data.get("gaming_eligible", False)
        reno = project_data.get("renovation_planned", False)
        construction = project_data.get("construction_drawings", False)
        exit_plan = project_data.get("exit_planning", False)
        preset = project_data.get("preset", "")

        # Start from preset if specified
        if preset:
            return cls.from_preset(preset, project_data.get("project_name", ""))

        # Auto-detect appropriate config
        flags = FeatureFlags(
            deal_screening=True,
            market_research=True,
            underwriting=True,
            deal_structuring=True,
            risk_assessment=True,
            due_diligence=True,
            generate_pdf_report=True,
        )

        # Gaming — enable if explicitly requested or if property type suggests it
        gaming_types = {"gas_station", "bar", "restaurant", "truck_stop", "tavern", "convenience_store"}
        if gaming or ptype in gaming_types:
            flags.gaming_analysis = True
            flags.compliance_docs = True

        # Contract review — enable for most deals
        if ptype:
            flags.contract_review = True
            flags.tax_planning = True

        # Renovation
        if reno:
            flags.renovation_planning = True

        # Full construction drawings
        if construction:
            flags.renovation_planning = True
            flags.construction_drawings = True
            flags.architectural_set = True
            flags.mep_drawings = True
            flags.structural_drawings = True
            flags.spec_book = True
            flags.generate_cad_files = True
            flags.generate_spec_book = True
            flags.generate_excel_model = True

        # Exit
        if exit_plan:
            flags.exit_planning = True

        return cls(
            project_name=project_data.get("project_name", ""),
            preset="auto",
            flags=flags,
        )

    @classmethod
    def all_on(cls, project_name: str = "") -> "PipelineConfig":
        """Everything enabled — full 17-agent pipeline."""
        return cls.from_preset("full_build", project_name)

    @classmethod
    def minimal(cls, project_name: str = "") -> "PipelineConfig":
        """Minimum viable — just screening."""
        return cls.from_preset("quick_screen", project_name)


# ═══════════════════════════════════════════════════════════════
# API HELPERS
# ═══════════════════════════════════════════════════════════════

def list_presets() -> Dict[str, Dict]:
    """List all available presets with their active agent counts."""
    result = {}
    for name, flags_dict in PRESETS.items():
        config = PipelineConfig.from_preset(name)
        result[name] = {
            "name": name,
            "active_agents": config.active_agents(),
            "agent_count": len(config.active_agents()),
            "features": flags_dict,
        }
    return result


def list_all_features() -> List[Dict]:
    """List all toggleable features with descriptions."""
    DESCRIPTIONS = {
        "deal_screening": "Initial deal sourcing and screening (Scout, Site Selector)",
        "market_research": "Comprehensive market intelligence report (Market Analyst)",
        "underwriting": "Financial underwriting and valuation (Underwriting Analyst)",
        "deal_structuring": "Capital stack and financing optimization (Deal Structurer)",
        "gaming_analysis": "Gaming terminal revenue analysis and optimization (Gaming Optimizer)",
        "risk_assessment": "7-category risk identification and quantification (Risk Officer)",
        "due_diligence": "40-item due diligence checklist execution (DD Manager)",
        "contract_review": "Lease and contract redlining (Contract Redliner)",
        "tax_planning": "Tax strategy, entity structure, cost seg, 1031 (Tax Strategist)",
        "renovation_planning": "Renovation scope, budget, and timeline (Renovation Planner)",
        "construction_drawings": "Master toggle for all construction drawing agents",
        "architectural_set": "Floor plans, site plans, elevations, code analysis (Architect)",
        "mep_drawings": "Mechanical, electrical, plumbing layouts and calcs (MEP Engineer)",
        "structural_drawings": "Foundation plans, framing, structural calcs (Structural Engineer)",
        "spec_book": "CSI MasterFormat specifications and schedules (Spec Writer)",
        "compliance_docs": "Gaming license applications and regulatory filings (Compliance Writer)",
        "exit_planning": "Disposition strategy, 1031 planning, rebalancing (Exit Strategist)",
        "generate_pdf_report": "Generate PDF summary report",
        "generate_excel_model": "Generate Excel financial model",
        "generate_cad_files": "Generate DXF/CAD drawing files",
        "generate_spec_book": "Generate spec book PDF",
    }
    return [
        {"feature": k, "description": v, "category": _categorize_feature(k)}
        for k, v in DESCRIPTIONS.items()
    ]


def _categorize_feature(feature: str) -> str:
    if feature in ("deal_screening", "market_research", "underwriting", "deal_structuring"):
        return "core_analysis"
    if feature in ("gaming_analysis",):
        return "gaming"
    if feature in ("risk_assessment", "due_diligence"):
        return "risk_dd"
    if feature in ("contract_review", "tax_planning"):
        return "legal_tax"
    if feature in ("renovation_planning", "construction_drawings", "architectural_set",
                    "mep_drawings", "structural_drawings", "spec_book"):
        return "construction"
    if feature in ("compliance_docs", "exit_planning"):
        return "compliance_exit"
    if feature.startswith("generate_"):
        return "output_format"
    return "other"
