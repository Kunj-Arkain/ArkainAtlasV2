"""
engine.tools_unified — Consolidated Tool Registry (Gap F)
============================================================
Single module that replaces data_tools.py + tools_data.py + TOOLS_PATCH.py.

Every tool has:
  - A ToolContract (inputs, outputs, evidence grade)
  - An implementation function (from data_tools.py)
  - Proper registration in the ToolRegistry

Usage:
    from tools_unified import create_tool_registry
    registry = create_tool_registry()
    result = registry.call("census_demographics", {"address": "123 Main"})
"""

from __future__ import annotations
from typing import Dict

try:
    from .tool_contracts import ToolRegistry, ToolContract
    from .evidence import EvidenceGrade, EvidenceMethod
    from . import data_tools
except ImportError:
    from tool_contracts import ToolRegistry, ToolContract
    from evidence import EvidenceGrade, EvidenceMethod
    import data_tools


# ── Tool contract definitions ──
# Each entry: (name, description, required_params, output_variables, grade, optional_params)

TOOL_DEFINITIONS = [
    {
        "name": "census_demographics",
        "description": "Census ACS demographics — population, income, unemployment",
        "required_params": ["address"],
        "optional_params": ["state", "county_fips"],
        "output_variables": ["population", "median_income", "unemployment"],
        "evidence_grade": EvidenceGrade.B,
        "evidence_method": EvidenceMethod.API_CALL,
    },
    {
        "name": "census_business_patterns",
        "description": "Census County Business Patterns — business counts, employees",
        "required_params": ["address"],
        "optional_params": ["state", "naics_code"],
        "output_variables": ["competitor_count"],
        "evidence_grade": EvidenceGrade.B,
        "evidence_method": EvidenceMethod.API_CALL,
    },
    {
        "name": "bls_employment",
        "description": "BLS employment statistics — unemployment rates, job counts",
        "required_params": ["address"],
        "optional_params": ["state"],
        "output_variables": ["unemployment", "population"],
        "evidence_grade": EvidenceGrade.B,
        "evidence_method": EvidenceMethod.API_CALL,
    },
    {
        "name": "fred_economic_data",
        "description": "FRED economic indicators — interest rates, CPI, GDP",
        "required_params": [],
        "optional_params": ["series_id"],
        "output_variables": ["interest_rate", "cap_rate", "median_income"],
        "evidence_grade": EvidenceGrade.A,
        "evidence_method": EvidenceMethod.API_CALL,
    },
    {
        "name": "gaming_board_data",
        "description": "State gaming board — terminal counts, NTI, revenue",
        "required_params": ["address"],
        "optional_params": ["state", "establishment_name"],
        "output_variables": ["nti_per_terminal", "terminal_count", "gaming_revenue"],
        "evidence_grade": EvidenceGrade.A,
        "evidence_method": EvidenceMethod.EXTERNAL_DB,
    },
    {
        "name": "environmental_risk",
        "description": "EPA environmental risk assessment — Superfund, brownfield",
        "required_params": ["address"],
        "optional_params": ["lat", "lon"],
        "output_variables": ["environmental_risk", "flood_risk"],
        "evidence_grade": EvidenceGrade.B,
        "evidence_method": EvidenceMethod.API_CALL,
    },
    {
        "name": "traffic_counts",
        "description": "State DOT traffic count data — AADT",
        "required_params": ["address"],
        "optional_params": ["state", "road_name"],
        "output_variables": ["traffic_count"],
        "evidence_grade": EvidenceGrade.B,
        "evidence_method": EvidenceMethod.API_CALL,
    },
    {
        "name": "property_records",
        "description": "County assessor property records — sqft, year built, lot size",
        "required_params": ["address"],
        "optional_params": [],
        "output_variables": ["sqft", "year_built", "lot_size_acres", "purchase_price"],
        "evidence_grade": EvidenceGrade.A,
        "evidence_method": EvidenceMethod.EXTERNAL_DB,
    },
    {
        "name": "location_scores",
        "description": "Location quality scores — walk score, transit, amenities",
        "required_params": ["address"],
        "optional_params": [],
        "output_variables": ["traffic_count", "crime_rate"],
        "evidence_grade": EvidenceGrade.C,
        "evidence_method": EvidenceMethod.API_CALL,
    },
    {
        "name": "crime_data",
        "description": "FBI UCR crime statistics — crime rate per 1000",
        "required_params": ["address"],
        "optional_params": ["state"],
        "output_variables": ["crime_rate"],
        "evidence_grade": EvidenceGrade.B,
        "evidence_method": EvidenceMethod.API_CALL,
    },
    {
        "name": "market_cap_rates",
        "description": "Market cap rate data — CoStar, LoopNet comps",
        "required_params": ["address"],
        "optional_params": ["property_type"],
        "output_variables": ["cap_rate", "exit_cap_rate"],
        "evidence_grade": EvidenceGrade.C,
        "evidence_method": EvidenceMethod.API_CALL,
    },
    {
        "name": "pull_comps",
        "description": "Comparable sales — price, cap rate, NOI from recent sales",
        "required_params": ["address"],
        "optional_params": ["property_type", "radius_miles"],
        "output_variables": ["cap_rate", "purchase_price", "noi", "exit_cap_rate"],
        "evidence_grade": EvidenceGrade.C,
        "evidence_method": EvidenceMethod.API_CALL,
    },
    {
        "name": "insurance_estimate",
        "description": "Insurance cost estimate for property",
        "required_params": ["address"],
        "optional_params": ["sqft", "year_built"],
        "output_variables": ["noi"],
        "evidence_grade": EvidenceGrade.C,
        "evidence_method": EvidenceMethod.CALCULATION,
    },
    {
        "name": "zoning_lookup",
        "description": "Municipal zoning classification lookup",
        "required_params": ["address"],
        "optional_params": [],
        "output_variables": ["environmental_risk"],
        "evidence_grade": EvidenceGrade.B,
        "evidence_method": EvidenceMethod.EXTERNAL_DB,
    },
    {
        "name": "utility_costs",
        "description": "Utility cost estimates — electric, gas, water",
        "required_params": ["address"],
        "optional_params": ["sqft"],
        "output_variables": ["noi"],
        "evidence_grade": EvidenceGrade.C,
        "evidence_method": EvidenceMethod.CALCULATION,
    },
    {
        "name": "competitor_scan",
        "description": "Nearby competitor analysis — count, distance, types",
        "required_params": ["address"],
        "optional_params": ["radius_miles", "business_type"],
        "output_variables": ["competitor_count"],
        "evidence_grade": EvidenceGrade.C,
        "evidence_method": EvidenceMethod.API_CALL,
    },
    {
        "name": "generate_term_sheets",
        "description": "Lender term sheet generation — rates, DSCR, LTV",
        "required_params": ["address"],
        "optional_params": ["noi", "purchase_price"],
        "output_variables": ["dscr", "irr", "interest_rate"],
        "evidence_grade": EvidenceGrade.C,
        "evidence_method": EvidenceMethod.CALCULATION,
    },
    {
        "name": "evaluate_deal",
        "description": "Full deal evaluation — cap rate, IRR, CoC, valuation",
        "required_params": ["address"],
        "optional_params": ["noi", "purchase_price"],
        "output_variables": ["noi", "cap_rate", "irr", "cash_on_cash", "exit_cap_rate"],
        "evidence_grade": EvidenceGrade.C,
        "evidence_method": EvidenceMethod.CALCULATION,
    },
]

# Map tool names to implementation functions in data_tools
_IMPL_MAP = {
    "census_demographics": "census_demographics",
    "census_business_patterns": "census_business_patterns",
    "bls_employment": "bls_employment",
    "fred_economic_data": "fred_economic_data",
    "gaming_board_data": "gaming_board_data",
    "environmental_risk": "environmental_risk",
    "traffic_counts": "traffic_counts",
    "property_records": "property_records",
    "location_scores": "location_scores",
    "crime_data": "crime_data",
    "market_cap_rates": "market_cap_rates",
    "insurance_estimate": "insurance_estimate",
    "zoning_lookup": "zoning_lookup",
    "utility_costs": "utility_costs",
    "competitor_scan": "competitor_scan",
}


def create_tool_registry(live: bool = True) -> ToolRegistry:
    """Create a fully-configured ToolRegistry with all tools registered.

    Args:
        live: If True, connect real implementations. If False, register
              contracts only (for FROZEN replay mode).
    """
    registry = ToolRegistry()

    for defn in TOOL_DEFINITIONS:
        contract = ToolContract(
            name=defn["name"],
            description=defn["description"],
            required_params=defn["required_params"],
            optional_params=defn.get("optional_params", []),
            output_variables=defn["output_variables"],
            evidence_grade=defn["evidence_grade"],
            evidence_method=defn.get("evidence_method", EvidenceMethod.API_CALL),
            is_live=live,
        )

        impl = None
        if live:
            impl_name = _IMPL_MAP.get(defn["name"])
            if impl_name and hasattr(data_tools, impl_name):
                fn = getattr(data_tools, impl_name)
                # Wrap data_tools functions (they take Dict params) for registry
                def _make_wrapper(func):
                    def wrapper(**kw):
                        return func(kw)
                    return wrapper
                impl = _make_wrapper(fn)

        registry.register(contract, implementation=impl)

    return registry


def tool_catalog() -> Dict:
    """Return a catalog of all available tools for documentation."""
    catalog = []
    for defn in TOOL_DEFINITIONS:
        catalog.append({
            "name": defn["name"],
            "description": defn["description"],
            "required_params": defn["required_params"],
            "output_variables": defn["output_variables"],
            "evidence_grade": defn["evidence_grade"].value,
        })
    return {"tools": catalog, "total": len(catalog)}
