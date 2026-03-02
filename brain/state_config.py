"""
engine.realestate.state_config — State Intelligence Registry
================================================================
All 50 states + DC with:
  - Gaming legality, terminal caps, tax rates, license requirements
  - Property tax rates, sales tax, income tax
  - Business-specific regulations per property type
  - Market benchmarks (cap rate ranges, rent/sqft norms)

The pipeline and agents pull from this to auto-apply state-specific
parameters during deal evaluation and market research.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════
# GAMING STATUS ENUM
# ═══════════════════════════════════════════════════════════════

GAMING_LEGAL = "legal"             # VGTs / slot-route active
GAMING_TRIBAL_ONLY = "tribal_only" # Only tribal casinos
GAMING_CASINO_ONLY = "casino_only" # Casino gaming, no route ops
GAMING_LOTTERY_ONLY = "lottery"    # State lottery only
GAMING_NONE = "none"               # No gaming
GAMING_PENDING = "pending"         # Legislation pending


# ═══════════════════════════════════════════════════════════════
# STATE REGISTRY
# ═══════════════════════════════════════════════════════════════

STATE_CONFIG: Dict[str, Dict[str, Any]] = {

    # ── GAMING-ACTIVE STATES (slot route / VGT) ──────────────

    "IL": {
        "name": "Illinois",
        "region": "midwest",
        "gaming": {
            "status": GAMING_LEGAL,
            "type": "video_gaming_terminals",
            "max_terminals_per_location": 6,
            "terminal_tax_rate": 0.34,  # Effective combined
            "state_share": 0.2933,
            "municipality_share": 0.05,
            "license_fee_location": 100,
            "license_fee_terminal_operator": 10000,
            "license_types": ["location", "terminal_operator", "manufacturer"],
            "liquor_license_required": True,
            "min_age": 21,
            "regulator": "Illinois Gaming Board",
            "avg_nti_per_terminal_monthly": 1800,
            "total_locations": 8200,
            "total_terminals": 48000,
            "yoy_growth": 0.04,
            "notes": "Largest VGT market in US. Location must hold liquor license. 6-terminal cap. Truck stops can have up to 10.",
        },
        "taxes": {
            "corporate_income": 0.099,  # 7% + 2.5% replacement + surcharge
            "personal_income": 0.0495,
            "sales_tax_state": 0.0625,
            "sales_tax_avg_local": 0.0275,
            "property_tax_rate_avg": 0.0223,  # Highest in US
        },
        "real_estate": {
            "avg_cap_rate_retail": 0.075,
            "avg_cap_rate_gas_station": 0.08,
            "avg_rent_sqft_retail": 14.50,
            "property_tax_note": "Very high property taxes — factor into NOI",
            "closing_cost_pct": 0.025,
        },
        "business_types": {
            "gas_station": {
                "env_regs": "IL EPA — UST registration required, LUST fund",
                "fuel_tax_per_gallon": 0.392,
                "cstore_tobacco_license": True,
                "notes": "Strong gaming upside. Tank compliance critical.",
            },
            "retail_strip": {
                "notes": "Gaming café / bar tenant can boost NOI. CAM recovery common.",
            },
            "qsr": {
                "health_dept": "County health department inspection",
                "notes": "Gaming not permitted in most QSR without liquor license.",
            },
            "dollar": {
                "notes": "Strong net-lease market. DG/DT growing aggressively in IL.",
            },
            "bin_store": {
                "notes": "Growing category. Gaming can be added if location gets liquor license.",
            },
            "shopping_center": {
                "notes": "Anchor-dependent. Gaming café can fill vacant space.",
            },
        },
    },

    "NV": {
        "name": "Nevada",
        "region": "west",
        "gaming": {
            "status": GAMING_LEGAL,
            "type": "full_casino_and_route",
            "max_terminals_per_location": 15,  # Restricted license
            "terminal_tax_rate": 0.0675,  # Graduated rate
            "license_types": ["nonrestricted", "restricted", "slot_route"],
            "liquor_license_required": False,
            "min_age": 21,
            "regulator": "Nevada Gaming Control Board",
            "avg_nti_per_terminal_monthly": 2400,
            "notes": "Most permissive gaming state. Slot routes in bars, grocery, convenience. Low gaming tax.",
        },
        "taxes": {
            "corporate_income": 0.0,
            "personal_income": 0.0,
            "sales_tax_state": 0.0685,
            "sales_tax_avg_local": 0.013,
            "property_tax_rate_avg": 0.0053,
        },
        "real_estate": {
            "avg_cap_rate_retail": 0.065,
            "avg_cap_rate_gas_station": 0.07,
            "avg_rent_sqft_retail": 18.00,
            "closing_cost_pct": 0.02,
        },
        "business_types": {
            "gas_station": {
                "fuel_tax_per_gallon": 0.23,
                "notes": "Gaming in every C-store. Very competitive market.",
            },
            "retail_strip": {
                "notes": "Las Vegas market heavily supply-constrained. Henderson/Summerlin expanding.",
            },
            "qsr": {
                "notes": "Very high foot traffic near strip. Gaming in some QSR locations.",
            },
        },
    },

    "PA": {
        "name": "Pennsylvania",
        "region": "northeast",
        "gaming": {
            "status": GAMING_LEGAL,
            "type": "video_gaming_terminals",
            "max_terminals_per_location": 5,
            "terminal_tax_rate": 0.52,  # Very high
            "license_types": ["terminal_operator", "establishment"],
            "liquor_license_required": True,
            "min_age": 21,
            "regulator": "PA Gaming Control Board",
            "avg_nti_per_terminal_monthly": 1500,
            "notes": "High tax rate reduces operator share. Tavern gaming + truck stops.",
        },
        "taxes": {
            "corporate_income": 0.0899,
            "personal_income": 0.0307,
            "sales_tax_state": 0.06,
            "sales_tax_avg_local": 0.002,
            "property_tax_rate_avg": 0.0153,
        },
        "real_estate": {
            "avg_cap_rate_retail": 0.07,
            "avg_cap_rate_gas_station": 0.075,
            "avg_rent_sqft_retail": 13.00,
            "closing_cost_pct": 0.03,
            "transfer_tax": 0.02,
        },
        "business_types": {
            "gas_station": {
                "fuel_tax_per_gallon": 0.576,
                "notes": "Highest fuel tax in US. Gaming helps offset thin fuel margins.",
            },
        },
    },

    "CO": {
        "name": "Colorado",
        "region": "west",
        "gaming": {
            "status": GAMING_CASINO_ONLY,
            "type": "casino_only",
            "notes": "Limited stakes casinos in Black Hawk, Central City, Cripple Creek only. No route gaming.",
        },
        "taxes": {
            "corporate_income": 0.044,
            "personal_income": 0.044,
            "sales_tax_state": 0.029,
            "sales_tax_avg_local": 0.047,
            "property_tax_rate_avg": 0.0051,
        },
        "real_estate": {
            "avg_cap_rate_retail": 0.06,
            "avg_cap_rate_gas_station": 0.065,
            "avg_rent_sqft_retail": 20.00,
            "closing_cost_pct": 0.02,
        },
        "business_types": {
            "gas_station": {
                "fuel_tax_per_gallon": 0.22,
                "notes": "No VGT revenue. Strong C-store and car wash income in Denver metro.",
            },
        },
    },

    "IN": {
        "name": "Indiana",
        "region": "midwest",
        "gaming": {
            "status": GAMING_CASINO_ONLY,
            "type": "casino_and_racino",
            "notes": "Casinos + racinos only. No distributed gaming / VGTs. Legislation introduced periodically.",
        },
        "taxes": {
            "corporate_income": 0.049,
            "personal_income": 0.0305,
            "sales_tax_state": 0.07,
            "sales_tax_avg_local": 0.0,
            "property_tax_rate_avg": 0.0085,
        },
        "real_estate": {
            "avg_cap_rate_retail": 0.075,
            "avg_cap_rate_gas_station": 0.08,
            "avg_rent_sqft_retail": 12.00,
            "closing_cost_pct": 0.02,
        },
        "business_types": {
            "gas_station": {
                "fuel_tax_per_gallon": 0.33,
                "notes": "No gaming. Focus on fuel volume and C-store. Good IL border play for price arbitrage.",
            },
        },
    },

    "OH": {
        "name": "Ohio",
        "region": "midwest",
        "gaming": {
            "status": GAMING_CASINO_ONLY,
            "type": "casino_and_racino",
            "notes": "4 casinos + 7 racinos. No distributed VGT gaming. Skill games in legal gray area.",
        },
        "taxes": {
            "corporate_income": 0.0,  # CAT (commercial activity tax) instead
            "personal_income": 0.04,
            "sales_tax_state": 0.0575,
            "sales_tax_avg_local": 0.0175,
            "property_tax_rate_avg": 0.0156,
        },
        "real_estate": {
            "avg_cap_rate_retail": 0.075,
            "avg_cap_rate_gas_station": 0.08,
            "avg_rent_sqft_retail": 11.50,
            "closing_cost_pct": 0.025,
        },
        "business_types": {
            "gas_station": {
                "fuel_tax_per_gallon": 0.385,
                "notes": "Skill game terminals exist in gray area. Monitor legal status.",
            },
        },
    },

    "WV": {
        "name": "West Virginia",
        "region": "southeast",
        "gaming": {
            "status": GAMING_LEGAL,
            "type": "video_lottery_terminals",
            "max_terminals_per_location": 5,
            "terminal_tax_rate": 0.50,
            "license_types": ["limited_video_lottery"],
            "liquor_license_required": True,
            "min_age": 21,
            "regulator": "WV Lottery Commission",
            "avg_nti_per_terminal_monthly": 1200,
            "notes": "VLTs in bars, restaurants, fraternal orgs. Must have food/liquor license.",
        },
        "taxes": {
            "corporate_income": 0.065,
            "personal_income": 0.065,
            "sales_tax_state": 0.06,
            "sales_tax_avg_local": 0.005,
            "property_tax_rate_avg": 0.0058,
        },
        "real_estate": {
            "avg_cap_rate_retail": 0.085,
            "avg_cap_rate_gas_station": 0.09,
            "avg_rent_sqft_retail": 9.00,
            "closing_cost_pct": 0.02,
        },
    },

    # ── GAMING-PENDING / SKILL-GAME STATES ────────────────────

    "VA": {
        "name": "Virginia",
        "region": "southeast",
        "gaming": {
            "status": GAMING_PENDING,
            "type": "skill_games_regulated",
            "notes": "Skill games were banned then reinstated. Casino licenses awarded to 5 cities. Distributed gaming TBD.",
        },
        "taxes": {
            "corporate_income": 0.06,
            "personal_income": 0.0575,
            "sales_tax_state": 0.053,
            "sales_tax_avg_local": 0.007,
            "property_tax_rate_avg": 0.0082,
        },
        "real_estate": {
            "avg_cap_rate_retail": 0.065,
            "avg_cap_rate_gas_station": 0.07,
            "avg_rent_sqft_retail": 16.00,
            "closing_cost_pct": 0.025,
        },
    },

    "GA": {
        "name": "Georgia",
        "region": "southeast",
        "gaming": {
            "status": GAMING_LOTTERY_ONLY,
            "type": "lottery_and_coin_operated",
            "notes": "Coin-operated amusement machines (COAMs) legal. Full casino legislation fails repeatedly.",
        },
        "taxes": {
            "corporate_income": 0.0549,
            "personal_income": 0.055,
            "sales_tax_state": 0.04,
            "sales_tax_avg_local": 0.035,
            "property_tax_rate_avg": 0.0092,
        },
        "real_estate": {
            "avg_cap_rate_retail": 0.065,
            "avg_cap_rate_gas_station": 0.07,
            "avg_rent_sqft_retail": 16.50,
            "closing_cost_pct": 0.02,
        },
        "business_types": {
            "gas_station": {
                "fuel_tax_per_gallon": 0.312,
                "notes": "COAM revenue possible. Strong C-store market in metro Atlanta.",
            },
        },
    },

    "TX": {
        "name": "Texas",
        "region": "south",
        "gaming": {
            "status": GAMING_TRIBAL_ONLY,
            "type": "tribal_only",
            "notes": "Only tribal casino (Kickapoo Lucky Eagle). Eight-liner machines in legal gray area. Casino legislation pushed annually.",
        },
        "taxes": {
            "corporate_income": 0.0,  # Franchise tax (margin tax) instead
            "personal_income": 0.0,
            "sales_tax_state": 0.0625,
            "sales_tax_avg_local": 0.02,
            "property_tax_rate_avg": 0.018,
        },
        "real_estate": {
            "avg_cap_rate_retail": 0.065,
            "avg_cap_rate_gas_station": 0.065,
            "avg_rent_sqft_retail": 18.00,
            "closing_cost_pct": 0.02,
        },
        "business_types": {
            "gas_station": {
                "fuel_tax_per_gallon": 0.20,
                "notes": "No income tax. Buc-ee's dominates highway. Eight-liners sketchy but present.",
            },
            "dollar": {
                "notes": "Massive DG expansion. Strong NNN market. No state income tax = lower breakeven.",
            },
        },
    },

    "FL": {
        "name": "Florida",
        "region": "southeast",
        "gaming": {
            "status": GAMING_TRIBAL_ONLY,
            "type": "tribal_compact",
            "notes": "Seminole Compact controls casino gaming. No distributed VGTs. Amendment 3 (2024) failed.",
        },
        "taxes": {
            "corporate_income": 0.055,
            "personal_income": 0.0,
            "sales_tax_state": 0.06,
            "sales_tax_avg_local": 0.01,
            "property_tax_rate_avg": 0.0097,
        },
        "real_estate": {
            "avg_cap_rate_retail": 0.06,
            "avg_cap_rate_gas_station": 0.06,
            "avg_rent_sqft_retail": 22.00,
            "closing_cost_pct": 0.02,
        },
        "business_types": {
            "gas_station": {
                "fuel_tax_per_gallon": 0.368,
                "notes": "No gaming. Strong tourism traffic. Wawa/RaceTrac competition heavy.",
            },
            "retail_strip": {
                "notes": "Population growth driving strong demand. Hurricane risk in coastal areas.",
            },
        },
    },

    "CA": {
        "name": "California",
        "region": "west",
        "gaming": {
            "status": GAMING_TRIBAL_ONLY,
            "type": "tribal_only",
            "notes": "Tribal casinos only. Card rooms legal. No VGTs or slot routes.",
        },
        "taxes": {
            "corporate_income": 0.088,
            "personal_income": 0.133,
            "sales_tax_state": 0.0725,
            "sales_tax_avg_local": 0.015,
            "property_tax_rate_avg": 0.0073,
        },
        "real_estate": {
            "avg_cap_rate_retail": 0.05,
            "avg_cap_rate_gas_station": 0.05,
            "avg_rent_sqft_retail": 30.00,
            "closing_cost_pct": 0.02,
            "notes": "Prop 13 caps property tax increases. Very high barriers to entry.",
        },
    },

    "NY": {
        "name": "New York",
        "region": "northeast",
        "gaming": {
            "status": GAMING_CASINO_ONLY,
            "type": "casino_and_racino",
            "notes": "Commercial casinos + racinos. VLTs at racetracks. No distributed gaming.",
        },
        "taxes": {
            "corporate_income": 0.065,
            "personal_income": 0.109,
            "sales_tax_state": 0.04,
            "sales_tax_avg_local": 0.045,
            "property_tax_rate_avg": 0.0172,
        },
        "real_estate": {
            "avg_cap_rate_retail": 0.055,
            "avg_cap_rate_gas_station": 0.06,
            "avg_rent_sqft_retail": 25.00,
            "closing_cost_pct": 0.04,
            "transfer_tax": 0.004,
        },
    },

    # ── REMAINING STATES (condensed) ──────────────────────────

    "AL": {"name": "Alabama", "region": "southeast", "gaming": {"status": GAMING_TRIBAL_ONLY, "notes": "Poarch Creek tribal casinos. No state-regulated gaming."}, "taxes": {"corporate_income": 0.065, "personal_income": 0.05, "sales_tax_state": 0.04, "sales_tax_avg_local": 0.052, "property_tax_rate_avg": 0.004}},
    "AK": {"name": "Alaska", "region": "west", "gaming": {"status": GAMING_NONE, "notes": "Pull-tabs and bingo only. No casino or VGT gaming."}, "taxes": {"corporate_income": 0.094, "personal_income": 0.0, "sales_tax_state": 0.0, "sales_tax_avg_local": 0.018, "property_tax_rate_avg": 0.012}},
    "AZ": {"name": "Arizona", "region": "west", "gaming": {"status": GAMING_TRIBAL_ONLY, "notes": "Tribal casinos widespread. No distributed gaming."}, "taxes": {"corporate_income": 0.049, "personal_income": 0.025, "sales_tax_state": 0.056, "sales_tax_avg_local": 0.028, "property_tax_rate_avg": 0.0063}},
    "AR": {"name": "Arkansas", "region": "south", "gaming": {"status": GAMING_CASINO_ONLY, "notes": "4 casino licenses. No distributed gaming."}, "taxes": {"corporate_income": 0.053, "personal_income": 0.055, "sales_tax_state": 0.065, "sales_tax_avg_local": 0.028, "property_tax_rate_avg": 0.006}},
    "CT": {"name": "Connecticut", "region": "northeast", "gaming": {"status": GAMING_TRIBAL_ONLY, "notes": "Foxwoods and Mohegan Sun only. No distributed gaming."}, "taxes": {"corporate_income": 0.075, "personal_income": 0.0699, "sales_tax_state": 0.0635, "sales_tax_avg_local": 0.0, "property_tax_rate_avg": 0.021}},
    "DE": {"name": "Delaware", "region": "northeast", "gaming": {"status": GAMING_LEGAL, "type": "video_lottery", "notes": "VLTs at 3 racinos. Sports betting legal. Small market."}, "taxes": {"corporate_income": 0.087, "personal_income": 0.066, "sales_tax_state": 0.0, "sales_tax_avg_local": 0.0, "property_tax_rate_avg": 0.0056}},
    "DC": {"name": "District of Columbia", "region": "northeast", "gaming": {"status": GAMING_NONE, "notes": "No gaming."}, "taxes": {"corporate_income": 0.084, "personal_income": 0.105, "sales_tax_state": 0.06, "sales_tax_avg_local": 0.0, "property_tax_rate_avg": 0.0085}},
    "HI": {"name": "Hawaii", "region": "west", "gaming": {"status": GAMING_NONE, "notes": "No gaming of any kind. Strictest state."}, "taxes": {"corporate_income": 0.064, "personal_income": 0.11, "sales_tax_state": 0.04, "sales_tax_avg_local": 0.005, "property_tax_rate_avg": 0.0028}},
    "ID": {"name": "Idaho", "region": "west", "gaming": {"status": GAMING_TRIBAL_ONLY, "notes": "Tribal casinos only."}, "taxes": {"corporate_income": 0.058, "personal_income": 0.058, "sales_tax_state": 0.06, "sales_tax_avg_local": 0.003, "property_tax_rate_avg": 0.0069}},
    "IA": {"name": "Iowa", "region": "midwest", "gaming": {"status": GAMING_CASINO_ONLY, "notes": "Casinos and racinos. No distributed VGTs."}, "taxes": {"corporate_income": 0.055, "personal_income": 0.06, "sales_tax_state": 0.06, "sales_tax_avg_local": 0.01, "property_tax_rate_avg": 0.0153}},
    "KS": {"name": "Kansas", "region": "midwest", "gaming": {"status": GAMING_CASINO_ONLY, "notes": "State-owned casinos. Tribal casinos. No distributed gaming."}, "taxes": {"corporate_income": 0.04, "personal_income": 0.057, "sales_tax_state": 0.065, "sales_tax_avg_local": 0.023, "property_tax_rate_avg": 0.014}},
    "KY": {"name": "Kentucky", "region": "southeast", "gaming": {"status": GAMING_PENDING, "notes": "HHR machines at horse tracks. Distributed gaming legislation advancing."}, "taxes": {"corporate_income": 0.05, "personal_income": 0.04, "sales_tax_state": 0.06, "sales_tax_avg_local": 0.0, "property_tax_rate_avg": 0.0086}},
    "LA": {"name": "Louisiana", "region": "south", "gaming": {"status": GAMING_LEGAL, "type": "video_poker", "max_terminals_per_location": 3, "terminal_tax_rate": 0.325, "notes": "Video poker in bars/restaurants/truck stops (parish opt-in). Max 3 devices.", "avg_nti_per_terminal_monthly": 1400}, "taxes": {"corporate_income": 0.075, "personal_income": 0.0425, "sales_tax_state": 0.0445, "sales_tax_avg_local": 0.055, "property_tax_rate_avg": 0.0055}},
    "ME": {"name": "Maine", "region": "northeast", "gaming": {"status": GAMING_CASINO_ONLY, "notes": "Two casinos. No distributed gaming."}, "taxes": {"corporate_income": 0.084, "personal_income": 0.0715, "sales_tax_state": 0.055, "sales_tax_avg_local": 0.0, "property_tax_rate_avg": 0.0136}},
    "MD": {"name": "Maryland", "region": "northeast", "gaming": {"status": GAMING_CASINO_ONLY, "notes": "6 casinos. No distributed gaming."}, "taxes": {"corporate_income": 0.0875, "personal_income": 0.0575, "sales_tax_state": 0.06, "sales_tax_avg_local": 0.0, "property_tax_rate_avg": 0.0109}},
    "MA": {"name": "Massachusetts", "region": "northeast", "gaming": {"status": GAMING_CASINO_ONLY, "notes": "3 casino licenses. No distributed gaming."}, "taxes": {"corporate_income": 0.08, "personal_income": 0.09, "sales_tax_state": 0.0625, "sales_tax_avg_local": 0.0, "property_tax_rate_avg": 0.0123}},
    "MI": {"name": "Michigan", "region": "midwest", "gaming": {"status": GAMING_TRIBAL_ONLY, "notes": "Tribal casinos + 3 Detroit commercial casinos. No VGTs."}, "taxes": {"corporate_income": 0.06, "personal_income": 0.0425, "sales_tax_state": 0.06, "sales_tax_avg_local": 0.0, "property_tax_rate_avg": 0.0154}},
    "MN": {"name": "Minnesota", "region": "midwest", "gaming": {"status": GAMING_TRIBAL_ONLY, "notes": "Tribal casinos only. Charitable gaming (pull-tabs, bingo)."}, "taxes": {"corporate_income": 0.098, "personal_income": 0.0985, "sales_tax_state": 0.0688, "sales_tax_avg_local": 0.006, "property_tax_rate_avg": 0.0112}},
    "MS": {"name": "Mississippi", "region": "south", "gaming": {"status": GAMING_CASINO_ONLY, "notes": "Commercial casinos along Gulf Coast and MS River. No distributed gaming."}, "taxes": {"corporate_income": 0.05, "personal_income": 0.05, "sales_tax_state": 0.07, "sales_tax_avg_local": 0.003, "property_tax_rate_avg": 0.008}},
    "MO": {"name": "Missouri", "region": "midwest", "gaming": {"status": GAMING_CASINO_ONLY, "notes": "Riverboat casinos. No distributed gaming. VGT legislation perennial."}, "taxes": {"corporate_income": 0.04, "personal_income": 0.048, "sales_tax_state": 0.04225, "sales_tax_avg_local": 0.04, "property_tax_rate_avg": 0.0097}},
    "MT": {"name": "Montana", "region": "west", "gaming": {"status": GAMING_LEGAL, "type": "video_gambling_machines", "max_terminals_per_location": 20, "terminal_tax_rate": 0.15, "notes": "VGMs in bars/taverns/casinos. Very permissive. Low tax.", "avg_nti_per_terminal_monthly": 900}, "taxes": {"corporate_income": 0.067, "personal_income": 0.069, "sales_tax_state": 0.0, "sales_tax_avg_local": 0.0, "property_tax_rate_avg": 0.0083}},
    "NE": {"name": "Nebraska", "region": "midwest", "gaming": {"status": GAMING_PENDING, "notes": "Casino gaming approved 2020. Racetrack casinos opening. VGT expansion discussed."}, "taxes": {"corporate_income": 0.058, "personal_income": 0.0664, "sales_tax_state": 0.055, "sales_tax_avg_local": 0.015, "property_tax_rate_avg": 0.0173}},
    "NH": {"name": "New Hampshire", "region": "northeast", "gaming": {"status": GAMING_NONE, "notes": "No casinos. Charitable gaming only."}, "taxes": {"corporate_income": 0.075, "personal_income": 0.0, "sales_tax_state": 0.0, "sales_tax_avg_local": 0.0, "property_tax_rate_avg": 0.0186}},
    "NJ": {"name": "New Jersey", "region": "northeast", "gaming": {"status": GAMING_CASINO_ONLY, "notes": "Atlantic City casinos. Online gaming legal. No distributed gaming."}, "taxes": {"corporate_income": 0.115, "personal_income": 0.1075, "sales_tax_state": 0.066, "sales_tax_avg_local": 0.0, "property_tax_rate_avg": 0.0249}},
    "NM": {"name": "New Mexico", "region": "west", "gaming": {"status": GAMING_TRIBAL_ONLY, "notes": "Tribal casinos + racinos. Fraternal/veterans org gaming."}, "taxes": {"corporate_income": 0.059, "personal_income": 0.059, "sales_tax_state": 0.05125, "sales_tax_avg_local": 0.026, "property_tax_rate_avg": 0.008}},
    "NC": {"name": "North Carolina", "region": "southeast", "gaming": {"status": GAMING_TRIBAL_ONLY, "notes": "Cherokee tribal casinos expanding. No distributed gaming."}, "taxes": {"corporate_income": 0.025, "personal_income": 0.045, "sales_tax_state": 0.0475, "sales_tax_avg_local": 0.023, "property_tax_rate_avg": 0.0084}},
    "ND": {"name": "North Dakota", "region": "midwest", "gaming": {"status": GAMING_LEGAL, "type": "electronic_pull_tabs", "notes": "Charitable gaming + tribal casinos. Electronic pull-tab machines in bars.", "avg_nti_per_terminal_monthly": 700}, "taxes": {"corporate_income": 0.043, "personal_income": 0.029, "sales_tax_state": 0.05, "sales_tax_avg_local": 0.02, "property_tax_rate_avg": 0.0098}},
    "OK": {"name": "Oklahoma", "region": "south", "gaming": {"status": GAMING_TRIBAL_ONLY, "notes": "Massive tribal gaming industry. 130+ tribal casinos. No non-tribal gaming."}, "taxes": {"corporate_income": 0.04, "personal_income": 0.0475, "sales_tax_state": 0.045, "sales_tax_avg_local": 0.047, "property_tax_rate_avg": 0.009}},
    "OR": {"name": "Oregon", "region": "west", "gaming": {"status": GAMING_LEGAL, "type": "video_lottery", "max_terminals_per_location": 6, "terminal_tax_rate": 0.30, "notes": "Oregon Lottery video poker in bars/restaurants. Max 6 per location.", "avg_nti_per_terminal_monthly": 1100}, "taxes": {"corporate_income": 0.076, "personal_income": 0.099, "sales_tax_state": 0.0, "sales_tax_avg_local": 0.0, "property_tax_rate_avg": 0.0097}},
    "RI": {"name": "Rhode Island", "region": "northeast", "gaming": {"status": GAMING_CASINO_ONLY, "notes": "2 casinos (Bally's). VLTs at casinos. No distributed gaming."}, "taxes": {"corporate_income": 0.07, "personal_income": 0.0599, "sales_tax_state": 0.07, "sales_tax_avg_local": 0.0, "property_tax_rate_avg": 0.0163}},
    "SC": {"name": "South Carolina", "region": "southeast", "gaming": {"status": GAMING_NONE, "notes": "No casino gaming. Video poker banned since 2000."}, "taxes": {"corporate_income": 0.05, "personal_income": 0.064, "sales_tax_state": 0.06, "sales_tax_avg_local": 0.018, "property_tax_rate_avg": 0.0057}},
    "SD": {"name": "South Dakota", "region": "midwest", "gaming": {"status": GAMING_LEGAL, "type": "video_lottery", "max_terminals_per_location": 10, "terminal_tax_rate": 0.50, "notes": "VLTs in bars/restaurants. Deadwood casinos. Max 10 VLTs per establishment.", "avg_nti_per_terminal_monthly": 800}, "taxes": {"corporate_income": 0.0, "personal_income": 0.0, "sales_tax_state": 0.042, "sales_tax_avg_local": 0.02, "property_tax_rate_avg": 0.013}},
    "TN": {"name": "Tennessee", "region": "southeast", "gaming": {"status": GAMING_NONE, "notes": "No casino or VGT gaming. Sports betting online only."}, "taxes": {"corporate_income": 0.065, "personal_income": 0.0, "sales_tax_state": 0.07, "sales_tax_avg_local": 0.025, "property_tax_rate_avg": 0.0066}},
    "UT": {"name": "Utah", "region": "west", "gaming": {"status": GAMING_NONE, "notes": "Strictest anti-gaming state. All gambling prohibited by constitution."}, "taxes": {"corporate_income": 0.0485, "personal_income": 0.0485, "sales_tax_state": 0.0485, "sales_tax_avg_local": 0.023, "property_tax_rate_avg": 0.0058}},
    "VT": {"name": "Vermont", "region": "northeast", "gaming": {"status": GAMING_NONE, "notes": "No casino gaming. Lottery only."}, "taxes": {"corporate_income": 0.085, "personal_income": 0.0875, "sales_tax_state": 0.06, "sales_tax_avg_local": 0.01, "property_tax_rate_avg": 0.019}},
    "WA": {"name": "Washington", "region": "west", "gaming": {"status": GAMING_TRIBAL_ONLY, "notes": "29 tribal casinos. Card rooms legal. No distributed gaming."}, "taxes": {"corporate_income": 0.0, "personal_income": 0.0, "sales_tax_state": 0.065, "sales_tax_avg_local": 0.028, "property_tax_rate_avg": 0.009}},
    "WI": {"name": "Wisconsin", "region": "midwest", "gaming": {"status": GAMING_TRIBAL_ONLY, "notes": "Tribal casinos only. No distributed gaming."}, "taxes": {"corporate_income": 0.075, "personal_income": 0.0765, "sales_tax_state": 0.05, "sales_tax_avg_local": 0.005, "property_tax_rate_avg": 0.0174}},
    "WY": {"name": "Wyoming", "region": "west", "gaming": {"status": GAMING_NONE, "notes": "No casino gaming. Pari-mutuel only."}, "taxes": {"corporate_income": 0.0, "personal_income": 0.0, "sales_tax_state": 0.04, "sales_tax_avg_local": 0.014, "property_tax_rate_avg": 0.0061}},
}


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def get_state(code: str) -> Dict[str, Any]:
    """Get full state config. Returns empty dict if unknown."""
    return STATE_CONFIG.get(code.upper(), {})


def get_gaming_status(code: str) -> str:
    """Get gaming legality status for a state."""
    sc = STATE_CONFIG.get(code.upper(), {})
    return sc.get("gaming", {}).get("status", GAMING_NONE)


def is_gaming_legal(code: str) -> bool:
    """Returns True if distributed gaming (VGT/VLT) is legal in this state."""
    return get_gaming_status(code) == GAMING_LEGAL


def get_gaming_states() -> List[str]:
    """Return list of state codes where distributed gaming is legal."""
    return [code for code, cfg in STATE_CONFIG.items()
            if cfg.get("gaming", {}).get("status") == GAMING_LEGAL]


def get_all_state_codes() -> List[str]:
    """Return sorted list of all state codes."""
    return sorted(STATE_CONFIG.keys())


def get_state_summary(code: str) -> Dict[str, Any]:
    """Return a compact summary for frontend display."""
    sc = STATE_CONFIG.get(code.upper(), {})
    if not sc:
        return {}

    gaming = sc.get("gaming", {})
    taxes = sc.get("taxes", {})
    re = sc.get("real_estate", {})

    return {
        "code": code.upper(),
        "name": sc.get("name", ""),
        "region": sc.get("region", ""),
        "gaming_status": gaming.get("status", GAMING_NONE),
        "gaming_type": gaming.get("type", ""),
        "gaming_max_terminals": gaming.get("max_terminals_per_location"),
        "gaming_tax_rate": gaming.get("terminal_tax_rate"),
        "gaming_notes": gaming.get("notes", ""),
        "property_tax_rate": taxes.get("property_tax_rate_avg"),
        "sales_tax_total": round(
            taxes.get("sales_tax_state", 0) + taxes.get("sales_tax_avg_local", 0), 4
        ),
        "income_tax": taxes.get("personal_income"),
        "corporate_tax": taxes.get("corporate_income"),
        "avg_cap_rate_retail": re.get("avg_cap_rate_retail"),
        "avg_cap_rate_gas_station": re.get("avg_cap_rate_gas_station"),
    }


def get_business_context(code: str, business_type: str) -> Dict[str, Any]:
    """Return state+business-type-specific context for deal pipeline.

    This is what agents use to extract relevant data per business per state.
    """
    sc = STATE_CONFIG.get(code.upper(), {})
    if not sc:
        return {"state": code.upper(), "error": "Unknown state"}

    gaming = sc.get("gaming", {})
    taxes = sc.get("taxes", {})
    re = sc.get("real_estate", {})
    biz = sc.get("business_types", {}).get(business_type, {})

    ctx = {
        "state": code.upper(),
        "state_name": sc.get("name", ""),
        "region": sc.get("region", ""),

        # Gaming context
        "gaming_legal": gaming.get("status") == GAMING_LEGAL,
        "gaming_status": gaming.get("status", GAMING_NONE),
        "gaming_type": gaming.get("type", ""),
        "gaming_max_terminals": gaming.get("max_terminals_per_location"),
        "gaming_tax_rate": gaming.get("terminal_tax_rate"),
        "gaming_liquor_required": gaming.get("liquor_license_required"),
        "gaming_avg_nti": gaming.get("avg_nti_per_terminal_monthly"),
        "gaming_regulator": gaming.get("regulator", ""),
        "gaming_notes": gaming.get("notes", ""),

        # Tax context
        "property_tax_rate": taxes.get("property_tax_rate_avg", 0),
        "sales_tax_total": round(
            taxes.get("sales_tax_state", 0) + taxes.get("sales_tax_avg_local", 0), 4
        ),
        "corporate_income_tax": taxes.get("corporate_income", 0),

        # Real estate context
        "avg_cap_rate": re.get(f"avg_cap_rate_{business_type}",
                               re.get("avg_cap_rate_retail", 0.07)),
        "avg_rent_sqft": re.get("avg_rent_sqft_retail", 0),
        "closing_cost_pct": re.get("closing_cost_pct", 0.025),
        "transfer_tax": re.get("transfer_tax", 0),

        # Business-type-specific
        **biz,
    }

    return ctx
