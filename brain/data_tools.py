"""
engine.brain.data_tools — Real-World Data Access Toolkit
============================================================
Gives agents access to every public data source they need to
produce institutional-quality reports.

Data Sources:
  - US Census Bureau (ACS demographics, business patterns)
  - Bureau of Labor Statistics (employment, wages, CPI)
  - FRED / Federal Reserve (interest rates, GDP, housing)
  - State Gaming Boards (terminal counts, NTI, revenue)
  - Environmental (EPA, flood zones, brownfields)
  - Traffic (FHWA, state DOTs, AADT)
  - Property Records (county assessors via web)
  - GIS / Location (drive-time, trade area, walkscore)
  - Business Data (SBA, NAICS, establishment counts)
  - Crime & Safety (FBI UCR, local PD stats)
  - Real Estate (cap rate surveys, CBRE, Marcus & Millichap)

Architecture:
  - Each data tool is a callable registered with ToolRegistry
  - All tools use web fetch + API calls under the hood
  - Results are cached (1hr default) to avoid repeat API calls
  - State config (state_config.py) provides baseline parameters
  - LLM synthesis available when raw data needs interpretation

Env vars (all optional — tools degrade gracefully):
  CENSUS_API_KEY    — from api.census.gov (free)
  BLS_API_KEY       — from bls.gov (free, 500 req/day)
  FRED_API_KEY      — from fred.stlouisfed.org (free)
  SERPER_API_KEY    — web search fallback for all queries
  ANTHROPIC_API_KEY — LLM synthesis (already present)
  WALKSCORE_API_KEY — from walkscore.com (free tier)
"""

from __future__ import annotations

import json
import logging
import os
import time
import hashlib
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# HTTP HELPERS
# ═══════════════════════════════════════════════════════════════

_CACHE: Dict[str, tuple] = {}
_CACHE_TTL = 3600  # 1 hour


def _cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _http_get(url: str, headers: Dict = None, timeout: int = 15) -> Dict:
    """HTTP GET with caching and error handling."""
    key = _cache_key(url)
    if key in _CACHE:
        data, ts = _CACHE[key]
        if time.time() - ts < _CACHE_TTL:
            return data

    import urllib.request
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = {"raw_text": body.decode("utf-8", errors="replace")[:10000]}
        _CACHE[key] = (data, time.time())
        return data
    except Exception as e:
        logger.warning(f"HTTP GET failed: {url[:100]}... → {e}")
        return {"error": str(e)}


def _web_search_fallback(query: str, num_results: int = 5) -> str:
    """Fall back to web search when a dedicated API isn't available."""
    try:
        from engine.strategic.search_providers import multi_search
        resp = multi_search(query, num_results=num_results)
        return resp.top_snippets or f"No results for: {query}"
    except Exception as e:
        return f"Search unavailable: {e}"


def _web_fetch(url: str) -> str:
    """Fetch a full web page and extract readable text."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; ArkainAtlasBot/1.0)",
        })
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        # Strip HTML tags for text extraction
        import re
        text = re.sub(r'<script[^>]*>.*?</script>', '', raw, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:15000]  # Cap at 15K chars
    except Exception as e:
        return f"Fetch failed: {e}"


# ═══════════════════════════════════════════════════════════════
# 1. US CENSUS BUREAU
# ═══════════════════════════════════════════════════════════════

def census_demographics(params: Dict) -> Dict:
    """Pull demographics from Census ACS 5-Year estimates.

    Params:
      - state: str (2-letter code, converted to FIPS)
      - city: str (optional, for place-level data)
      - county: str (optional)
      - variables: list[str] (defaults to core set)

    Returns population, income, age, housing, education data.
    """
    api_key = os.environ.get("CENSUS_API_KEY", "")
    state_code = params.get("state", "IL")
    city = params.get("city", "")
    county = params.get("county", "")

    # FIPS mapping (subset — full mapping in production)
    STATE_FIPS = {
        "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
        "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12",
        "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
        "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
        "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
        "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
        "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
        "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
        "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
        "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
        "WY": "56",
    }

    fips = STATE_FIPS.get(state_code.upper(), "17")

    # Core ACS variables
    variables = [
        "B01003_001E",  # Total population
        "B19013_001E",  # Median household income
        "B25077_001E",  # Median home value
        "B25064_001E",  # Median gross rent
        "B01002_001E",  # Median age
        "B23025_005E",  # Unemployed
        "B23025_002E",  # In labor force
        "B15003_022E",  # Bachelor's degree
        "B15003_023E",  # Master's degree
        "B25001_001E",  # Total housing units
        "B25002_003E",  # Vacant housing units
        "B08301_001E",  # Total commuters (for traffic proxy)
        "B01003_001E",  # Total population (for density calc)
    ]
    var_str = ",".join(variables)

    if api_key:
        url = (
            f"https://api.census.gov/data/2023/acs/acs5"
            f"?get=NAME,{var_str}&for=state:{fips}&key={api_key}"
        )
        data = _http_get(url)

        if isinstance(data, list) and len(data) > 1:
            headers = data[0]
            values = data[1]
            result = dict(zip(headers, values))
            return {
                "source": "census_acs_5yr",
                "state": state_code,
                "data": result,
                "population": _safe_int(result.get("B01003_001E")),
                "median_household_income": _safe_int(result.get("B19013_001E")),
                "median_home_value": _safe_int(result.get("B25077_001E")),
                "median_rent": _safe_int(result.get("B25064_001E")),
                "median_age": _safe_float(result.get("B01002_001E")),
                "unemployment_count": _safe_int(result.get("B23025_005E")),
                "labor_force": _safe_int(result.get("B23025_002E")),
                "housing_units": _safe_int(result.get("B25001_001E")),
                "vacant_units": _safe_int(result.get("B25002_003E")),
            }

    # Fallback to web search
    return {
        "source": "web_search",
        "state": state_code,
        "city": city,
        "findings": _web_search_fallback(
            f"{city} {state_code} demographics population median income 2024 2025"
        ),
    }


def census_business_patterns(params: Dict) -> Dict:
    """Pull County Business Patterns for establishment counts.

    Useful for competition analysis — how many gas stations, restaurants,
    retail stores in a county.
    """
    api_key = os.environ.get("CENSUS_API_KEY", "")
    state = params.get("state", "IL")
    county = params.get("county", "")
    naics = params.get("naics", "447110")  # Gas stations default

    NAICS_LABELS = {
        "447110": "Gas Stations with C-Stores",
        "447190": "Other Gas Stations",
        "722511": "Full-Service Restaurants",
        "722513": "Limited-Service Restaurants",
        "445120": "Convenience Stores",
        "713210": "Casinos",
        "453310": "Used Merchandise Stores (Bin Stores)",
        "452210": "Department Stores",
        "452319": "General Merchandise (Dollar Stores)",
        "531120": "Commercial RE Lessors",
    }

    if api_key:
        # State-level business pattern
        STATE_FIPS = {"IL": "17", "NV": "32", "PA": "42", "TX": "48", "FL": "12", "CA": "06", "NY": "36", "OH": "39"}
        fips = STATE_FIPS.get(state.upper(), "17")
        url = (
            f"https://api.census.gov/data/2022/cbp"
            f"?get=NAICS2017,ESTAB,EMPSZES,EMP&for=state:{fips}"
            f"&NAICS2017={naics}&key={api_key}"
        )
        data = _http_get(url)
        if isinstance(data, list) and len(data) > 1:
            return {
                "source": "census_cbp",
                "naics": naics,
                "naics_label": NAICS_LABELS.get(naics, naics),
                "state": state,
                "data": data,
            }

    return {
        "source": "web_search",
        "naics": naics,
        "naics_label": NAICS_LABELS.get(naics, naics),
        "findings": _web_search_fallback(
            f"number of {NAICS_LABELS.get(naics, 'businesses')} {state} county business patterns"
        ),
    }


# ═══════════════════════════════════════════════════════════════
# 2. BUREAU OF LABOR STATISTICS
# ═══════════════════════════════════════════════════════════════

def bls_employment(params: Dict) -> Dict:
    """Pull employment and wage data from BLS.

    Params:
      - state: str
      - city: str / MSA
      - industry: str (NAICS or keyword)
      - series_ids: list[str] (specific BLS series)
    """
    api_key = os.environ.get("BLS_API_KEY", "")
    state = params.get("state", "IL")
    city = params.get("city", "")

    if api_key:
        # LAUS series for state unemployment
        # Format: LASST{FIPS}0000000000003 for unemployment rate
        STATE_FIPS = {"IL": "17", "NV": "32", "PA": "42", "TX": "48", "FL": "12", "CA": "06", "NY": "36", "OH": "39"}
        fips = STATE_FIPS.get(state.upper(), "17")

        series_ids = params.get("series_ids", [
            f"LASST{fips}0000000000003",  # Unemployment rate
            f"LASST{fips}0000000000005",  # Employment count
            f"LASST{fips}0000000000006",  # Unemployment count
        ])

        url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        payload = json.dumps({
            "seriesid": series_ids,
            "startyear": "2023",
            "endyear": "2025",
            "registrationkey": api_key,
        }).encode()

        import urllib.request
        try:
            req = urllib.request.Request(
                url, data=payload, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())

            results = data.get("Results", {}).get("series", [])
            parsed = []
            for series in results:
                sid = series.get("seriesID", "")
                latest = series.get("data", [{}])[0] if series.get("data") else {}
                parsed.append({
                    "series_id": sid,
                    "latest_value": latest.get("value", ""),
                    "period": latest.get("periodName", ""),
                    "year": latest.get("year", ""),
                })

            return {
                "source": "bls_api",
                "state": state,
                "series": parsed,
            }
        except Exception as e:
            logger.warning(f"BLS API failed: {e}")

    return {
        "source": "web_search",
        "state": state,
        "findings": _web_search_fallback(
            f"{city} {state} unemployment rate employment wages 2025"
        ),
    }


# ═══════════════════════════════════════════════════════════════
# 3. FRED / FEDERAL RESERVE
# ═══════════════════════════════════════════════════════════════

def fred_economic_data(params: Dict) -> Dict:
    """Pull economic time series from FRED.

    Params:
      - series_ids: list of FRED series (default: key RE metrics)
      - observation_start: str (YYYY-MM-DD)
    """
    api_key = os.environ.get("FRED_API_KEY", "")

    series_ids = params.get("series_ids", [
        "MORTGAGE30US",    # 30-year fixed mortgage rate
        "DFF",             # Federal funds effective rate
        "CPIAUCSL",        # Consumer Price Index
        "MSPUS",           # Median home sales price US
        "RRVRUSQ156N",     # Rental vacancy rate
        "UNRATE",          # National unemployment rate
        "GDP",             # Gross domestic product
        "HOUST",           # Housing starts
        "CSUSHPINSA",      # Case-Shiller home price index
    ])

    if api_key:
        results = {}
        for sid in series_ids[:10]:  # Cap at 10 to limit API calls
            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={sid}&api_key={api_key}"
                f"&file_type=json&sort_order=desc&limit=6"
            )
            data = _http_get(url)
            obs = data.get("observations", [])
            if obs:
                results[sid] = {
                    "latest_value": obs[0].get("value", ""),
                    "latest_date": obs[0].get("date", ""),
                    "prior_value": obs[1].get("value", "") if len(obs) > 1 else "",
                    "prior_date": obs[1].get("date", "") if len(obs) > 1 else "",
                }

        return {"source": "fred_api", "series": results}

    return {
        "source": "web_search",
        "findings": _web_search_fallback(
            "current mortgage rate federal funds rate CPI housing starts 2025"
        ),
    }


# ═══════════════════════════════════════════════════════════════
# 4. STATE GAMING BOARD DATA
# ═══════════════════════════════════════════════════════════════

def gaming_board_data(params: Dict) -> Dict:
    """Pull gaming revenue/terminal data from state gaming boards.

    Each state publishes data differently. This tool knows where to look
    and falls back to web search when direct APIs aren't available.
    """
    state = params.get("state", "IL")
    city = params.get("city", "")
    county = params.get("county", "")

    # Direct data source URLs by state
    GAMING_DATA_URLS = {
        "IL": "https://www.igb.illinois.gov/VideoReports.aspx",
        "PA": "https://gamingcontrolboard.pa.gov/reports-data/gaming-revenue-reports",
        "NV": "https://gaming.nv.gov/index.aspx?page=149",
        "WV": "https://wvlottery.com/general-information/financials/",
        "LA": "https://lgcb.dps.louisiana.gov/reports.html",
        "MT": "https://doj.mt.gov/gaming/",
        "OR": "https://www.oregonlottery.org/financial-reports/",
        "SD": "https://dor.sd.gov/businesses/gaming/",
    }

    from engine.realestate.state_config import get_state, is_gaming_legal
    state_cfg = get_state(state)
    gaming_cfg = state_cfg.get("gaming", {})

    result = {
        "state": state,
        "gaming_legal": is_gaming_legal(state),
        "gaming_status": gaming_cfg.get("status", "none"),
        "gaming_type": gaming_cfg.get("type", ""),
        "max_terminals": gaming_cfg.get("max_terminals_per_location"),
        "tax_rate": gaming_cfg.get("terminal_tax_rate"),
        "avg_nti_per_terminal": gaming_cfg.get("avg_nti_per_terminal_monthly"),
        "total_locations": gaming_cfg.get("total_locations"),
        "total_terminals": gaming_cfg.get("total_terminals"),
        "regulator": gaming_cfg.get("regulator", ""),
        "data_url": GAMING_DATA_URLS.get(state, ""),
    }

    # Try to fetch latest data from gaming board
    if state in GAMING_DATA_URLS:
        page_text = _web_fetch(GAMING_DATA_URLS[state])
        if len(page_text) > 100 and "error" not in page_text.lower()[:50]:
            result["board_page_excerpt"] = page_text[:3000]

    # Supplement with web search for specific area
    search_queries = [
        f"{state} gaming board video gaming terminal revenue {city or county} 2024 2025",
        f"{state} VGT net terminal income per machine monthly average",
    ]
    if city:
        search_queries.append(f"gaming terminal locations near {city} {state}")

    web_findings = []
    for q in search_queries:
        findings = _web_search_fallback(q, num_results=3)
        if findings and "No results" not in findings:
            web_findings.append({"query": q, "findings": findings})

    result["web_research"] = web_findings
    return result


# ═══════════════════════════════════════════════════════════════
# 5. ENVIRONMENTAL & FLOOD DATA
# ═══════════════════════════════════════════════════════════════

def environmental_risk(params: Dict) -> Dict:
    """Check environmental risk factors for a property.

    Checks: EPA brownfields, underground storage tanks (USTs),
    FEMA flood zones, Superfund proximity.
    """
    address = params.get("address", "")
    state = params.get("state", "")
    lat = params.get("latitude")
    lng = params.get("longitude")

    result = {"address": address, "state": state, "checks": {}}

    # EPA Envirofacts — search nearby facilities
    if address:
        epa_query = f"EPA brownfields contamination near {address}"
        result["checks"]["brownfields"] = _web_search_fallback(epa_query, 3)

    # Underground Storage Tanks (critical for gas stations)
    property_type = params.get("property_type", "")
    if property_type in ("gas_station", "convenience_store") or "gas" in address.lower():
        ust_query = f"underground storage tank {address} {state} EPA UST"
        result["checks"]["underground_storage_tanks"] = _web_search_fallback(ust_query, 3)

        ust_compliance = f"{state} underground storage tank compliance LUST cleanup"
        result["checks"]["ust_compliance_program"] = _web_search_fallback(ust_compliance, 3)

    # FEMA flood zone
    flood_query = f"FEMA flood zone map {address}"
    result["checks"]["flood_zone"] = _web_search_fallback(flood_query, 3)

    # Phase I/II environmental assessment requirements
    env_query = f"Phase I environmental assessment requirements {state} commercial property"
    result["checks"]["phase1_requirements"] = _web_search_fallback(env_query, 3)

    # Nearby Superfund sites
    if address:
        superfund_query = f"EPA Superfund site near {address}"
        result["checks"]["superfund_proximity"] = _web_search_fallback(superfund_query, 3)

    return result


# ═══════════════════════════════════════════════════════════════
# 6. TRAFFIC DATA
# ═══════════════════════════════════════════════════════════════

def traffic_counts(params: Dict) -> Dict:
    """Get traffic count data (AADT) for a location.

    Sources: State DOT databases, FHWA, web search fallback.
    """
    address = params.get("address", "")
    state = params.get("state", "")
    city = params.get("city", "")

    # State DOT traffic count URLs
    DOT_URLS = {
        "IL": "https://idot.illinois.gov/transportation-system/traffic-data",
        "TX": "https://www.txdot.gov/data-maps/traffic-counts.html",
        "FL": "https://tdaappsprod.dot.state.fl.us/fto/",
        "CA": "https://dot.ca.gov/programs/traffic-operations/census",
        "NY": "https://www.dot.ny.gov/divisions/engineering/technical-services/highway-data-services",
    }

    result = {
        "address": address,
        "state": state,
        "dot_url": DOT_URLS.get(state, ""),
    }

    # Street-level traffic search
    queries = [
        f"traffic count AADT {address} vehicles per day",
        f"{city} {state} average daily traffic count major roads",
    ]

    findings = []
    for q in queries:
        f = _web_search_fallback(q, num_results=3)
        if f and "No results" not in f:
            findings.append({"query": q, "findings": f})

    result["traffic_data"] = findings

    # Nearby major road identification
    road_query = f"major highways interstates near {address} {city} {state}"
    result["nearby_roads"] = _web_search_fallback(road_query, 3)

    return result


# ═══════════════════════════════════════════════════════════════
# 7. PROPERTY RECORDS / TAX ASSESSMENT
# ═══════════════════════════════════════════════════════════════

def property_records(params: Dict) -> Dict:
    """Look up property tax records, assessments, and ownership.

    Falls back to web search since county systems vary wildly.
    """
    address = params.get("address", "")
    county = params.get("county", "")
    state = params.get("state", "")
    parcel_id = params.get("parcel_id", "")

    result = {"address": address, "county": county, "state": state}

    queries = []
    if parcel_id:
        queries.append(f"parcel {parcel_id} {county} county {state} property tax assessment")
    queries.extend([
        f"{address} property tax assessment value {county} county {state}",
        f"{county} county {state} property tax rate 2024 2025 commercial",
        f"{address} owner property records {state}",
    ])

    findings = []
    for q in queries:
        f = _web_search_fallback(q, num_results=3)
        if f and "No results" not in f:
            findings.append({"query": q, "findings": f})

    result["records"] = findings
    return result


# ═══════════════════════════════════════════════════════════════
# 8. WALK SCORE / LOCATION QUALITY
# ═══════════════════════════════════════════════════════════════

def location_scores(params: Dict) -> Dict:
    """Get Walk Score, Transit Score, and Bike Score for a location.

    Also estimates trade area characteristics.
    """
    address = params.get("address", "")
    lat = params.get("latitude")
    lng = params.get("longitude")

    api_key = os.environ.get("WALKSCORE_API_KEY", "")

    result = {"address": address}

    if api_key and lat and lng:
        url = (
            f"https://api.walkscore.com/score"
            f"?format=json&address={quote_plus(address)}"
            f"&lat={lat}&lon={lng}&wsapikey={api_key}"
        )
        data = _http_get(url)
        if not data.get("error"):
            result["walkscore"] = data.get("walkscore")
            result["description"] = data.get("description")
            result["transit_score"] = data.get("transit", {}).get("score")
            result["bike_score"] = data.get("bike", {}).get("score")
            return result

    # Fallback to web search
    result["findings"] = _web_search_fallback(
        f"walk score transit score {address}"
    )
    return result


# ═══════════════════════════════════════════════════════════════
# 9. CRIME & SAFETY DATA
# ═══════════════════════════════════════════════════════════════

def crime_data(params: Dict) -> Dict:
    """Get crime and safety statistics for an area."""
    city = params.get("city", "")
    state = params.get("state", "")
    address = params.get("address", "")

    queries = [
        f"{city} {state} crime rate statistics 2024 2025 safe",
        f"{city} {state} violent crime property crime per capita",
    ]
    if address:
        queries.append(f"crime safety {address} neighborhood")

    findings = []
    for q in queries:
        f = _web_search_fallback(q, num_results=3)
        if f and "No results" not in f:
            findings.append({"query": q, "findings": f})

    return {"city": city, "state": state, "crime_data": findings}


# ═══════════════════════════════════════════════════════════════
# 10. REAL ESTATE MARKET DATA
# ═══════════════════════════════════════════════════════════════

def market_cap_rates(params: Dict) -> Dict:
    """Get current cap rate data and market benchmarks.

    Sources: CBRE, Marcus & Millichap, CoStar research, FRED.
    """
    property_type = params.get("property_type", "retail")
    state = params.get("state", "")
    city = params.get("city", "")

    from engine.realestate.state_config import get_business_context
    state_ctx = get_business_context(state, property_type) if state else {}

    queries = [
        f"commercial real estate cap rate {property_type.replace('_', ' ')} 2024 2025",
        f"{city} {state} retail cap rate commercial property values",
        f"Marcus Millichap CBRE {property_type.replace('_', ' ')} cap rate survey",
    ]

    findings = []
    for q in queries:
        f = _web_search_fallback(q, num_results=3)
        if f and "No results" not in f:
            findings.append({"query": q, "findings": f})

    result = {
        "property_type": property_type,
        "state": state,
        "city": city,
        "state_avg_cap_rate": state_ctx.get("avg_cap_rate"),
        "state_avg_rent_sqft": state_ctx.get("avg_rent_sqft"),
        "web_research": findings,
    }
    return result


# ═══════════════════════════════════════════════════════════════
# 11. INSURANCE ESTIMATOR
# ═══════════════════════════════════════════════════════════════

def insurance_estimate(params: Dict) -> Dict:
    """Estimate commercial property insurance costs.

    Uses property type, location, and size to estimate premiums.
    Falls back to industry benchmarks.
    """
    property_type = params.get("property_type", "retail_strip")
    state = params.get("state", "")
    purchase_price = float(params.get("purchase_price", 0))
    sqft = float(params.get("sqft", 0))
    year_built = int(params.get("year_built", 2000))

    # Industry benchmark rates (per $1,000 of insured value)
    BASE_RATES = {
        "gas_station": 8.50,        # Higher — fuel/environmental risk
        "retail_strip": 4.50,
        "shopping_center": 4.00,
        "qsr": 6.00,               # Grease fire risk
        "dollar": 3.50,            # Single tenant, low risk
        "bin_store": 4.00,
    }

    # State risk multipliers (hurricane, earthquake, etc.)
    STATE_MULTIPLIERS = {
        "FL": 1.65, "TX": 1.35, "LA": 1.55, "CA": 1.40,
        "OK": 1.25, "SC": 1.30, "NC": 1.20, "MS": 1.40,
        "AL": 1.25, "GA": 1.15, "NV": 0.90, "IL": 1.00,
        "PA": 0.95, "OH": 0.95, "IN": 1.00, "CO": 0.95,
    }

    base_rate = BASE_RATES.get(property_type, 4.50)
    state_mult = STATE_MULTIPLIERS.get(state, 1.00)

    # Age adjustment
    age = 2025 - year_built if year_built > 0 else 25
    age_mult = 1.0 + max(0, (age - 10)) * 0.01  # +1% per year over 10

    insured_value = purchase_price * 0.80  # Exclude land
    if insured_value <= 0:
        insured_value = sqft * 120 if sqft > 0 else 500000  # Fallback estimate

    annual_premium = (insured_value / 1000) * base_rate * state_mult * age_mult

    # Additional coverages
    liability = annual_premium * 0.25  # General liability ~25% of property
    umbrella = 1200  # Standard $1M umbrella
    environmental = 2500 if property_type == "gas_station" else 0
    gaming_bond = 500 if params.get("gaming_eligible") else 0

    total = annual_premium + liability + umbrella + environmental + gaming_bond

    return {
        "property_type": property_type,
        "state": state,
        "insured_value": round(insured_value),
        "annual_property_premium": round(annual_premium),
        "general_liability": round(liability),
        "umbrella_policy": umbrella,
        "environmental_coverage": environmental,
        "gaming_bond": gaming_bond,
        "total_annual_insurance": round(total),
        "monthly_insurance": round(total / 12),
        "rate_per_sqft": round(total / max(sqft, 1), 2) if sqft else None,
        "methodology": "Industry benchmark rates × state risk × age adjustment",
    }


# ═══════════════════════════════════════════════════════════════
# 12. ZONING & PERMITTING
# ═══════════════════════════════════════════════════════════════

def zoning_lookup(params: Dict) -> Dict:
    """Look up zoning and permitting requirements for a property."""
    address = params.get("address", "")
    city = params.get("city", "")
    state = params.get("state", "")
    property_type = params.get("property_type", "")

    queries = [
        f"{city} {state} zoning map commercial {property_type.replace('_', ' ')}",
        f"{city} {state} building permit requirements commercial property",
    ]
    if property_type == "gas_station":
        queries.append(f"{city} {state} gas station zoning fuel storage permit")
    if "gaming" in str(params):
        queries.append(f"{city} {state} gaming terminal zoning video gaming location requirements")

    findings = []
    for q in queries:
        f = _web_search_fallback(q, num_results=3)
        if f and "No results" not in f:
            findings.append({"query": q, "findings": f})

    return {"address": address, "city": city, "state": state, "zoning_research": findings}


# ═══════════════════════════════════════════════════════════════
# 13. WEB PAGE FETCH (for following search result links)
# ═══════════════════════════════════════════════════════════════

def fetch_webpage(params: Dict) -> Dict:
    """Fetch and extract text from a web page URL.

    Agents use this to follow links from search results and read
    full articles, reports, gaming board data pages, etc.
    """
    url = params.get("url", "")
    if not url:
        return {"error": "No URL provided"}

    text = _web_fetch(url)
    return {
        "url": url,
        "content": text,
        "length": len(text),
        "truncated": len(text) >= 14900,
    }


# ═══════════════════════════════════════════════════════════════
# 14. UTILITY COST ESTIMATOR
# ═══════════════════════════════════════════════════════════════

def utility_costs(params: Dict) -> Dict:
    """Estimate utility costs (electric, gas, water, sewer) for a commercial property."""
    state = params.get("state", "")
    sqft = float(params.get("sqft", 3000))
    property_type = params.get("property_type", "retail_strip")

    # Average commercial electricity rates by state (cents/kWh)
    ELEC_RATES = {
        "IL": 9.5, "TX": 8.2, "CA": 18.5, "NY": 16.0, "FL": 10.5,
        "PA": 9.8, "OH": 9.0, "IN": 9.2, "NV": 8.0, "GA": 9.5,
        "CO": 9.8, "WV": 8.5, "VA": 8.8, "NJ": 13.0, "MA": 16.0,
    }

    # kWh per sqft per year by property type
    ENERGY_INTENSITY = {
        "gas_station": 25,
        "retail_strip": 18,
        "shopping_center": 16,
        "qsr": 45,  # Heavy kitchen use
        "dollar": 14,
        "bin_store": 12,
    }

    rate = ELEC_RATES.get(state, 10.0) / 100  # Convert to $/kWh
    intensity = ENERGY_INTENSITY.get(property_type, 18)
    annual_kwh = sqft * intensity
    annual_electric = annual_kwh * rate

    # Natural gas (~30% of electric for retail, higher for QSR)
    gas_ratio = 0.6 if property_type == "qsr" else 0.30
    annual_gas = annual_electric * gas_ratio

    # Water/sewer (~$2-4/sqft/yr for commercial)
    water_rate = 3.5 if property_type == "qsr" else 2.0
    annual_water = sqft * water_rate

    total = annual_electric + annual_gas + annual_water

    return {
        "state": state,
        "sqft": sqft,
        "property_type": property_type,
        "annual_electric": round(annual_electric),
        "electric_rate_kwh": round(rate, 3),
        "annual_gas": round(annual_gas),
        "annual_water_sewer": round(annual_water),
        "total_annual_utilities": round(total),
        "monthly_utilities": round(total / 12),
        "per_sqft_annual": round(total / max(sqft, 1), 2),
    }


# ═══════════════════════════════════════════════════════════════
# 15. COMPETITOR MAPPING
# ═══════════════════════════════════════════════════════════════

def competitor_scan(params: Dict) -> Dict:
    """Find and analyze competitors near a property."""
    address = params.get("address", "")
    city = params.get("city", "")
    state = params.get("state", "")
    property_type = params.get("property_type", "gas_station")
    radius = params.get("radius_miles", 3)

    SEARCH_TERMS = {
        "gas_station": ["gas stations", "convenience stores", "truck stops"],
        "retail_strip": ["strip malls", "retail centers", "shopping plazas"],
        "qsr": ["fast food restaurants", "drive through restaurants", "quick service"],
        "dollar": ["dollar stores", "Dollar General", "Dollar Tree", "Family Dollar"],
        "bin_store": ["bin stores", "liquidation stores", "overstock retail"],
        "shopping_center": ["shopping centers", "retail plazas", "strip malls"],
    }

    terms = SEARCH_TERMS.get(property_type, [property_type.replace("_", " ")])

    findings = []
    for term in terms:
        q = f"{term} near {address or city + ' ' + state}"
        f = _web_search_fallback(q, num_results=5)
        if f and "No results" not in f:
            findings.append({"category": term, "findings": f})

    # Gaming-specific competition
    from engine.realestate.state_config import is_gaming_legal
    if is_gaming_legal(state):
        gaming_q = f"video gaming terminals locations near {address or city + ' ' + state}"
        gaming_f = _web_search_fallback(gaming_q, 3)
        findings.append({"category": "gaming_locations", "findings": gaming_f})

    return {
        "address": address,
        "city": city,
        "state": state,
        "property_type": property_type,
        "radius_miles": radius,
        "competitor_research": findings,
    }


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _safe_int(val) -> Optional[int]:
    try:
        return int(val) if val else None
    except (ValueError, TypeError):
        return None


def _safe_float(val) -> Optional[float]:
    try:
        return float(val) if val else None
    except (ValueError, TypeError):
        return None
