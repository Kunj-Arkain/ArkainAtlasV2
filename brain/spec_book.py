"""
engine.construction.spec_book — CSI MasterFormat Specification Generator
==========================================================================
Generates project-specific construction specifications organized by
CSI MasterFormat 2016 divisions. Output as PDF spec book.

Covers typical divisions for gas station / retail / QSR renovations:
  Division 01 - General Requirements
  Division 02 - Existing Conditions
  Division 03 - Concrete
  Division 04 - Masonry
  Division 05 - Metals
  Division 06 - Wood, Plastics, Composites
  Division 07 - Thermal & Moisture Protection
  Division 08 - Openings
  Division 09 - Finishes
  Division 10 - Specialties
  Division 11 - Equipment
  Division 12 - Furnishings
  Division 13 - Special Construction (Gaming Rooms)
  Division 22 - Plumbing
  Division 23 - HVAC
  Division 26 - Electrical
  Division 27 - Communications
  Division 28 - Electronic Safety & Security
  Division 31 - Earthwork
  Division 32 - Exterior Improvements
  Division 33 - Utilities
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════
# SPEC SECTION DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════

@dataclass
class SpecSection:
    """A single specification section."""
    number: str          # e.g. "09 29 00"
    title: str           # e.g. "GYPSUM BOARD"
    division: int = 9
    parts: Dict[str, List[str]] = field(default_factory=dict)
    # Parts: GENERAL, PRODUCTS, EXECUTION (3-part spec format)

    def to_dict(self) -> Dict:
        return {"number": self.number, "title": self.title,
                "division": self.division, "parts": self.parts}


@dataclass
class SpecBook:
    """Complete project specification book."""
    project_name: str
    project_number: str
    address: str
    client_name: str
    architect: str = "Arkain Atlas"
    date: str = ""
    sections: List[SpecSection] = field(default_factory=list)

    def __post_init__(self):
        if not self.date:
            self.date = time.strftime("%B %d, %Y")

    def add_section(self, section: SpecSection):
        self.sections.append(section)

    def table_of_contents(self) -> List[Dict]:
        return [{"number": s.number, "title": s.title} for s in self.sections]

    def to_dict(self) -> Dict:
        return {
            "project_name": self.project_name,
            "project_number": self.project_number,
            "section_count": len(self.sections),
            "divisions": sorted(set(s.division for s in self.sections)),
            "toc": self.table_of_contents(),
        }


# ═══════════════════════════════════════════════════════════════
# SPEC TEMPLATES BY PROPERTY TYPE
# ═══════════════════════════════════════════════════════════════

def generate_gas_station_specs(project: Dict) -> SpecBook:
    """Generate specs for a gas station renovation/buildout."""
    book = SpecBook(
        project_name=project.get("project_name", "Gas Station Renovation"),
        project_number=project.get("project_number", ""),
        address=project.get("address", ""),
        client_name=project.get("client_name", ""),
    )

    sqft = project.get("sqft", 3000)
    terminals = project.get("terminal_count", 6)
    state = project.get("state", "IL")
    has_gaming = project.get("gaming_eligible", True)
    has_car_wash = project.get("car_wash", False)

    # Division 01 — General Requirements
    book.add_section(SpecSection("01 10 00", "SUMMARY OF WORK", 1, {
        "GENERAL": [
            f"Project: Renovation of existing gas station / convenience store, approximately {sqft:,} SF",
            f"Location: {project.get('address', 'TBD')}",
            "Work includes: interior renovation, MEP upgrades, ADA compliance, site improvements",
            f"Gaming room buildout for {terminals} video gaming terminals" if has_gaming else "",
            "Contractor shall visit site and verify all existing conditions prior to bidding",
            "Work shall be performed during normal business hours unless noted otherwise",
            "Maintain continuous fuel operations during construction where possible",
        ],
        "PRODUCTS": [
            "All materials shall be new unless noted otherwise",
            "Substitutions require written approval 10 days prior to bid date",
        ],
        "EXECUTION": [
            "Coordinate all work with Owner and fuel delivery schedule",
            "Protect existing fuel dispensers, USTs, and piping from damage",
            "Daily cleanup of construction debris required",
        ],
    }))

    book.add_section(SpecSection("01 23 00", "ALTERNATES", 1, {
        "GENERAL": [
            "Alternate 1: LED canopy lighting upgrade",
            "Alternate 2: EV charging station installation (2 stations)" if sqft > 2000 else "",
            f"Alternate 3: Car wash addition" if has_car_wash else "",
            f"Alternate 4: Gaming room premium finishes" if has_gaming else "",
        ],
    }))

    book.add_section(SpecSection("01 50 00", "TEMPORARY FACILITIES", 1, {
        "GENERAL": [
            "Provide temporary power from existing service during construction",
            "Provide temporary restroom facilities if existing restrooms are under renovation",
            "Maintain pedestrian and vehicular access to fuel islands at all times",
            "Provide dust barriers between construction area and operating retail space",
        ],
    }))

    # Division 02 — Existing Conditions
    book.add_section(SpecSection("02 41 00", "DEMOLITION", 2, {
        "GENERAL": [
            "Selective demolition of interior partitions, finishes, and equipment as shown on drawings",
            "Protect existing structure, MEP systems, and fuel infrastructure from damage",
            "Survey for asbestos-containing materials (ACM) and lead-based paint (LBP) prior to demolition",
            "If ACM or LBP found, stop work and notify Owner immediately",
        ],
        "EXECUTION": [
            "Remove interior partitions at gaming room location to studs",
            "Remove existing flooring in renovation areas to subfloor",
            "Cap and protect all exposed MEP lines during demolition",
            "Dispose of all demolition debris per local regulations",
            "Existing UST piping: DO NOT disturb without environmental consultant present",
        ],
    }))

    # Division 03 — Concrete
    book.add_section(SpecSection("03 30 00", "CAST-IN-PLACE CONCRETE", 3, {
        "GENERAL": [
            "Concrete for equipment pads, ramp repairs, and slab patching",
            "Design mix: 4,000 PSI minimum at 28 days",
        ],
        "PRODUCTS": [
            "Portland cement: ASTM C150, Type I/II",
            "Aggregate: ASTM C33, 3/4 inch maximum",
            "Reinforcement: ASTM A615, Grade 60",
            "Fiber reinforcement: polypropylene fibers at 1.5 lb/CY for slab-on-grade",
        ],
        "EXECUTION": [
            "Place concrete on prepared, compacted subgrade with vapor barrier",
            "Finish: broom finish for exterior, hard trowel for interior",
            "Cure: 7-day wet cure or approved curing compound",
            "ADA ramps: 1:12 slope maximum, detectable warning surfaces per ADA",
        ],
    }))

    # Division 07 — Thermal & Moisture
    book.add_section(SpecSection("07 21 00", "THERMAL INSULATION", 7, {
        "PRODUCTS": [
            "Batt insulation: R-19 minimum in walls, R-30 in ceiling",
            f"Rigid insulation: R-10 minimum continuous at gaming room walls" if has_gaming else "",
            "Vapor barrier: 6-mil polyethylene at warm side of insulation",
        ],
    }))

    book.add_section(SpecSection("07 50 00", "MEMBRANE ROOFING", 7, {
        "PRODUCTS": [
            "Single-ply TPO membrane: 60-mil minimum, white reflective",
            "Insulation: polyiso, minimum R-25 (code minimum per IECC climate zone)",
            "Manufacturer: Carlisle, Firestone, or Johns Manville",
        ],
        "EXECUTION": [
            "Fully adhered system with 20-year NDL warranty minimum",
            "Flash all penetrations per manufacturer's detail",
            "Install overflow scuppers at all low points",
        ],
    }))

    # Division 08 — Openings
    book.add_section(SpecSection("08 11 00", "METAL DOORS AND FRAMES", 8, {
        "PRODUCTS": [
            "Hollow metal doors: 18-gauge, flush, 1-3/4 inch thick",
            "Frames: 16-gauge welded, double rabbet",
            "Hardware: Grade 1 commercial, ADA-compliant lever handles",
            "Entry doors: storefront aluminum, thermally broken",
            "Automatic door operator at main entry (ADA requirement)",
        ],
    }))

    # Division 09 — Finishes
    book.add_section(SpecSection("09 29 00", "GYPSUM BOARD", 9, {
        "PRODUCTS": [
            "Standard: 5/8 inch Type X gypsum board",
            "Wet areas: moisture-resistant (green board) or cement board at restrooms",
            f"Gaming room: 5/8 inch Type X, Level 4 finish" if has_gaming else "",
            "Metal framing: 3-5/8 inch, 20-gauge steel studs at 16 inch OC",
        ],
    }))

    book.add_section(SpecSection("09 30 00", "TILING", 9, {
        "PRODUCTS": [
            "Restroom floors: porcelain tile, 12x12, PEI Class 4 minimum",
            "Restroom walls: ceramic tile to 48 inches AFF at wet walls",
            "Mortar: ANSI A118.4 latex-modified thin-set",
            "Grout: ANSI A118.6 polymer-modified unsanded grout",
        ],
    }))

    book.add_section(SpecSection("09 65 00", "RESILIENT FLOORING", 9, {
        "PRODUCTS": [
            "Retail area: luxury vinyl plank (LVP), 20-mil wear layer, commercial grade",
            f"Gaming area: LVP or commercial carpet tile, 28 oz minimum" if has_gaming else "",
            "Color and pattern: selected by Owner from manufacturer's standard range",
            "Manufacturer: Shaw, Armstrong, or Mohawk commercial line",
        ],
    }))

    book.add_section(SpecSection("09 91 00", "PAINTING", 9, {
        "PRODUCTS": [
            "Interior walls: latex eggshell, 2 coats over primer",
            "Trim: semi-gloss latex, 2 coats",
            "Exterior: elastomeric coating system over CMU/stucco",
            "Manufacturer: Sherwin-Williams or Benjamin Moore, commercial grade",
        ],
    }))

    # Division 10 — Specialties
    book.add_section(SpecSection("10 14 00", "SIGNAGE", 10, {
        "PRODUCTS": [
            "ADA signage: tactile and braille per ADA 703",
            "Room identification: acrylic with raised text",
            f"Gaming signage: per {state} Gaming Board requirements" if has_gaming else "",
            "Exterior illuminated sign: LED channel letters, UL listed",
            "Fuel price sign: LED digital display, auto-dimming",
        ],
    }))

    book.add_section(SpecSection("10 28 00", "TOILET ACCESSORIES", 10, {
        "PRODUCTS": [
            "Toilet paper dispenser: surface-mounted, stainless steel",
            "Paper towel dispenser: recessed or surface-mounted, touchless",
            "Soap dispenser: wall-mounted, touchless",
            "Grab bars: 1-1/2 inch diameter, stainless steel, per ADA",
            "Mirror: plate glass, 24x36 minimum, bottom at 40 inches AFF max (ADA)",
            "Baby changing station: surface-mounted, in at least one restroom",
        ],
    }))

    # Division 11 — Equipment
    book.add_section(SpecSection("11 13 00", "LOADING DOCK EQUIPMENT", 11, {
        "PRODUCTS": [
            "Walk-in cooler: 8x10 minimum, prefabricated, R-32 walls, R-40 ceiling",
            "Walk-in freezer: 6x8 minimum, prefabricated, R-32 walls, R-40 ceiling",
            "Beverage cooler: glass-door reach-in, Energy Star rated",
            "Coffee station: commercial brewer with hot water tower",
            "Food service equipment: NSF-listed, per health department requirements",
        ],
    }))

    # Division 13 — Special Construction (Gaming)
    if has_gaming:
        book.add_section(SpecSection("13 00 00", "GAMING ROOM CONSTRUCTION", 13, {
            "GENERAL": [
                f"Construct dedicated gaming room for {terminals} video gaming terminals",
                f"Minimum area: {terminals * 80} SF ({terminals} terminals × 80 SF each)",
                f"Comply with all {state} Gaming Board facility requirements",
                "Room shall be visible from main retail counter for security monitoring",
                "Provide direct access from main retail floor — no separate exterior entry",
            ],
            "PRODUCTS": [
                f"Gaming terminals: provided by licensed terminal operator (not in this contract)",
                "ATM: provided by ATM vendor, contractor provides power and data",
                "Seating: commercial-grade gaming stools or chairs, min 1 per terminal",
                f"Surveillance cameras: IP-based, min 1 per terminal + 1 at entry (per {state} req)",
                "DVR/NVR: minimum 30-day recording retention",
                "Signage: responsible gaming signage per state requirements",
                "Age verification signage: '21 AND OVER ONLY' at entry",
            ],
            "EXECUTION": [
                f"Electrical: {terminals} dedicated 20-amp circuits (1 per terminal)",
                "1 additional dedicated 20-amp circuit for gaming server/router",
                "1 dedicated circuit for ATM",
                "Data: CAT6 cable to each terminal position + 1 for server",
                "Internet: minimum 50 Mbps dedicated connection for gaming",
                "HVAC: supplemental cooling — terminals generate approximately 500 BTU/hr each",
                f"Additional cooling load: {terminals * 500:,} BTU/hr minimum",
                "Lighting: minimum 50 footcandles at terminal face",
                "Flooring: commercial carpet tile or LVP, static-dissipative",
                "Walls: painted gypsum board, Level 4 finish minimum",
                "Ceiling: 2x4 lay-in acoustic tile, 9 foot minimum height",
            ],
        }))

    # Division 22 — Plumbing
    book.add_section(SpecSection("22 00 00", "PLUMBING", 22, {
        "GENERAL": [
            "ADA-compliant restroom fixtures",
            "Mop sink in janitor closet",
            "Hot water heater replacement if unit exceeds 12 years",
        ],
        "PRODUCTS": [
            "Water closets: 1.28 GPF maximum, elongated bowl, ADA height (17-19 inches)",
            "Lavatories: wall-hung or vanity-top, ADA compliant",
            "Faucets: sensor-operated, 0.5 GPM aerator",
            "Water heater: commercial tank or tankless, Energy Star",
            "Piping: Type L copper or PEX (per local code), PVC DWV",
        ],
    }))

    # Division 23 — HVAC
    book.add_section(SpecSection("23 00 00", "HVAC", 23, {
        "GENERAL": [
            "Replace or supplement existing HVAC to serve renovated spaces",
            f"Additional {terminals * 500:,} BTU/hr cooling for gaming area" if has_gaming else "",
            "System shall maintain 72°F ±2°F at all occupied spaces",
        ],
        "PRODUCTS": [
            "Packaged rooftop unit: minimum 14 SEER2, gas heat",
            "Ductwork: galvanized steel, sealed per SMACNA Class A",
            "Diffusers: 4-way supply, stamped steel",
            "Thermostat: programmable, 7-day, Wi-Fi capable",
            "Refrigerant: R-410A or approved alternative",
        ],
        "EXECUTION": [
            "Design cooling load per Manual J or equivalent",
            "Balance all supply and return air per AABC standards",
            "Provide CO detection at fuel-adjacent spaces per code",
            "Maintain minimum outdoor air per ASHRAE 62.1",
        ],
    }))

    # Division 26 — Electrical
    book.add_section(SpecSection("26 00 00", "ELECTRICAL", 26, {
        "GENERAL": [
            "Upgrade electrical service as required for renovation scope",
            f"Gaming electrical: {terminals} dedicated 20A/120V circuits" if has_gaming else "",
            "Provide new LED lighting throughout renovated areas",
        ],
        "PRODUCTS": [
            "Panel: Square D QO or Eaton BR, copper bus",
            f"Gaming panel: dedicated 100A subpanel for gaming circuits" if has_gaming else "",
            "Wiring: THHN/THWN copper, 12 AWG minimum for 20A circuits",
            "Receptacles: commercial-grade, 20A, tamper-resistant",
            "Lighting: LED, 4000K, 90+ CRI, DLC-listed",
            "Emergency lighting: LED, 90-minute battery backup",
            "Exit signs: LED, double-face at all required exits",
        ],
        "EXECUTION": [
            "All work per NEC 2020 and local amendments",
            "Provide arc-fault protection per NEC 210.12",
            "Label all circuits at panel with typed directory",
            "Test all GFCI and AFCI devices prior to final inspection",
        ],
    }))

    # Division 27 — Communications
    book.add_section(SpecSection("27 10 00", "STRUCTURED CABLING", 27, {
        "PRODUCTS": [
            "Cable: CAT6A plenum-rated, Belden or Panduit",
            "Jacks: CAT6A keystone, color-coded per TIA-606",
            "Patch panel: 24-port, 1U rack-mounted",
            f"Gaming: {terminals + 2} data drops (1/terminal + server + spare)" if has_gaming else "",
            "POS system: 2 data drops at each register",
            "Surveillance: 1 data drop per camera location",
        ],
    }))

    # Division 28 — Security
    book.add_section(SpecSection("28 23 00", "VIDEO SURVEILLANCE", 28, {
        "PRODUCTS": [
            "Cameras: IP-based, minimum 2MP, IR night vision",
            "Coverage: all entry/exit points, register area, fuel islands",
            f"Gaming area: 1 camera per terminal + 1 at gaming room entry" if has_gaming else "",
            "NVR: minimum 30-day recording at all cameras simultaneously",
            "Monitor: 32-inch minimum at manager station",
        ],
    }))

    # Division 32 — Exterior
    book.add_section(SpecSection("32 12 00", "ASPHALT PAVING", 32, {
        "PRODUCTS": [
            "Base: 8-inch compacted aggregate base, 95% standard Proctor",
            "Surface: 3-inch hot-mix asphalt, IDOT Class I surface mix",
            "Sealcoat: coal tar emulsion, 2 coats",
            "Striping: thermoplastic, 4-inch white, ADA blue for accessible",
        ],
    }))

    book.add_section(SpecSection("32 17 00", "SITE CONCRETE", 32, {
        "PRODUCTS": [
            "Sidewalks: 4-inch, 4000 PSI, broom finish",
            "ADA ramps: per ADA/PROWAG, detectable warning tiles",
            "Curbing: 6-inch extruded concrete, machine-placed",
            "Bollards: 4-inch Schedule 40 steel, filled with concrete, at storefront and fuel islands",
        ],
    }))

    return book


def generate_retail_specs(project: Dict) -> SpecBook:
    """Generate specs for a retail strip / dollar store renovation."""
    book = SpecBook(
        project_name=project.get("project_name", "Retail Renovation"),
        project_number=project.get("project_number", ""),
        address=project.get("address", ""),
        client_name=project.get("client_name", ""),
    )

    sqft = project.get("sqft", 5000)

    book.add_section(SpecSection("01 10 00", "SUMMARY OF WORK", 1, {
        "GENERAL": [
            f"Renovation of existing retail space, approximately {sqft:,} SF",
            "Work includes: tenant improvement, MEP upgrades, ADA compliance",
            "Maintain adjacent tenant operations during construction",
        ],
    }))

    book.add_section(SpecSection("09 29 00", "GYPSUM BOARD", 9, {
        "PRODUCTS": [
            "5/8 inch Type X at all new partitions",
            "STC-50 minimum at demising walls (double layer each side)",
            "Metal framing: 3-5/8 inch, 20-gauge at 16 inch OC",
        ],
    }))

    book.add_section(SpecSection("09 51 00", "ACOUSTICAL CEILINGS", 9, {
        "PRODUCTS": [
            "Grid: 15/16 inch exposed tee, white",
            "Tile: 2x4, mineral fiber, NRC 0.55 minimum",
            "Height: 9 feet AFF minimum in retail areas",
        ],
    }))

    return book


# ═══════════════════════════════════════════════════════════════
# SPEC BOOK PDF RENDERER
# ═══════════════════════════════════════════════════════════════

def render_spec_book_pdf(book: SpecBook, filepath: str) -> str:
    """Render complete spec book as a formatted PDF."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas

    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    c = canvas.Canvas(filepath, pagesize=letter)
    w, h = letter
    margin = 1 * inch
    page_num = [0]

    def new_page():
        if page_num[0] > 0:
            c.showPage()
        page_num[0] += 1
        # Header
        c.setFont("Helvetica-Bold", 8)
        c.drawString(margin, h - 0.5 * inch, book.project_name)
        c.drawRightString(w - margin, h - 0.5 * inch, book.project_number)
        c.setLineWidth(0.5)
        c.line(margin, h - 0.6 * inch, w - margin, h - 0.6 * inch)
        # Footer
        c.setFont("Helvetica", 7)
        c.drawCentredString(w / 2, 0.5 * inch, f"Page {page_num[0]}")
        c.line(margin, 0.65 * inch, w - margin, 0.65 * inch)
        return h - 0.8 * inch

    # Cover page
    y = new_page()
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(w / 2, h * 0.65, "PROJECT SPECIFICATIONS")
    c.setFont("Helvetica", 14)
    c.drawCentredString(w / 2, h * 0.58, book.project_name)
    c.setFont("Helvetica", 11)
    c.drawCentredString(w / 2, h * 0.52, book.address)
    c.drawCentredString(w / 2, h * 0.47, f"Project No: {book.project_number}")
    c.drawCentredString(w / 2, h * 0.42, book.date)
    c.setFont("Helvetica", 10)
    c.drawCentredString(w / 2, h * 0.32, f"Prepared for: {book.client_name}")
    c.drawCentredString(w / 2, h * 0.28, f"Prepared by: {book.architect}")

    # Table of contents
    y = new_page()
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "TABLE OF CONTENTS")
    y -= 30
    current_div = -1
    for section in book.sections:
        if section.division != current_div:
            current_div = section.division
            y -= 10
            c.setFont("Helvetica-Bold", 10)
            c.drawString(margin, y, f"DIVISION {current_div:02d}")
            y -= 16
        c.setFont("Helvetica", 9)
        c.drawString(margin + 20, y, f"Section {section.number}  —  {section.title}")
        y -= 14
        if y < margin + 30:
            y = new_page()

    # Spec sections
    for section in book.sections:
        y = new_page()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, f"SECTION {section.number}")
        y -= 18
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, section.title)
        y -= 24

        for part_name, items in section.parts.items():
            c.setFont("Helvetica-Bold", 10)
            c.drawString(margin, y, f"PART {_part_number(part_name)} — {part_name}")
            y -= 16

            for item in items:
                if not item:
                    continue
                c.setFont("Helvetica", 9)
                # Word wrap
                lines = _wrap_text(item, 80)
                for i, line in enumerate(lines):
                    prefix = "•  " if i == 0 else "   "
                    c.drawString(margin + 15, y, f"{prefix}{line}")
                    y -= 13
                    if y < margin + 30:
                        y = new_page()
            y -= 8

        # End of section marker
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(w / 2, y, f"— END OF SECTION {section.number} —")

    c.save()
    return filepath


def _part_number(name: str) -> str:
    mapping = {"GENERAL": "1", "PRODUCTS": "2", "EXECUTION": "3"}
    return mapping.get(name, "1")


def _wrap_text(text: str, max_chars: int = 80) -> List[str]:
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > max_chars:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines or [""]


# ═══════════════════════════════════════════════════════════════
# SPEC GENERATORS BY TYPE
# ═══════════════════════════════════════════════════════════════

SPEC_GENERATORS = {
    "gas_station":       generate_gas_station_specs,
    "convenience_store": generate_gas_station_specs,
    "truck_stop":        generate_gas_station_specs,
    "retail_strip":      generate_retail_specs,
    "dollar_store":      generate_retail_specs,
    "qsr":               generate_retail_specs,
    "bin_store":         generate_retail_specs,
}


def generate_specs(project: Dict) -> SpecBook:
    """Auto-select spec generator based on property type."""
    ptype = project.get("property_type", "gas_station")
    generator = SPEC_GENERATORS.get(ptype, generate_gas_station_specs)
    return generator(project)
