"""
engine.construction.drawing_engine — Architectural Drawing Engine
===================================================================
Generates production construction documents as PDF sheets and DXF files.

Sheet sizes: ARCH D (24×36), ARCH E (30×42), ANSI D (22×34), A1, A3
Scale support: 1/4"=1'-0", 1/8"=1'-0", 1/16"=1'-0", 1"=10', etc.
Output formats: PDF (reportlab), DXF (raw text), SVG (raw XML)

Drawing types generated:
  - Site Plans (A1.0)
  - Floor Plans (A2.x)
  - Reflected Ceiling Plans (A3.x)
  - Exterior Elevations (A4.x)
  - Building Sections (A5.x)
  - Wall Sections / Details (A6.x)
  - Electrical Plans (E1.x)
  - Plumbing Plans (P1.x)
  - HVAC Plans (M1.x)
  - Structural Plans (S1.x)
  - Demolition Plans (D1.x)

Requires: reportlab (PDF), math, struct (DXF)
Optional: ezdxf (enhanced DXF), svgwrite (enhanced SVG)

Usage:
    from engine.construction.drawing_engine import DrawingEngine, SheetConfig

    engine = DrawingEngine(project_name="456 Oak Ave Gas Station")
    sheet = engine.new_sheet("A2.1", "FLOOR PLAN - EXISTING", scale="1/4 inch = 1 foot")

    # Draw walls, doors, windows, fixtures
    sheet.wall(0, 0, 60, 0, thickness=6)  # 60' wall, 6" thick
    sheet.door(20, 0, width=36, swing="in")
    sheet.window(35, 0, width=48)
    sheet.dimension(0, 0, 60, 0, offset=24)  # 60'-0" dimension

    # Gaming area
    sheet.room_label(30, 15, "GAMING AREA", "425 SF")
    sheet.equipment("VGT", 25, 10, count=6, spacing=48)

    # Generate output
    engine.render_pdf("/output/drawings/A2.1_floor_plan.pdf")
    engine.render_dxf("/output/drawings/A2.1_floor_plan.dxf")
"""

from __future__ import annotations

import math
import os
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Sheet sizes in points (1 point = 1/72 inch)
PTS_PER_INCH = 72.0

SHEET_SIZES = {
    "ARCH_D": (24 * PTS_PER_INCH, 36 * PTS_PER_INCH),    # 24×36
    "ARCH_E": (30 * PTS_PER_INCH, 42 * PTS_PER_INCH),    # 30×42
    "ANSI_D": (22 * PTS_PER_INCH, 34 * PTS_PER_INCH),    # 22×34
    "A1":     (23.39 * PTS_PER_INCH, 33.11 * PTS_PER_INCH),
    "A3":     (11.69 * PTS_PER_INCH, 16.54 * PTS_PER_INCH),
    "LETTER": (8.5 * PTS_PER_INCH, 11 * PTS_PER_INCH),
}

# Scale factors: drawing_inches per real_foot
SCALES = {
    "1/4 inch = 1 foot":    0.25,    # 1/4" = 1'-0" (standard floor plan)
    "1/8 inch = 1 foot":    0.125,   # 1/8" = 1'-0" (site plan / large buildings)
    "1/16 inch = 1 foot":   0.0625,  # 1/16" = 1'-0" (site plan)
    "3/8 inch = 1 foot":    0.375,   # 3/8" = 1'-0" (enlarged plans)
    "1/2 inch = 1 foot":    0.5,     # 1/2" = 1'-0" (details)
    "3/4 inch = 1 foot":    0.75,    # 3/4" = 1'-0" (large details)
    "1 inch = 1 foot":      1.0,     # 1" = 1'-0" (full details)
    "1.5 inch = 1 foot":    1.5,     # 1-1/2" = 1'-0" (wall sections)
    "3 inch = 1 foot":      3.0,     # 3" = 1'-0" (fine details)
    "1 inch = 10 feet":     0.1 / 12.0 * PTS_PER_INCH,  # site plans
    "1 inch = 20 feet":     0.05 / 12.0 * PTS_PER_INCH,
    "1 inch = 50 feet":     0.02 / 12.0 * PTS_PER_INCH,
}

# Line weights (in points)
LINE_WEIGHTS = {
    "cut":        0.8,    # Walls/objects cut by plan
    "profile":    0.5,    # Object outlines beyond cut
    "medium":     0.35,   # Fixtures, casework
    "light":      0.25,   # Dimensions, hatching
    "fine":       0.15,   # Text leaders, hidden
    "dashed":     0.25,   # Hidden lines
}

# Standard text heights
TEXT_HEIGHTS = {
    "title":      14,     # Drawing title
    "subtitle":   10,     # Room names
    "dimension":  7,      # Dimension text
    "note":       6,      # General notes
    "fine":       5,      # Small annotations
}

# AIA layer naming convention
LAYERS = {
    "A-WALL":      {"color": "black",  "weight": "cut",     "desc": "Walls"},
    "A-WALL-PATT": {"color": "gray",   "weight": "light",   "desc": "Wall hatch/fill"},
    "A-DOOR":      {"color": "black",  "weight": "medium",  "desc": "Doors"},
    "A-GLAZ":      {"color": "black",  "weight": "medium",  "desc": "Windows/glazing"},
    "A-FLOR-FIXT": {"color": "black",  "weight": "medium",  "desc": "Floor fixtures"},
    "A-FLOR-PATT": {"color": "gray",   "weight": "fine",    "desc": "Floor pattern"},
    "A-CLNG":      {"color": "gray",   "weight": "light",   "desc": "Ceiling grid"},
    "A-FURN":      {"color": "gray",   "weight": "light",   "desc": "Furniture"},
    "A-EQPM":      {"color": "black",  "weight": "medium",  "desc": "Equipment"},
    "A-ANNO-DIMS": {"color": "black",  "weight": "light",   "desc": "Dimensions"},
    "A-ANNO-NOTE": {"color": "black",  "weight": "fine",    "desc": "Notes/text"},
    "A-ANNO-SYMB": {"color": "black",  "weight": "light",   "desc": "Symbols"},
    "E-LITE":      {"color": "blue",   "weight": "medium",  "desc": "Lighting"},
    "E-POWR":      {"color": "red",    "weight": "medium",  "desc": "Power/receptacles"},
    "E-COMM":      {"color": "green",  "weight": "light",   "desc": "Communications"},
    "P-FIXT":      {"color": "blue",   "weight": "medium",  "desc": "Plumbing fixtures"},
    "P-PIPE":      {"color": "blue",   "weight": "light",   "desc": "Piping"},
    "M-DUCT":      {"color": "green",  "weight": "medium",  "desc": "Ductwork"},
    "M-EQPM":      {"color": "green",  "weight": "medium",  "desc": "HVAC equipment"},
    "S-COLS":      {"color": "red",    "weight": "cut",     "desc": "Structural columns"},
    "S-BEAM":      {"color": "red",    "weight": "medium",  "desc": "Beams"},
    "S-FNDN":      {"color": "red",    "weight": "cut",     "desc": "Foundation"},
}


# ═══════════════════════════════════════════════════════════════
# DRAWING ELEMENT DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class DrawingElement:
    """A single element on a drawing sheet."""
    element_type: str  # wall, door, window, dimension, text, equipment, etc.
    layer: str = "A-WALL"
    x1: float = 0
    y1: float = 0
    x2: float = 0
    y2: float = 0
    properties: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "type": self.element_type, "layer": self.layer,
            "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
            "properties": self.properties,
        }


@dataclass
class SheetConfig:
    """Configuration for a single drawing sheet."""
    sheet_number: str       # e.g. "A2.1"
    sheet_title: str        # e.g. "FLOOR PLAN - EXISTING"
    sheet_size: str = "ARCH_D"
    scale: str = "1/4 inch = 1 foot"
    discipline: str = "A"   # A=Arch, S=Structural, M=Mech, E=Elec, P=Plumb
    elements: List[DrawingElement] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    revision: str = ""
    date: str = ""

    @property
    def scale_factor(self) -> float:
        return SCALES.get(self.scale, 0.25)

    @property
    def sheet_width(self) -> float:
        return SHEET_SIZES.get(self.sheet_size, SHEET_SIZES["ARCH_D"])[1]  # landscape

    @property
    def sheet_height(self) -> float:
        return SHEET_SIZES.get(self.sheet_size, SHEET_SIZES["ARCH_D"])[0]  # landscape

    def to_dict(self) -> Dict:
        return {
            "sheet_number": self.sheet_number,
            "sheet_title": self.sheet_title,
            "sheet_size": self.sheet_size,
            "scale": self.scale,
            "discipline": self.discipline,
            "element_count": len(self.elements),
            "note_count": len(self.notes),
        }


# ═══════════════════════════════════════════════════════════════
# DRAWING SHEET — HIGH-LEVEL API
# ═══════════════════════════════════════════════════════════════

class DrawingSheet:
    """High-level API for adding elements to a sheet.

    All dimensions are in REAL FEET. The engine converts to
    drawing units using the sheet's scale factor.
    """

    def __init__(self, config: SheetConfig):
        self.config = config
        self._scale = config.scale_factor * PTS_PER_INCH  # pts per real foot
        # Drawing area (inside border)
        self._margin = 0.75 * PTS_PER_INCH
        self._title_block_h = 1.5 * PTS_PER_INCH
        self._origin_x = self._margin + 0.5 * PTS_PER_INCH
        self._origin_y = self._margin + self._title_block_h + 0.5 * PTS_PER_INCH

    def _to_pts(self, feet: float) -> float:
        """Convert real feet to drawing points."""
        return feet * self._scale

    def _to_sheet(self, x_ft: float, y_ft: float) -> Tuple[float, float]:
        """Convert real-world coordinates to sheet coordinates."""
        return (self._origin_x + self._to_pts(x_ft),
                self._origin_y + self._to_pts(y_ft))

    # ── Architectural Elements ────────────────────────────

    def wall(self, x1: float, y1: float, x2: float, y2: float,
             thickness: float = 6, wall_type: str = "standard"):
        """Draw a wall. Coordinates in feet, thickness in inches."""
        self.config.elements.append(DrawingElement(
            "wall", "A-WALL", x1, y1, x2, y2,
            {"thickness_in": thickness, "wall_type": wall_type},
        ))

    def door(self, x: float, y: float, width: float = 36,
             swing: str = "in", door_type: str = "single"):
        """Place a door. Width in inches."""
        self.config.elements.append(DrawingElement(
            "door", "A-DOOR", x, y, x + width / 12, y,
            {"width_in": width, "swing": swing, "door_type": door_type},
        ))

    def window(self, x: float, y: float, width: float = 48,
               height: float = 48, sill_height: float = 36):
        """Place a window. Dimensions in inches."""
        self.config.elements.append(DrawingElement(
            "window", "A-GLAZ", x, y, x + width / 12, y,
            {"width_in": width, "height_in": height, "sill_in": sill_height},
        ))

    def room_label(self, x: float, y: float, name: str,
                   area_sf: str = "", room_number: str = ""):
        """Place a room name and area label."""
        self.config.elements.append(DrawingElement(
            "room_label", "A-ANNO-NOTE", x, y, x, y,
            {"name": name, "area": area_sf, "number": room_number},
        ))

    # ── Equipment ─────────────────────────────────────────

    def equipment(self, equip_type: str, x: float, y: float,
                  count: int = 1, spacing: float = 4,
                  label: str = ""):
        """Place equipment (VGTs, coolers, shelving, etc.)."""
        self.config.elements.append(DrawingElement(
            "equipment", "A-EQPM", x, y, x + count * spacing, y,
            {"equip_type": equip_type, "count": count,
             "spacing_ft": spacing, "label": label},
        ))

    def fixture(self, fixture_type: str, x: float, y: float, rotation: float = 0):
        """Place a plumbing/architectural fixture."""
        layer = "P-FIXT" if fixture_type in ("toilet", "sink", "urinal", "mop_sink") else "A-FLOR-FIXT"
        self.config.elements.append(DrawingElement(
            "fixture", layer, x, y, x, y,
            {"fixture_type": fixture_type, "rotation": rotation},
        ))

    # ── MEP Elements ──────────────────────────────────────

    def electrical_panel(self, x: float, y: float, amps: int = 200, label: str = "MDP"):
        self.config.elements.append(DrawingElement(
            "elec_panel", "E-POWR", x, y, x, y,
            {"amps": amps, "label": label},
        ))

    def receptacle(self, x: float, y: float, circuit: str = "", dedicated: bool = False):
        self.config.elements.append(DrawingElement(
            "receptacle", "E-POWR", x, y, x, y,
            {"circuit": circuit, "dedicated": dedicated},
        ))

    def light_fixture(self, x: float, y: float, fixture_type: str = "2x4_troffer"):
        self.config.elements.append(DrawingElement(
            "light", "E-LITE", x, y, x, y,
            {"fixture_type": fixture_type},
        ))

    def duct_run(self, x1: float, y1: float, x2: float, y2: float,
                 size: str = "12x8", cfm: int = 0):
        self.config.elements.append(DrawingElement(
            "duct", "M-DUCT", x1, y1, x2, y2,
            {"size": size, "cfm": cfm},
        ))

    def diffuser(self, x: float, y: float, diff_type: str = "supply", size: str = "12x12"):
        self.config.elements.append(DrawingElement(
            "diffuser", "M-EQPM", x, y, x, y,
            {"type": diff_type, "size": size},
        ))

    def pipe_run(self, x1: float, y1: float, x2: float, y2: float,
                 pipe_type: str = "CW", size: str = "3/4"):
        self.config.elements.append(DrawingElement(
            "pipe", "P-PIPE", x1, y1, x2, y2,
            {"pipe_type": pipe_type, "size": size},
        ))

    # ── Structural Elements ───────────────────────────────

    def column(self, x: float, y: float, size: str = "W10x33"):
        self.config.elements.append(DrawingElement(
            "column", "S-COLS", x, y, x, y,
            {"size": size},
        ))

    def beam(self, x1: float, y1: float, x2: float, y2: float, size: str = "W12x26"):
        self.config.elements.append(DrawingElement(
            "beam", "S-BEAM", x1, y1, x2, y2,
            {"size": size},
        ))

    def footing(self, x: float, y: float, width: float = 3, depth: float = 3):
        self.config.elements.append(DrawingElement(
            "footing", "S-FNDN", x, y, x + width, y + depth,
            {"width_ft": width, "depth_ft": depth},
        ))

    # ── Annotations ───────────────────────────────────────

    def dimension(self, x1: float, y1: float, x2: float, y2: float,
                  offset: float = 2, text: str = ""):
        """Add a dimension line. Offset in feet from the measured line."""
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        dim_text = text or _format_feet_inches(dist)
        self.config.elements.append(DrawingElement(
            "dimension", "A-ANNO-DIMS", x1, y1, x2, y2,
            {"offset_ft": offset, "text": dim_text},
        ))

    def note(self, x: float, y: float, text: str, number: int = 0):
        self.config.elements.append(DrawingElement(
            "note", "A-ANNO-NOTE", x, y, x, y,
            {"text": text, "number": number},
        ))

    def section_mark(self, x: float, y: float, section_id: str = "1",
                     sheet_ref: str = "A5.1", direction: str = "right"):
        self.config.elements.append(DrawingElement(
            "section_mark", "A-ANNO-SYMB", x, y, x, y,
            {"section_id": section_id, "sheet_ref": sheet_ref, "direction": direction},
        ))

    def north_arrow(self, x: float, y: float, rotation: float = 0):
        self.config.elements.append(DrawingElement(
            "north_arrow", "A-ANNO-SYMB", x, y, x, y,
            {"rotation": rotation},
        ))

    def add_note(self, text: str):
        """Add to the sheet's general notes list."""
        self.config.notes.append(text)


# ═══════════════════════════════════════════════════════════════
# DRAWING ENGINE — ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

class DrawingEngine:
    """Orchestrates multi-sheet drawing set generation."""

    def __init__(
        self,
        project_name: str = "",
        project_number: str = "",
        client_name: str = "",
        address: str = "",
        architect: str = "Arkain Atlas",
    ):
        self.project_name = project_name
        self.project_number = project_number or f"PRJ-{int(time.time()) % 100000:05d}"
        self.client_name = client_name
        self.address = address
        self.architect = architect
        self.sheets: List[DrawingSheet] = []
        self.cover_sheet_data: Dict = {}

    def new_sheet(self, sheet_number: str, sheet_title: str,
                  scale: str = "1/4 inch = 1 foot",
                  sheet_size: str = "ARCH_D") -> DrawingSheet:
        """Create a new drawing sheet."""
        discipline = sheet_number[0] if sheet_number else "A"
        config = SheetConfig(
            sheet_number=sheet_number,
            sheet_title=sheet_title,
            sheet_size=sheet_size,
            scale=scale,
            discipline=discipline,
            date=time.strftime("%m/%d/%Y"),
        )
        sheet = DrawingSheet(config)
        self.sheets.append(sheet)
        return sheet

    def sheet_index(self) -> List[Dict]:
        """Get the drawing index (cover sheet list)."""
        return [
            {"number": s.config.sheet_number, "title": s.config.sheet_title,
             "size": s.config.sheet_size, "scale": s.config.scale,
             "elements": len(s.config.elements)}
            for s in self.sheets
        ]

    # ── PDF Rendering ─────────────────────────────────────

    def render_all_pdf(self, output_dir: str) -> List[str]:
        """Render all sheets as individual PDF files."""
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for sheet in self.sheets:
            filename = f"{sheet.config.sheet_number}_{_safe_filename(sheet.config.sheet_title)}.pdf"
            filepath = os.path.join(output_dir, filename)
            self._render_sheet_pdf(sheet, filepath)
            paths.append(filepath)
        return paths

    def render_combined_pdf(self, filepath: str) -> str:
        """Render all sheets into a single multi-page PDF."""
        from reportlab.lib.pagesizes import landscape
        from reportlab.pdfgen import canvas

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        c = canvas.Canvas(filepath)

        for sheet in self.sheets:
            w, h = sheet.config.sheet_width, sheet.config.sheet_height
            c.setPageSize((w, h))
            self._draw_sheet_on_canvas(c, sheet)
            c.showPage()

        c.save()
        logger.info(f"Combined PDF: {filepath} ({len(self.sheets)} sheets)")
        return filepath

    def _render_sheet_pdf(self, sheet: DrawingSheet, filepath: str):
        """Render a single sheet to PDF."""
        from reportlab.pdfgen import canvas

        w, h = sheet.config.sheet_width, sheet.config.sheet_height
        c = canvas.Canvas(filepath, pagesize=(w, h))
        self._draw_sheet_on_canvas(c, sheet)
        c.save()

    def _draw_sheet_on_canvas(self, c, sheet: DrawingSheet):
        """Draw all elements of a sheet onto a reportlab canvas."""
        w = sheet.config.sheet_width
        h = sheet.config.sheet_height
        margin = 0.5 * PTS_PER_INCH
        tb_h = 1.25 * PTS_PER_INCH  # title block height

        # ── Border ────────────────────────────────────────
        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(2)
        c.rect(margin, margin, w - 2 * margin, h - 2 * margin)

        # ── Title Block ───────────────────────────────────
        tb_y = margin
        tb_w = w - 2 * margin
        c.setLineWidth(1)
        c.rect(margin, tb_y, tb_w, tb_h)

        # Vertical dividers in title block
        col1 = margin + tb_w * 0.55
        col2 = margin + tb_w * 0.75
        c.line(col1, tb_y, col1, tb_y + tb_h)
        c.line(col2, tb_y, col2, tb_y + tb_h)

        # Title block text
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin + 10, tb_y + tb_h - 18, self.project_name.upper())
        c.setFont("Helvetica", 7)
        c.drawString(margin + 10, tb_y + tb_h - 30, self.address)
        c.drawString(margin + 10, tb_y + tb_h - 42, f"Project: {self.project_number}")
        c.drawString(margin + 10, tb_y + tb_h - 54, f"Client: {self.client_name}")

        c.setFont("Helvetica-Bold", 8)
        c.drawString(col1 + 8, tb_y + tb_h - 18, "SHEET:")
        c.setFont("Helvetica-Bold", 18)
        c.drawString(col1 + 8, tb_y + tb_h - 50, sheet.config.sheet_number)

        c.setFont("Helvetica-Bold", 8)
        c.drawString(col2 + 8, tb_y + tb_h - 18, sheet.config.sheet_title)
        c.setFont("Helvetica", 7)
        c.drawString(col2 + 8, tb_y + tb_h - 32, f"Scale: {sheet.config.scale}")
        c.drawString(col2 + 8, tb_y + tb_h - 44, f"Date: {sheet.config.date}")
        c.drawString(col2 + 8, tb_y + tb_h - 56, f"Drawn by: {self.architect}")

        # ── Drawing Area ──────────────────────────────────
        draw_y = margin + tb_h + 0.25 * PTS_PER_INCH
        draw_h = h - 2 * margin - tb_h - 0.25 * PTS_PER_INCH
        draw_x = margin + 0.25 * PTS_PER_INCH
        draw_w = w - 2 * margin - 0.25 * PTS_PER_INCH

        scale = sheet._scale
        ox, oy = draw_x + 36, draw_y + 36  # origin offset

        # ── Render Elements ───────────────────────────────
        for elem in sheet.config.elements:
            self._render_element(c, elem, ox, oy, scale)

        # ── General Notes ─────────────────────────────────
        if sheet.config.notes:
            notes_x = w - margin - 3.5 * PTS_PER_INCH
            notes_y = draw_y + draw_h - 20
            c.setFont("Helvetica-Bold", 8)
            c.drawString(notes_x, notes_y, "GENERAL NOTES:")
            c.setFont("Helvetica", 6)
            for i, note in enumerate(sheet.config.notes):
                c.drawString(notes_x, notes_y - 14 - i * 10, f"{i + 1}. {note}")

    def _render_element(self, c, elem: DrawingElement,
                        ox: float, oy: float, scale: float):
        """Render a single element on the canvas."""
        props = elem.properties
        x1 = ox + elem.x1 * scale
        y1 = oy + elem.y1 * scale
        x2 = ox + elem.x2 * scale
        y2 = oy + elem.y2 * scale

        layer_info = LAYERS.get(elem.layer, {"weight": "medium"})
        lw = LINE_WEIGHTS.get(layer_info.get("weight", "medium"), 0.35)
        c.setLineWidth(lw)
        c.setStrokeColorRGB(0, 0, 0)

        if elem.element_type == "wall":
            thick_pts = (props.get("thickness_in", 6) / 12.0) * scale
            # Draw double-line wall
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                nx = -dy / length * thick_pts / 2
                ny = dx / length * thick_pts / 2
                c.setLineWidth(lw)
                c.line(x1 + nx, y1 + ny, x2 + nx, y2 + ny)
                c.line(x1 - nx, y1 - ny, x2 - nx, y2 - ny)
                # End caps
                c.line(x1 + nx, y1 + ny, x1 - nx, y1 - ny)
                c.line(x2 + nx, y2 + ny, x2 - nx, y2 - ny)
                # Fill
                c.setFillColorRGB(0.85, 0.85, 0.85)
                path = c.beginPath()
                path.moveTo(x1 + nx, y1 + ny)
                path.lineTo(x2 + nx, y2 + ny)
                path.lineTo(x2 - nx, y2 - ny)
                path.lineTo(x1 - nx, y1 - ny)
                path.close()
                c.drawPath(path, fill=1, stroke=0)
                # Redraw outlines
                c.line(x1 + nx, y1 + ny, x2 + nx, y2 + ny)
                c.line(x1 - nx, y1 - ny, x2 - nx, y2 - ny)

        elif elem.element_type == "door":
            width_pts = (props.get("width_in", 36) / 12.0) * scale
            c.setLineWidth(0.5)
            # Door opening (break in wall)
            c.setStrokeColorRGB(1, 1, 1)
            c.line(x1, y1, x1 + width_pts, y1)
            # Door leaf
            c.setStrokeColorRGB(0, 0, 0)
            c.line(x1, y1, x1 + width_pts * 0.7, y1 + width_pts * 0.7)
            # Swing arc
            c.setDash([2, 2])
            c.arc(x1 - width_pts * 0.7, y1 - width_pts * 0.7,
                  x1 + width_pts * 0.7, y1 + width_pts * 0.7, 0, 90)
            c.setDash([])

        elif elem.element_type == "window":
            width_pts = (props.get("width_in", 48) / 12.0) * scale
            c.setLineWidth(0.5)
            c.line(x1, y1 - 1.5, x1 + width_pts, y1 - 1.5)
            c.line(x1, y1 + 1.5, x1 + width_pts, y1 + 1.5)

        elif elem.element_type == "room_label":
            c.setFont("Helvetica-Bold", 9)
            c.drawCentredString(x1, y1 + 5, props.get("name", ""))
            if props.get("area"):
                c.setFont("Helvetica", 7)
                c.drawCentredString(x1, y1 - 8, props["area"])
            if props.get("number"):
                c.setFont("Helvetica", 6)
                c.drawCentredString(x1, y1 - 18, f"Room {props['number']}")

        elif elem.element_type == "equipment":
            count = props.get("count", 1)
            spacing = props.get("spacing_ft", 4) * scale
            equip_w = 2.0 * scale  # ~2ft wide equipment symbol
            equip_h = 1.5 * scale
            for i in range(count):
                ex = x1 + i * spacing
                c.rect(ex - equip_w / 2, y1 - equip_h / 2, equip_w, equip_h)
                c.setFont("Helvetica", 5)
                c.drawCentredString(ex, y1 - equip_h / 2 - 8,
                                    props.get("equip_type", "EQ"))

        elif elem.element_type == "dimension":
            offset = props.get("offset_ft", 2) * scale
            c.setLineWidth(0.2)
            c.setFont("Helvetica", 6)
            # Horizontal dim
            dim_y = y1 + offset
            c.line(x1, dim_y, x2, dim_y)
            # Ticks
            c.line(x1, dim_y - 4, x1, dim_y + 4)
            c.line(x2, dim_y - 4, x2, dim_y + 4)
            # Extension lines
            c.setDash([1, 2])
            c.line(x1, y1, x1, dim_y - 2)
            c.line(x2, y2, x2, dim_y - 2)
            c.setDash([])
            # Text
            mid_x = (x1 + x2) / 2
            c.drawCentredString(mid_x, dim_y + 3, props.get("text", ""))

        elif elem.element_type == "note":
            c.setFont("Helvetica", 6)
            text = props.get("text", "")
            num = props.get("number", 0)
            prefix = f"{num}. " if num else ""
            c.drawString(x1, y1, f"{prefix}{text}")

        elif elem.element_type == "elec_panel":
            # Panel symbol: rectangle with X
            pw, ph = 1.5 * scale, 2 * scale
            c.rect(x1 - pw / 2, y1 - ph / 2, pw, ph)
            c.line(x1 - pw / 2, y1 - ph / 2, x1 + pw / 2, y1 + ph / 2)
            c.line(x1 - pw / 2, y1 + ph / 2, x1 + pw / 2, y1 - ph / 2)
            c.setFont("Helvetica-Bold", 6)
            c.drawCentredString(x1, y1 - ph / 2 - 10, props.get("label", "PANEL"))

        elif elem.element_type == "receptacle":
            # Circle with two lines
            c.circle(x1, y1, 3, fill=0)
            c.line(x1 - 2, y1 + 1, x1 + 2, y1 + 1)
            c.line(x1 - 2, y1 - 1, x1 + 2, y1 - 1)

        elif elem.element_type == "light":
            # 2x4 troffer: rectangle
            lw_pts = 2 * scale
            lh_pts = 1 * scale
            c.rect(x1 - lw_pts / 2, y1 - lh_pts / 2, lw_pts, lh_pts)
            c.line(x1 - lw_pts / 2, y1, x1 + lw_pts / 2, y1)

        elif elem.element_type == "duct":
            c.setLineWidth(0.5)
            c.setDash([4, 2])
            c.line(x1, y1, x2, y2)
            c.setDash([])
            c.setFont("Helvetica", 5)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            c.drawCentredString(mid_x, mid_y + 5, props.get("size", ""))

        elif elem.element_type == "column":
            sz = 0.8 * scale
            c.setFillColorRGB(0, 0, 0)
            c.rect(x1 - sz / 2, y1 - sz / 2, sz, sz, fill=1)
            c.setFillColorRGB(0, 0, 0)

        elif elem.element_type == "beam":
            c.setLineWidth(0.6)
            c.setDash([6, 3])
            c.line(x1, y1, x2, y2)
            c.setDash([])

        elif elem.element_type == "footing":
            c.setLineWidth(0.8)
            c.setDash([4, 2])
            c.rect(x1, y1, (x2 - x1), (y2 - y1))
            c.setDash([])

    # ── DXF Rendering ─────────────────────────────────────

    def render_all_dxf(self, output_dir: str) -> List[str]:
        """Render all sheets as DXF files (raw text format)."""
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for sheet in self.sheets:
            filename = f"{sheet.config.sheet_number}_{_safe_filename(sheet.config.sheet_title)}.dxf"
            filepath = os.path.join(output_dir, filename)
            self._render_sheet_dxf(sheet, filepath)
            paths.append(filepath)
        return paths

    def _render_sheet_dxf(self, sheet: DrawingSheet, filepath: str):
        """Generate a minimal DXF file for a sheet."""
        scale = sheet._scale / PTS_PER_INCH  # inches per real foot
        lines = []
        lines.append("0\nSECTION\n2\nHEADER\n0\nENDSEC")

        # TABLES section — layers
        lines.append("0\nSECTION\n2\nTABLES")
        lines.append("0\nTABLE\n2\nLAYER")
        for layer_name in LAYERS:
            lines.append(f"0\nLAYER\n2\n{layer_name}\n70\n0\n62\n7\n6\nCONTINUOUS")
        lines.append("0\nENDTAB\n0\nENDSEC")

        # ENTITIES section
        lines.append("0\nSECTION\n2\nENTITIES")
        for elem in sheet.config.elements:
            x1 = elem.x1 * scale * 12  # convert to inches in DXF
            y1 = elem.y1 * scale * 12
            x2 = elem.x2 * scale * 12
            y2 = elem.y2 * scale * 12

            if elem.element_type in ("wall", "beam", "duct", "pipe"):
                lines.append(
                    f"0\nLINE\n8\n{elem.layer}\n"
                    f"10\n{x1:.4f}\n20\n{y1:.4f}\n30\n0.0\n"
                    f"11\n{x2:.4f}\n21\n{y2:.4f}\n31\n0.0"
                )
            elif elem.element_type in ("room_label", "note"):
                text = elem.properties.get("name", elem.properties.get("text", ""))
                lines.append(
                    f"0\nTEXT\n8\n{elem.layer}\n"
                    f"10\n{x1:.4f}\n20\n{y1:.4f}\n30\n0.0\n"
                    f"40\n3.0\n1\n{text}"
                )
            elif elem.element_type == "column":
                sz = 10  # inches
                lines.append(
                    f"0\nSOLID\n8\n{elem.layer}\n"
                    f"10\n{x1 - sz / 2:.4f}\n20\n{y1 - sz / 2:.4f}\n"
                    f"11\n{x1 + sz / 2:.4f}\n21\n{y1 - sz / 2:.4f}\n"
                    f"12\n{x1 - sz / 2:.4f}\n22\n{y1 + sz / 2:.4f}\n"
                    f"13\n{x1 + sz / 2:.4f}\n23\n{y1 + sz / 2:.4f}"
                )
        lines.append("0\nENDSEC\n0\nEOF")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

    # ── Project Data Export ────────────────────────────────

    def export_project_json(self, filepath: str) -> str:
        """Export the full project as JSON (for reload/edit)."""
        data = {
            "project_name": self.project_name,
            "project_number": self.project_number,
            "client_name": self.client_name,
            "address": self.address,
            "architect": self.architect,
            "sheet_count": len(self.sheets),
            "sheets": [],
        }
        for sheet in self.sheets:
            sheet_data = sheet.config.to_dict()
            sheet_data["elements"] = [e.to_dict() for e in sheet.config.elements]
            sheet_data["notes"] = sheet.config.notes
            data["sheets"].append(sheet_data)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return filepath


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _format_feet_inches(feet: float) -> str:
    """Convert decimal feet to architectural notation: 12'-6\" """
    ft = int(feet)
    inches = round((feet - ft) * 12)
    if inches == 12:
        ft += 1
        inches = 0
    if inches == 0:
        return f"{ft}'-0\""
    return f"{ft}'-{inches}\""


def _safe_filename(title: str) -> str:
    """Convert a title to a safe filename."""
    return "".join(c if c.isalnum() or c in "-_ " else "" for c in title).strip().replace(" ", "_")[:50]
