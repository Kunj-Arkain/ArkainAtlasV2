"""
engine.construction.tools — Construction Drawing & Spec Tools
"""
from __future__ import annotations
import json, math, os, time
from typing import Any, Dict, List, Optional

IBC_OCCUPANCY = {
    "gas_station":       {"group": "M",   "type": "Mercantile",  "sprinkler_threshold_sf": 12000},
    "convenience_store": {"group": "M",   "type": "Mercantile",  "sprinkler_threshold_sf": 12000},
    "qsr":               {"group": "A-2", "type": "Assembly",    "sprinkler_threshold_sf": 5000},
    "restaurant":        {"group": "A-2", "type": "Assembly",    "sprinkler_threshold_sf": 5000},
    "bar":               {"group": "A-2", "type": "Assembly",    "sprinkler_threshold_sf": 5000},
    "retail_strip":      {"group": "M",   "type": "Mercantile",  "sprinkler_threshold_sf": 12000},
    "dollar_store":      {"group": "M",   "type": "Mercantile",  "sprinkler_threshold_sf": 12000},
    "office":            {"group": "B",   "type": "Business",    "sprinkler_threshold_sf": 12000},
    "warehouse":         {"group": "S-1", "type": "Storage",     "sprinkler_threshold_sf": 12000},
}
OCCUPANT_LOAD_FACTORS = {"M": 60, "A-2": 15, "B": 100, "S-1": 200}
FIXTURE_REQUIREMENTS = {
    "M": {"wc_per_person": {"male": 75, "female": 40}, "lavatory_per_person": {"male": 100, "female": 100}, "drinking_fountain": 100},
    "A-2": {"wc_per_person": {"male": 40, "female": 40}, "lavatory_per_person": {"male": 75, "female": 75}, "drinking_fountain": 100},
    "B": {"wc_per_person": {"male": 50, "female": 50}, "lavatory_per_person": {"male": 80, "female": 80}, "drinking_fountain": 100},
}
NEC_LOADS = {
    "general_lighting": 3.5, "receptacle": 1.0, "signage": 1200,
    "vgt_per_terminal": 300, "vgt_server": 500,
    "cooler_walk_in": 3000, "cooler_reach_in": 1500,
    "fuel_dispenser": 600, "car_wash": 15000,
    "ev_charger_l2": 7200, "ev_charger_l3": 50000,
}
HVAC_LOADS = {
    "gas_station": 35, "convenience_store": 35, "qsr": 45, "restaurant": 45,
    "bar": 40, "retail_strip": 30, "dollar_store": 28, "office": 25,
    "warehouse": 15, "gaming_area": 50,
}
STRUCTURAL_LOADS = {
    "dead_load_roof": 20, "dead_load_floor": 15,
    "live_load_retail": 100, "live_load_office": 50, "live_load_storage": 125,
    "live_load_roof": 20, "snow_load_base": 25, "wind_speed_base": 115, "seismic_sds_base": 0.5,
}


def code_analysis(params):
    ptype = params.get("property_type", "gas_station")
    sqft = params.get("sqft", 3000)
    stories = params.get("stories", 1)
    year_built = params.get("year_built", 2000)
    occ = IBC_OCCUPANCY.get(ptype, IBC_OCCUPANCY["retail_strip"])
    occ_group = occ["group"]
    load_factor = OCCUPANT_LOAD_FACTORS.get(occ_group, 60)
    occupant_load = math.ceil(sqft / load_factor)
    sprinkler_required = sqft > occ["sprinkler_threshold_sf"]
    egress_width = math.ceil(occupant_load * (0.2 if sprinkler_required else 0.3))
    min_exits = 1 if occupant_load < 50 else (2 if occupant_load < 500 else 3)
    fixtures = FIXTURE_REQUIREMENTS.get(occ_group, FIXTURE_REQUIREMENTS["M"])
    male_occ = occupant_load // 2
    female_occ = occupant_load - male_occ
    male_wc = max(1, math.ceil(male_occ / fixtures["wc_per_person"]["male"]))
    female_wc = max(1, math.ceil(female_occ / fixtures["wc_per_person"]["female"]))
    male_lav = max(1, math.ceil(male_occ / fixtures["lavatory_per_person"]["male"]))
    female_lav = max(1, math.ceil(female_occ / fixtures["lavatory_per_person"]["female"]))
    drinking = max(1, math.ceil(occupant_load / fixtures["drinking_fountain"]))
    ada_parking = max(1, math.ceil(sqft / 1000 * 0.04))
    if stories <= 1 and sqft <= 12000: ctype = "V-B (Wood Frame)"
    elif stories <= 2 and sqft <= 24000: ctype = "V-A (Protected Wood)"
    elif sqft <= 36000: ctype = "III-B (Ordinary)"
    else: ctype = "II-B (Non-combustible)"
    return {
        "occupancy_group": occ_group, "occupancy_type": occ["type"],
        "construction_type": ctype, "occupant_load": occupant_load,
        "load_factor_sf_per_person": load_factor,
        "sprinkler": {"required": sprinkler_required, "threshold_sf": occ["sprinkler_threshold_sf"],
                      "type": "NFPA 13" if sprinkler_required else "Not Required"},
        "egress": {"min_exits": min_exits, "total_egress_width_in": egress_width,
                   "max_travel_distance_ft": 250 if sprinkler_required else 200,
                   "exit_sign_illuminated": True, "emergency_lighting": True},
        "plumbing_fixtures": {
            "male_wc": male_wc, "female_wc": female_wc,
            "male_lavatory": male_lav, "female_lavatory": female_lav,
            "drinking_fountain": drinking, "mop_sink": 1, "ada_restroom_required": True,
            "total_fixture_count": male_wc + female_wc + male_lav + female_lav + drinking + 1,
        },
        "ada": {"required": True, "accessible_route": True, "accessible_restroom": True,
                "accessible_parking_spaces": ada_parking, "ramp_needed": year_built < 1992},
        "fire": {"fire_alarm": sqft > 5000 or occ_group.startswith("A"),
                 "fire_extinguishers": max(1, math.ceil(sqft / 3000)),
                 "exit_signs": min_exits, "emergency_lights": max(2, min_exits * 2)},
    }


def electrical_load_calc(params):
    ptype = params.get("property_type", "gas_station")
    sqft = params.get("sqft", 3000)
    terminals = params.get("terminal_count", 0)
    walk_in = params.get("walk_in_coolers", 1 if ptype in ("gas_station", "convenience_store") else 0)
    reach_in = params.get("reach_in_coolers", 4 if ptype in ("gas_station", "convenience_store") else 0)
    dispensers = params.get("fuel_dispensers", 8 if ptype == "gas_station" else 0)
    ev_l2 = params.get("ev_chargers_l2", 0)
    ev_l3 = params.get("ev_chargers_l3", 0)
    car_wash = params.get("car_wash", False)
    gaming_sqft = params.get("gaming_sqft", 0)
    loads = {}
    loads["general_lighting"] = sqft * NEC_LOADS["general_lighting"]
    loads["receptacles"] = sqft * NEC_LOADS["receptacle"]
    hvac_btu = sqft * HVAC_LOADS.get(ptype, 30) + gaming_sqft * HVAC_LOADS["gaming_area"]
    hvac_tons = hvac_btu / 12000
    loads["hvac"] = hvac_tons * 1200
    if terminals > 0:
        loads["gaming_terminals"] = terminals * NEC_LOADS["vgt_per_terminal"]
        loads["gaming_server"] = NEC_LOADS["vgt_server"]
    if walk_in: loads["walk_in_coolers"] = walk_in * NEC_LOADS["cooler_walk_in"]
    if reach_in: loads["reach_in_coolers"] = reach_in * NEC_LOADS["cooler_reach_in"]
    if dispensers: loads["fuel_dispensers"] = dispensers * NEC_LOADS["fuel_dispenser"]
    loads["signage"] = NEC_LOADS["signage"] * 2
    if ev_l2: loads["ev_chargers_l2"] = ev_l2 * NEC_LOADS["ev_charger_l2"]
    if ev_l3: loads["ev_chargers_l3"] = ev_l3 * NEC_LOADS["ev_charger_l3"]
    if car_wash: loads["car_wash"] = NEC_LOADS["car_wash"]
    total_va = sum(loads.values())
    demand_va = 10000 + (total_va - 10000) * 0.50 if total_va > 10000 else total_va
    demand_125 = demand_va * 1.25
    amps = demand_125 / 240
    service = 200 if amps <= 200 else (400 if amps <= 400 else (600 if amps <= 600 else (800 if amps <= 800 else 1200)))
    circuit_20a = math.ceil(loads.get("general_lighting", 0) / 2400) + math.ceil(loads.get("receptacles", 0) / 2400)
    circuit_ded = terminals + walk_in + (1 if car_wash else 0) + ev_l2 + ev_l3
    circuit_hvac = math.ceil(hvac_tons / 5)
    return {
        "loads_va": loads, "total_connected_va": round(total_va),
        "demand_va": round(demand_va), "demand_va_125pct": round(demand_125),
        "service": {"size_amps": service, "voltage": "120/240V 1PH" if service <= 200 else "120/208V 3PH",
                    "main_breaker": f"{service}A", "meter_type": "CT-rated" if service > 200 else "Self-contained"},
        "panel_schedule": {"main_distribution_panel": f"MDP - {service}A",
                           "lighting_panel": f"LP - {min(100, circuit_20a * 2)}A",
                           "hvac_panel": f"HP - {min(100, circuit_hvac * 30)}A" if circuit_hvac > 2 else None,
                           "gaming_panel": "GP - 60A" if terminals > 0 else None},
        "circuits": {"20a_branch": circuit_20a, "dedicated": circuit_ded, "hvac": circuit_hvac,
                     "total": circuit_20a + circuit_ded + circuit_hvac},
        "gaming_electrical": {"terminals": terminals, "dedicated_20a_circuits": terminals,
                              "server_circuit": 1 if terminals > 0 else 0,
                              "total_gaming_va": loads.get("gaming_terminals", 0) + loads.get("gaming_server", 0),
                              "note": "Each VGT requires dedicated 20A/120V circuit per NEC 210.23"} if terminals > 0 else None,
        "hvac_summary": {"total_btu": round(hvac_btu), "tonnage": round(hvac_tons, 1),
                         "units": math.ceil(hvac_tons / 5),
                         "type": "Rooftop Package Unit" if hvac_tons >= 3 else "Split System"},
    }


def hvac_sizing(params):
    ptype = params.get("property_type", "gas_station")
    sqft = params.get("sqft", 3000)
    gaming_sqft = params.get("gaming_sqft", 0)
    cz = params.get("climate_zone", 4)
    base_btu = sqft * HVAC_LOADS.get(ptype, 30)
    gaming_btu = gaming_sqft * HVAC_LOADS["gaming_area"]
    total_cool = base_btu + gaming_btu
    tons = total_cool / 12000
    heat_f = {1:15,2:20,3:25,4:30,5:35,6:40,7:50,8:60}
    heat_btu = sqft * heat_f.get(cz, 30) + gaming_sqft * 20
    if tons <= 2: eq_type, eq_count = "Ductless Mini-Split", math.ceil(tons / 1.5)
    elif tons <= 5: eq_type, eq_count = "Split System", 1
    elif tons <= 25: eq_type, eq_count = "Rooftop Package Unit (RTU)", math.ceil(tons / 7.5)
    else: eq_type, eq_count = "Rooftop Package Unit (RTU)", math.ceil(tons / 10)
    cfm = total_cool / 12000 * 400
    duct_area = cfm / 800
    duct_w = math.sqrt(duct_area * 144) * 1.4
    duct_h = duct_area * 144 / max(duct_w, 1)
    return {
        "cooling": {"total_btu": round(total_cool), "tonnage": round(tons, 1), "base_btu": round(base_btu), "gaming_btu": round(gaming_btu)},
        "heating": {"total_btu": round(heat_btu), "mbh": round(heat_btu / 1000, 1), "fuel_type": "Gas" if cz >= 4 else "Heat Pump"},
        "equipment": {"type": eq_type, "count": eq_count, "size_per_unit_tons": round(tons / max(eq_count,1), 1),
                      "efficiency_seer": 16 if tons <= 5 else 14, "efficiency_afue": 95 if cz >= 5 else 80, "refrigerant": "R-410A"},
        "ductwork": {"total_cfm": round(cfm), "main_trunk_size": f"{round(duct_w)}x{round(duct_h)}",
                     "branch_count": math.ceil(sqft / 200), "diffuser_count": math.ceil(sqft / 150),
                     "return_grille_count": max(2, math.ceil(sqft / 500))},
        "ventilation": {"outdoor_air_cfm": round(sqft * 0.15), "exhaust_cfm": round(sqft * 0.10) if ptype in ("qsr", "restaurant") else 0,
                        "kitchen_hood": ptype in ("qsr", "restaurant")},
        "gaming_zone": {"supplemental_cooling_btu": round(gaming_btu),
                        "note": "Gaming area requires dedicated thermostat zone"} if gaming_sqft > 0 else None,
        "controls": {"thermostat_zones": 1 + (1 if gaming_sqft > 0 else 0) + (1 if ptype in ("qsr","restaurant") else 0),
                     "type": "Programmable digital", "setpoints": {"cooling": 74, "heating": 68}},
    }


def plumbing_design(params):
    ptype = params.get("property_type", "gas_station")
    sqft = params.get("sqft", 3000)
    has_kitchen = ptype in ("qsr", "restaurant", "bar")
    code = code_analysis({"property_type": ptype, "sqft": sqft})
    fx = code["plumbing_fixtures"]
    wfu = (fx["male_wc"]+fx["female_wc"])*4 + (fx["male_lavatory"]+fx["female_lavatory"])*1 + 3 + (6 if has_kitchen else 0) + fx["drinking_fountain"]*0.5
    if wfu <= 15: svc = "3/4 inch"
    elif wfu <= 30: svc = "1 inch"
    elif wfu <= 60: svc = "1-1/4 inch"
    else: svc = "1-1/2 inch"
    gph = (fx["male_lavatory"]+fx["female_lavatory"])*2 + (10 if has_kitchen else 0) + 3
    if gph <= 20: wh_size, wh_type = 20, "Tank (Electric)"
    elif gph <= 40: wh_size, wh_type = 40, "Tank (Gas)"
    else: wh_size, wh_type = gph, "Tankless (Gas)"
    dfu = wfu * 2
    if dfu <= 20: drain = "3 inch"
    elif dfu <= 50: drain = "4 inch"
    else: drain = "6 inch"
    return {
        "fixtures": {**fx}, "water_supply": {"service_size": svc, "total_fixture_units": round(wfu, 1)},
        "hot_water": {"demand_gph": gph, "heater_size_gal": wh_size, "heater_type": wh_type, "temperature": 120},
        "drainage": {"building_drain_size": drain, "total_drainage_fu": dfu,
                     "cleanouts": max(2, math.ceil(sqft / 1500)), "floor_drains": max(1, math.ceil(sqft / 1000)),
                     "grease_trap": has_kitchen, "grease_trap_size": "50 GPM" if has_kitchen else None},
        "gas": {"service_required": True},
        "special": {"fuel_system": ptype == "gas_station", "ust_piping": ptype == "gas_station"},
    }


def structural_calc(params):
    ptype = params.get("property_type", "gas_station")
    sqft = params.get("sqft", 3000)
    width = params.get("width_ft", 50)
    depth = params.get("depth_ft", 60)
    stories = params.get("stories", 1)
    ll = STRUCTURAL_LOADS["live_load_retail"]
    dlr = STRUCTURAL_LOADS["dead_load_roof"]
    dlf = STRUCTURAL_LOADS["dead_load_floor"]
    snow = STRUCTURAL_LOADS["snow_load_base"]
    total_roof = dlr + max(ll * 0.3, snow)
    bays_x = max(2, math.ceil(width / 25))
    bays_y = max(2, math.ceil(depth / 25))
    bay_x = width / bays_x
    bay_y = depth / bays_y
    cols = (bays_x + 1) * (bays_y + 1)
    trib = bay_x * bay_y
    col_load = trib * total_roof * stories
    if col_load < 30000: col_size = "HSS 4x4x1/4"
    elif col_load < 60000: col_size = "W8x31"
    elif col_load < 100000: col_size = "W10x49"
    else: col_size = "W12x65"
    span = max(bay_x, bay_y)
    if span <= 20: beam = "Open Web Steel Joist 18K3"
    elif span <= 30: beam = "W12x26"
    elif span <= 40: beam = "W16x36"
    else: beam = "W21x50"
    soil = 2000
    ft_area = col_load / soil
    ft_size = math.ceil(math.sqrt(ft_area) * 12) / 12
    slab = 5 if ptype in ("gas_station", "warehouse") else 4
    return {
        "loads": {"dead_load_roof_psf": dlr, "live_load_psf": ll, "snow_load_psf": snow, "total_roof_psf": round(total_roof)},
        "framing": {"system": "Steel frame with open web steel joists" if sqft > 2000 else "Wood frame",
                    "column_grid": f"{bay_x:.0f}\' x {bay_y:.0f}\' bays", "bays_x": bays_x, "bays_y": bays_y,
                    "column_count": cols, "column_size": col_size, "beam_size": beam,
                    "joist_size": "18K3" if span <= 30 else "24K4", "roof_deck": "1.5\" Type B, 22 GA"},
        "foundation": {"type": "Spread footings with SOG", "soil_bearing_psf": soil,
                        "footing_size": f"{ft_size:.1f}\' x {ft_size:.1f}\' x 12\"",
                        "slab_thickness_in": slab, "vapor_barrier": "15 mil polyethylene"},
        "lateral_system": {"type": "Steel moment frames" if sqft > 5000 else "Braced frames"},
        "canopy": {"required": True, "type": "Steel canopy with HSS columns",
                   "column_size": "HSS 8x8x3/8", "height": "14\'-0\" clear"} if ptype == "gas_station" else None,
    }


def construction_schedule(params):
    ptype = params.get("property_type", "gas_station")
    gaming = params.get("gaming_eligible", False)
    scope = params.get("scope", "full")
    phases = []
    wk = 1
    phases.append({"phase": "PRE-CONSTRUCTION", "start_week": wk, "tasks": [
        {"task": "Design & Engineering", "weeks": 2, "start": wk},
        {"task": "Permit Applications", "weeks": 1, "start": wk + 1},
        {"task": "Bidding & Contractor Selection", "weeks": 2, "start": wk + 1}]})
    wk += 3
    tasks_p = [{"task": "Building Permit Review", "weeks": 4, "start": wk}]
    if gaming: tasks_p.append({"task": "Gaming License Application", "weeks": 8, "start": wk})
    phases.append({"phase": "PERMITTING", "start_week": wk, "tasks": tasks_p})
    wk += 4
    if scope in ("full", "interior_only"):
        phases.append({"phase": "DEMOLITION", "start_week": wk, "tasks": [
            {"task": "Interior Demolition", "weeks": 1, "start": wk}]})
        wk += 1
    tasks_r = [{"task": "Framing", "weeks": 2, "start": wk}, {"task": "Electrical Rough-in", "weeks": 2, "start": wk+1},
               {"task": "Plumbing Rough-in", "weeks": 2, "start": wk+1}, {"task": "HVAC Rough-in", "weeks": 2, "start": wk+1}]
    if gaming: tasks_r.append({"task": "Gaming Electrical", "weeks": 1, "start": wk+2})
    phases.append({"phase": "ROUGH-IN", "start_week": wk, "tasks": tasks_r})
    wk += 3
    phases.append({"phase": "INSPECTIONS", "start_week": wk, "tasks": [
        {"task": "Rough-in Inspections", "weeks": 1, "start": wk}]})
    wk += 1
    phases.append({"phase": "FINISHES", "start_week": wk, "tasks": [
        {"task": "Insulation & Drywall", "weeks": 2, "start": wk}, {"task": "Ceiling & Flooring", "weeks": 1, "start": wk+2},
        {"task": "Painting & Millwork", "weeks": 1, "start": wk+3}]})
    wk += 4
    tasks_e = [{"task": "Fixture & Equipment Install", "weeks": 1, "start": wk}, {"task": "Signage", "weeks": 1, "start": wk+1}]
    if gaming:
        tasks_e.append({"task": "Gaming Terminal Install", "weeks": 1, "start": wk})
        tasks_e.append({"task": "Security Camera Install", "weeks": 1, "start": wk})
    phases.append({"phase": "EQUIPMENT & FF&E", "start_week": wk, "tasks": tasks_e})
    wk += 2
    tasks_c = [{"task": "Punch List", "weeks": 1, "start": wk}, {"task": "Final Inspections", "weeks": 1, "start": wk+1},
               {"task": "Certificate of Occupancy", "weeks": 1, "start": wk+1}]
    if gaming: tasks_c.append({"task": "Gaming Board Inspection", "weeks": 1, "start": wk+2})
    phases.append({"phase": "CLOSEOUT", "start_week": wk, "tasks": tasks_c})
    wk += 3
    for p in phases:
        p["end_week"] = max(t["start"] + t["weeks"] for t in p["tasks"]) if p["tasks"] else p["start_week"]
    return {"total_duration_weeks": wk, "total_duration_months": round(wk / 4.33, 1), "phases": phases}


def generate_drawing_set(params):
    from engine.drawing_engine import DrawingEngine
    ptype = params.get("property_type", "gas_station")
    sqft = params.get("sqft", 3000)
    width = params.get("width_ft", 50)
    depth = params.get("depth_ft", 60)
    address = params.get("address", "")
    project_name = params.get("project_name", f"{address} Renovation")
    terminals = params.get("terminal_count", 0)
    gaming = params.get("gaming_eligible", False)
    disciplines = params.get("disciplines", ["A", "E", "P", "M", "S"])
    output_dir = params.get("output_dir", "/tmp/drawings")
    engine = DrawingEngine(project_name=project_name, client_name=params.get("client_name", ""), address=address)
    sheets = []
    cover = engine.new_sheet("G0.1", "COVER SHEET", scale="1/8 inch = 1 foot")
    sheets.append("G0.1")
    if "A" in disciplines:
        for num, title in [("A1.0","SITE PLAN"),("A1.1","DEMOLITION PLAN"),("A2.0","FLOOR PLAN - EXISTING"),
                           ("A2.1","FLOOR PLAN - PROPOSED"),("A3.1","REFLECTED CEILING PLAN"),
                           ("A4.1","FRONT ELEVATION"),("A4.2","REAR ELEVATION"),("A4.3","LEFT ELEVATION"),
                           ("A4.4","RIGHT ELEVATION"),("A5.1","BUILDING SECTION"),("A6.1","WALL SECTIONS AND DETAILS")]:
            s = engine.new_sheet(num, title)
            s.wall(0, 0, width, 0); s.wall(width, 0, width, depth)
            s.wall(width, depth, 0, depth); s.wall(0, depth, 0, 0)
            if num == "A2.1" and gaming and terminals > 0:
                gw = max(15, terminals * 4 + 6)
                gd = 15
                s.wall(width-gw, 0, width-gw, gd, thickness=4)
                s.wall(width-gw, gd, width, gd, thickness=4)
                s.room_label(width-gw/2, gd/2, "GAMING AREA", f"{gw*gd} SF")
                s.equipment("VGT", width-gw+3, gd/2, count=terminals, spacing=4)
            sheets.append(num)
    if "S" in disciplines:
        for num, title in [("S1.1","FOUNDATION PLAN"),("S2.1","FRAMING PLAN"),("S3.1","STRUCTURAL DETAILS")]:
            s = engine.new_sheet(num, title)
            if num == "S2.1":
                sc = structural_calc({"property_type": ptype, "sqft": sqft, "width_ft": width, "depth_ft": depth})
                bx, by = sc["framing"]["bays_x"], sc["framing"]["bays_y"]
                for ix in range(bx+1):
                    for iy in range(by+1):
                        s.column(ix*width/bx, iy*depth/by, sc["framing"]["column_size"])
            sheets.append(num)
    if "E" in disciplines:
        for num, title in [("E1.1","ELECTRICAL POWER PLAN"),("E2.1","LIGHTING PLAN"),("E3.1","PANEL SCHEDULES")]:
            s = engine.new_sheet(num, title)
            if num == "E1.1":
                s.electrical_panel(2, depth-2, amps=200, label="MDP")
                if terminals > 0: s.electrical_panel(width-4, 2, amps=60, label="GP")
            sheets.append(num)
    if "P" in disciplines:
        for num, title in [("P1.1","PLUMBING PLAN"),("P2.1","PLUMBING DETAILS")]:
            engine.new_sheet(num, title); sheets.append(num)
    if "M" in disciplines:
        for num, title in [("M1.1","HVAC PLAN"),("M2.1","HVAC SCHEDULES")]:
            engine.new_sheet(num, title); sheets.append(num)
    os.makedirs(output_dir, exist_ok=True)
    pdfs = engine.render_all_pdf(output_dir)
    dxfs = engine.render_all_dxf(output_dir)
    combined = os.path.join(output_dir, "FULL_SET.pdf")
    engine.render_combined_pdf(combined)
    proj = engine.export_project_json(os.path.join(output_dir, "project.json"))
    return {"project_name": project_name, "sheets_created": sheets, "sheet_count": len(sheets),
            "disciplines": disciplines, "pdf_files": pdfs, "dxf_files": dxfs,
            "combined_pdf": combined, "project_file": proj, "output_dir": output_dir}


def generate_spec_book(params):
    ptype = params.get("property_type", "gas_station")
    sqft = params.get("sqft", 3000)
    gaming = params.get("gaming_eligible", False)
    terminals = params.get("terminal_count", 0)
    project_name = params.get("project_name", "")
    output_path = params.get("output_path", "/tmp/specs/spec_book.pdf")
    sections = [
        {"division": "01", "title": "GENERAL REQUIREMENTS", "sections": [
            {"number": "01 10 00", "title": "Summary of Work", "content": f"Project: {project_name}, {sqft:,} SF {ptype}"},
            {"number": "01 40 00", "title": "Quality Requirements"}, {"number": "01 70 00", "title": "Execution and Closeout"}]},
        {"division": "03", "title": "CONCRETE", "sections": [
            {"number": "03 30 00", "title": "Cast-in-Place Concrete", "content": "Footings 3000 PSI, Slab 4000 PSI"}]},
        {"division": "05", "title": "METALS", "sections": [
            {"number": "05 12 00", "title": "Structural Steel", "content": "ASTM A992 Grade 50"}]},
        {"division": "07", "title": "THERMAL & MOISTURE", "sections": [
            {"number": "07 21 00", "title": "Insulation", "content": "Roof R-30, Walls R-19"},
            {"number": "07 52 00", "title": "Roofing"}]},
        {"division": "08", "title": "OPENINGS", "sections": [
            {"number": "08 11 00", "title": "Metal Doors"}, {"number": "08 41 00", "title": "Storefronts"}]},
        {"division": "09", "title": "FINISHES", "sections": [
            {"number": "09 29 00", "title": "Gypsum Board"}, {"number": "09 51 00", "title": "Acoustical Ceilings"},
            {"number": "09 65 00", "title": "Resilient Flooring"}, {"number": "09 91 00", "title": "Painting"}]},
        {"division": "22", "title": "PLUMBING", "sections": [
            {"number": "22 11 00", "title": "Water Distribution"}, {"number": "22 13 00", "title": "Sanitary Sewerage"},
            {"number": "22 40 00", "title": "Plumbing Fixtures"}]},
        {"division": "23", "title": "HVAC", "sections": [
            {"number": "23 31 00", "title": "Ductwork"}, {"number": "23 74 00", "title": "Packaged HVAC Equipment"}]},
        {"division": "26", "title": "ELECTRICAL", "sections": [
            {"number": "26 24 00", "title": "Panelboards"}, {"number": "26 51 00", "title": "Interior Lighting"},
            {"number": "26 56 00", "title": "Exterior Lighting"}]},
    ]
    if gaming and terminals > 0:
        sections.append({"division": "11", "title": "GAMING EQUIPMENT", "sections": [
            {"number": "11 48 00", "title": "Gaming Infrastructure",
             "content": f"{terminals} dedicated 20A circuits, CAT6 per terminal, cameras, ATM"}]})
    if ptype == "gas_station":
        sections.append({"division": "33", "title": "FUEL SYSTEM", "sections": [
            {"number": "33 56 00", "title": "Fuel Storage Tanks", "content": "Double-wall fiberglass USTs per EPA 40 CFR 280"}]})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _render_spec_pdf(sections, project_name, output_path)
    json_path = output_path.replace(".pdf", ".json")
    with open(json_path, "w") as f: json.dump({"project": project_name, "divisions": sections}, f, indent=2)
    return {"project_name": project_name, "division_count": len(sections),
            "section_count": sum(len(d["sections"]) for d in sections),
            "divisions": [{"number": d["division"], "title": d["title"], "sections": len(d["sections"])} for d in sections],
            "pdf_path": output_path, "json_path": json_path}


def _render_spec_pdf(sections, project_name, filepath):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    c = canvas.Canvas(filepath, pagesize=letter)
    w, h = letter
    m = 1 * inch
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(w/2, h/2+60, "PROJECT MANUAL")
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(w/2, h/2+30, "TECHNICAL SPECIFICATIONS")
    c.setFont("Helvetica", 14)
    c.drawCentredString(w/2, h/2-10, project_name)
    c.setFont("Helvetica", 10)
    c.drawCentredString(w/2, h/2-40, f"Date: {time.strftime('%B %d, %Y')}")
    c.showPage()
    y = h - m
    c.setFont("Helvetica-Bold", 16)
    c.drawString(m, y, "TABLE OF CONTENTS"); y -= 30
    c.setFont("Helvetica", 10)
    for div in sections:
        c.drawString(m, y, f"Division {div['division']} - {div['title']}"); y -= 14
        for sec in div["sections"]:
            c.drawString(m+30, y, f"{sec['number']}  {sec['title']}"); y -= 12
            if y < m+30: c.showPage(); y = h - m
        y -= 8
    c.showPage()
    for div in sections:
        y = h - m
        c.setFont("Helvetica-Bold", 14)
        c.drawString(m, y, f"DIVISION {div['division']} - {div['title']}"); y -= 30
        for sec in div["sections"]:
            if y < m+80: c.showPage(); y = h - m
            c.setFont("Helvetica-Bold", 11)
            c.drawString(m, y, f"SECTION {sec['number']}"); y -= 16
            c.drawString(m, y, sec["title"]); y -= 20
            c.setFont("Helvetica", 9)
            for line in sec.get("content", "Refer to drawings.").split("\n"):
                if y < m+20: c.showPage(); y = h - m
                c.drawString(m+20, y, line); y -= 12
            y -= 10
        c.showPage()
    c.save()
