"""
api/utils_folder/synthetic_generator.py

Synthetic dataset generator with named scenario profiles.

Machines have realistic role-based names (cutting / assembly / molding),
products use wire-harness nomenclature, and process steps follow a factory
flow with category-specific operations and qualifiers.

Public API
----------
SCENARIO_CATALOG          : dict    – named scenario profiles (backend source of truth)
get_scenario_catalog()    : dict    – serialisable copy of SCENARIO_CATALOG
generate_synthetic_dataset()        – low-level generator (full parameter control)
generate_scenario_dataset()         – named-scenario shortcut
insert_synthetic_dataset_into_db()  – persist a generated dataset into Django ORM
"""
from __future__ import annotations

import re
import random
import string
import time
from typing import Any

# ---------------------------------------------------------------------------
# Scenario Catalog  (backend source of truth for all four profiles)
# ---------------------------------------------------------------------------

SCENARIO_CATALOG: dict[str, dict] = {
    "Low Demand / High Availability": {
        "icon":     "🟢",
        "color":    "#22c55e",
        "subtitle": "Slack capacity — baseline stress-free scenario",
        "description": (
            "Few products with modest demand scheduled across a generous machine pool. "
            "Machines run well below capacity, producing short makespans with no bottlenecks. "
            "Use this as a sanity-check baseline or to tune scheduler parameters."
        ),
        "tags":    ["baseline", "slack", "easy"],
        "params":  {
            "num_products":      5,
            "num_machines":      8,
            "demand_min":       100,
            "demand_max":     2_000,
            "steps_per_product": 3,
            "seed":             42,
        },
        "expected": {
            "Avg utilisation":  "< 40%",
            "Makespan":         "Short",
            "Bottleneck risk":  "Low",
            "Scheduling style": "Easy dispatch",
        },
    },
    "High Demand / Bottleneck Machines": {
        "icon":     "🔴",
        "color":    "#ef4444",
        "subtitle": "Demand spike on constrained machines",
        "description": (
            "Many products with high annual demand all competing for a small set of machines. "
            "At least one machine will hit critical utilisation (>85%). "
            "Tests the scheduler's ability to sequence under heavy load and PuLP optimisation."
        ),
        "tags":    ["bottleneck", "stress", "hard"],
        "params":  {
            "num_products":      20,
            "num_machines":       3,
            "demand_min":    10_000,
            "demand_max":   100_000,
            "steps_per_product":  6,
            "seed":              77,
        },
        "expected": {
            "Avg utilisation":  "> 85%",
            "Makespan":         "Long",
            "Bottleneck risk":  "High",
            "Scheduling style": "MILP-critical",
        },
    },
    "Capacity-Constrained Scheduling": {
        "icon":     "🟡",
        "color":    "#f59e0b",
        "subtitle": "Moderate demand, tight machine allocation",
        "description": (
            "A medium-sized product mix with moderate demand pressed against a limited "
            "machine count. No single machine is overwhelmed, but the scheduler must "
            "carefully interleave batches. Good for testing gap-elimination and compaction."
        ),
        "tags":    ["constrained", "medium", "interleaving"],
        "params":  {
            "num_products":      12,
            "num_machines":       4,
            "demand_min":      5_000,
            "demand_max":     30_000,
            "steps_per_product":  5,
            "seed":             123,
        },
        "expected": {
            "Avg utilisation":  "60–85%",
            "Makespan":         "Medium",
            "Bottleneck risk":  "Medium",
            "Scheduling style": "Gap-elimination focus",
        },
    },
    "Variable Processing Time": {
        "icon":     "🔵",
        "color":    "#60a5fa",
        "subtitle": "Highly heterogeneous step durations",
        "description": (
            "Products with a very wide demand spread (500 → 50 000 units) and randomised "
            "routing depths (3–8 steps). Exposes scheduling sensitivity to time variability "
            "and produces interesting Gantt shapes — ideal for visual exploration."
        ),
        "tags":    ["variable", "mixed", "exploration"],
        "params":  {
            "num_products":      10,
            "num_machines":       6,
            "demand_min":        500,
            "demand_max":     50_000,
            "steps_per_product":  0,   # 0 = random 3–8 per product
            "seed":             999,
        },
        "expected": {
            "Avg utilisation":  "Varies",
            "Makespan":         "Variable",
            "Bottleneck risk":  "Unpredictable",
            "Scheduling style": "Exploration",
        },
    },
}

# ---------------------------------------------------------------------------
# Machine Grammar
# ---------------------------------------------------------------------------

_MACHINE_BRANDS = [
    "Vortex", "Nexus", "Helios", "Axon", "Praxis",
    "Vektor", "Orbis", "Ferrum", "Stratos", "Callix",
    "Dynex", "Kronos", "Lumis", "Torq", "Zephyr",
    "Cryon", "Valdex", "Pinnex", "Solux", "Arcus",
]
_MACHINE_MODEL_PREFIXES = ["ZX", "TX", "MX", "CX", "RX", "EX", "SX", "AX", "DX", "FX"]
_MACHINE_MODEL_NUMBERS  = list(range(200, 1000, 20))

_STATION_OPERATIONS = [
    "Wire Cutting", "Cable Stripping", "Crimping", "Conductor Joining",
    "Terminal Insertion", "Connector Assembly", "Wire Rolling",
    "Tape Wrapping", "Tube Assembly", "Grommet Fitting",
    "Seal Insertion", "Cable Bundling", "Continuity Testing",
    "Pull-Force Testing", "Visual Inspection",
]

_MOLDING_BRANDS = [
    "Engel", "Battenfeld", "Demag", "Wittmann", "Haitian",
    "Chen Hsong", "Toyo", "Toshiba", "Niigata", "Sumitomo",
]
_MOLDING_MODEL_PREFIXES = ["ST", "XT", "GT", "HT", "ET", "MT"]
_MOLDING_MODEL_NUMBERS  = list(range(250, 800, 25))

# ---------------------------------------------------------------------------
# Product Grammar
# ---------------------------------------------------------------------------

_WIRE_CONFIGS = [
    ("Single Wire",    "single",  1),
    ("2-Wire Jacket",  "jacket",  2),
    ("3-Wire Jacket",  "jacket",  3),
    ("4-Wire Jacket",  "jacket",  4),
    ("5-Wire Jacket",  "jacket",  5),
    ("Twisted Pair",   "twisted", 2),
    ("Twisted Quad",   "twisted", 4),
    ("Shielded 2-Wire","shielded",2),
    ("Coaxial Cable",  "coaxial", 1),
]

_CONNECTOR_FAMILIES = [
    ("DCC",      ["1x", "2x", "3x"]),
    ("MQS",      ["1x", "2x"]),
    ("JPT",      ["1x", "2x", "3x"]),
    ("HSD",      ["1x", "2x"]),
    ("FAKRA",    ["1x", "2x"]),
    ("HVL",      ["1x"]),
    ("AMP",      ["1x", "2x", "3x"]),
    ("RAST",     ["1x", "2x"]),
    ("Micro-Fit",["1x", "2x"]),
    ("Mini-Fit", ["1x", "2x", "3x"]),
]

_DCC_TYPE_MAP = {"1x": "Single", "2x": "Dual", "3x": "Triple"}

_VARIANT_SUFFIXES = [""] * 3 + list("ABCDEFGHJKMNPRST")

_DEMAND_WEIGHTS = [0.2, 0.5, 0.3]
_DEMAND_RANGES  = {"low": (80, 400), "mid": (400, 1500), "high": (1500, 4500)}
_BATCH_SIZES    = [50, 100, 150, 200, 250]

# ---------------------------------------------------------------------------
# Process Step Grammar
# ---------------------------------------------------------------------------

_STEP_OPERATIONS = {
    "cut_strip": [
        "Cutting & Stripping",
        "Precision Cutting",
        "Stripping & End-Preparation",
        "Jacket Removal & Stripping",
        "Wire Cutting & Separation",
        "Multi-Wire Stripping",
        "Conductor Exposure",
    ],
    "crimp": [
        "Terminal Crimping",
        "Crimping & Sleeve Insertion",
        "Contact Crimping",
        "End-Crimp Application",
        "Wire Crimping & Assembly",
        "Seal Crimping",
        "Ferrule Crimping",
    ],
    "assembly": [
        "Connector Housing Assembly",
        "Terminal Insertion",
        "Connector Body Assembly",
        "Pin Insertion & Lock",
        "Connector Mating & Latching",
        "Seal & Connector Assembly",
        "Manual Connector Build",
        "Sub-Assembly Integration",
    ],
    "tube_grommet": [
        "Corrugated Tube Fitting",
        "PUR Tube Assembly",
        "Grommet Insertion",
        "Tube & Grommet Sub-Assembly",
        "Protective Sleeve Fitting",
        "Conduit Assembly",
        "Rubber Grommet Seating",
    ],
    "wrap_tape": [
        "Wire Pair Rolling & Taping",
        "Cable Bundling & Taping",
        "Spiral Wrap Application",
        "PVC Tape Wrapping",
        "Protective Taping",
        "Harness Taping",
        "Cloth Tape Application",
    ],
    "mold": [
        "Overmolding",
        "Injection Overmolding",
        "Connector Overmolding",
        "Strain-Relief Molding",
        "Encapsulation Molding",
        "Insert Molding",
    ],
    "test": [
        "Electrical Continuity Test",
        "Pull-Force Verification",
        "Visual Quality Inspection",
        "HV Withstand Test",
        "Seal Integrity Check",
    ],
}

_STEP_QUALIFIERS = {
    "cut_strip": [
        "{n}-Wire Jacket, {l}mm",
        "Single Conductors, {l}mm Cut Length",
        "{n}-Core Cable, {l}mm",
        "Twisted {n}-Wire, Strip {s}mm / Cut {l}mm",
        "Jacket Cable {n}-Wire, Strip Length {s}mm",
    ],
    "crimp": [
        "{connector} Contact, Wire Gauge {g} AWG",
        "{connector} Terminal, Crimp Force {f}N",
        "Tin-Plated Contact, {g} AWG",
        "Double Crimp — Insulation + Conductor",
        "{connector} Seal Crimp, {g} AWG",
    ],
    "assembly": [
        "{connector} Housing, {p}-Pin",
        "{connector} Body, {p}-Way",
        "{p}-Position {connector}, Locking Clip",
        "Coding — {orient}",
        "{connector} Connector, {cpa}",
    ],
    "tube_grommet": [
        "{td}mm Tube, {l}mm Length",
        "Rubber Grommet {td}mm, {l}mm Cable",
        "Corrugated Tube {td}mm, {l}mm",
        "ID {td}mm PUR Tube, {l}mm",
    ],
    "wrap_tape": [
        "Wire Pairs, {l}mm Overlap",
        "Full Harness, {t}mm Tape Width",
        "Spiral 50% Overlap, {l}mm Section",
        "{t}mm PVC Tape, {l}mm Bundle Length",
    ],
    "mold": [
        "{orient} Connector, {cpa}",
        "Straight-Exit, {cpa}",
        "{orient} Bend, CPA Clip {cpa}",
    ],
    "test": [
        "All Circuits, {v}V Continuity",
        "Crimp Pull-Force, Sample {pct}%",
        "Visual — Seal & Terminal Seating",
        "{v}V HV Isolation, 1s Dwell",
    ],
}

_STEP_FILLERS: dict[str, list] = {
    "n":         [2, 3, 4, 5, 6, 8, 10],
    "l":         [80, 100, 120, 150, 200, 250, 300],
    "s":         [5, 6, 8, 10, 12],
    "g":         ["0.35", "0.5", "0.75", "1.0", "1.5", "2.5"],
    "f":         [30, 40, 50, 60, 80, 100],
    "v":         [12, 24, 48, 60],
    "p":         [2, 3, 4, 6, 8, 12],
    "t":         [9, 15, 19, 25],
    "td":        ["3.5", "5.0", "6.0", "7.0", "8.0"],
    "cpa":       ["With CPA", "No CPA"],
    "orient":    ["180 Straight", "90 Bottom", "90 Right", "45 Angled"],
    "connector": [],  # filled per product
    "pct":       [5, 10, 20],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fill_qualifier(template: str, connector: str, rng: random.Random) -> str:
    """Fill {placeholders} in a step qualifier template."""
    fillers = dict(_STEP_FILLERS)
    fillers["connector"] = [connector]
    t = template
    for key, values in fillers.items():
        placeholder = f"{{{key}}}"
        if placeholder in t and values:
            t = t.replace(placeholder, str(rng.choice(values)))
    t = re.sub(r"\{[^}]+\}", "", t)
    return t.strip().strip(",").strip()


def _generate_machines(n: int, rng: random.Random) -> list[dict]:
    """Generate n machines with role-based names (cutting / assembly / molding)."""
    machines: list[dict] = []
    used_names: set = set()

    n_molding  = max(1, round(n * 0.30))
    n_cutting  = max(1, round(n * 0.30))
    n_stations = max(1, n - n_molding - n_cutting)

    for _ in range(n_cutting):
        b1, b2 = rng.sample(_MACHINE_BRANDS, 2)
        m1 = f"{rng.choice(_MACHINE_MODEL_PREFIXES)}-{rng.choice(_MACHINE_MODEL_NUMBERS)}"
        m2 = f"{rng.choice(_MACHINE_MODEL_PREFIXES)}-{rng.choice(_MACHINE_MODEL_NUMBERS)}"
        name = f"{b1} {m1} / {b2} {m2}"
        if name in used_names:
            m1   = f"{rng.choice(_MACHINE_MODEL_PREFIXES)}-{rng.choice(_MACHINE_MODEL_NUMBERS)}"
            name = f"{b1} {m1} / {b2} {m2}"
        used_names.add(name)
        machines.append({"name": name, "available_hours_per_day": 24.0, "_role": "cutting"})

    ops = rng.sample(_STATION_OPERATIONS, min(n_stations, len(_STATION_OPERATIONS)))
    if len(ops) < n_stations:
        ops += rng.choices(_STATION_OPERATIONS, k=n_stations - len(ops))
    for op in ops[:n_stations]:
        name = f"{op} Station"
        if name in used_names:
            name = f"{op} & QC Station"
        used_names.add(name)
        machines.append({"name": name, "available_hours_per_day": 16.0, "_role": "assembly"})

    per_group = max(4, round(12 / max(n_molding, 1)))
    start_idx = 1
    for _ in range(n_molding):
        brand = rng.choice(_MOLDING_BRANDS)
        model = f"{rng.choice(_MOLDING_MODEL_PREFIXES)}{rng.choice(_MOLDING_MODEL_NUMBERS)}"
        end_idx = start_idx + per_group - 1
        name = f"{brand} {model} Machine {start_idx}-{end_idx}"
        used_names.add(name)
        machines.append({"name": name, "available_hours_per_day": 24.0, "_role": "molding"})
        start_idx = end_idx + 1

    return machines


def _scaled_demand_ranges(demand_range: tuple[int, int]) -> dict[str, tuple[int, int]]:
    """
    Split a (min, max) demand range into low / mid / high tier sub-ranges
    using the same tier-weight proportions as _DEMAND_WEIGHTS [0.2, 0.5, 0.3].

    This ensures that the tiered distribution respects the caller-supplied
    demand_range rather than always using the hard-coded 80–4500 defaults.
    """
    d_min, d_max = demand_range
    span = max(d_max - d_min, 3)

    low_top = d_min + max(1, int(span * 0.20))
    mid_top = d_min + max(2, int(span * 0.70))

    return {
        "low":  (d_min,    max(d_min + 1, low_top)),
        "mid":  (low_top,  max(low_top + 1, mid_top)),
        "high": (mid_top,  d_max),
    }


def _generate_products(
    n: int,
    rng: random.Random,
    demand_range: tuple[int, int] | None = None,
) -> list[dict]:
    """Generate n products with realistic wire-harness descriptions.

    When demand_range is provided it overrides the module-level _DEMAND_RANGES
    so the generated demands are scaled to the caller's (min, max) bounds.
    """
    effective_ranges = (
        _scaled_demand_ranges(demand_range) if demand_range is not None else _DEMAND_RANGES
    )

    products: list[dict] = []
    used_descs: set = set()

    part_formats = [
        lambda: f"{rng.choice(list(string.ascii_uppercase))}{rng.randint(1,9)}{rng.randint(1000,9999)}",
        lambda: f"{rng.randint(10,99)}{rng.choice(list('ABCDEFGHJKLMNPQRSTVWXYZ'))}{rng.randint(100,999)}",
        lambda: f"{''.join(rng.choices(string.ascii_uppercase, k=2))}{rng.randint(10000,99999)}",
        lambda: f"{rng.choice(list(string.ascii_uppercase))}{rng.randint(10,99)}-{rng.randint(100,999)}",
        lambda: f"{rng.randint(100,999)}-{rng.choice(list(string.ascii_uppercase))}{rng.randint(10,99)}",
    ]

    for i in range(1, n + 1):
        wire_label, wire_category, _ = rng.choice(_WIRE_CONFIGS)
        conn_family, conn_multipliers = rng.choice(_CONNECTOR_FAMILIES)
        conn_multiplier = rng.choice(conn_multipliers)
        connector_str = f"{conn_multiplier}{conn_family}"
        dcc_type      = f"{_DCC_TYPE_MAP.get(conn_multiplier, conn_multiplier)} {conn_family}"

        part_fn   = rng.choice(part_formats)
        part_code = part_fn()
        suffix    = rng.choice(_VARIANT_SUFFIXES)

        description = f"{wire_label} {connector_str} Module {part_code}{suffix}"
        if description in used_descs:
            description = f"{wire_label} {connector_str} Module {part_fn()}{suffix}"
        used_descs.add(description)

        tier   = rng.choices(["low", "mid", "high"], weights=_DEMAND_WEIGHTS)[0]
        demand = rng.randint(*effective_ranges[tier])
        batch  = rng.choice(_BATCH_SIZES)

        products.append({
            "item":           i,
            "sap_tn":         f"TN-{rng.randint(100000, 999999)}",
            "sap_pl":         f"PL-{rng.randint(1000, 9999)}" if rng.random() > 0.25 else None,
            "dcc_type":       dcc_type,
            "description":    description,
            "demand_2024":    demand,
            "batch_size":     batch,
            "num_batches":    max(1, round(demand / batch)),
            "_wire_category": wire_category,
            "_connector":     conn_family,
        })

    return products


def _build_step(op_category: str, connector: str, machine_name: str,
                step_num: int, product_item: int, cycle_range: tuple,
                workers: float, rng: random.Random) -> dict:
    """Build one process step dict (internal format)."""
    operation      = rng.choice(_STEP_OPERATIONS[op_category])
    qualifier_tmpl = rng.choice(_STEP_QUALIFIERS[op_category])
    qualifier      = _fill_qualifier(qualifier_tmpl, connector, rng)
    step_name      = f"{operation} — {qualifier}" if qualifier else operation

    return {
        "product_item":       product_item,
        "step_number":        step_num,
        "machine_name":       machine_name,
        "step_name":          step_name,
        "cycle_time_seconds": round(rng.uniform(*cycle_range), 2),
        "workers_required":   workers,
    }


def _generate_process_steps(
    products: list,
    machines: list,
    avg_steps: int | None,
    rng: random.Random,
) -> list[dict]:
    """
    Assign process steps following factory flow:

    1. Cut & Strip  → cutting machine  (always)
    2. Crimp        → cutting machine  (always)
    3. Tube/Grommet → assembly station [optional ~65%]
    4. Connector Assembly → assembly station (always)
    5. Wrap / Tape  → assembly station [optional]
    6. Overmolding  → molding machine  (always last)
    +  Test         → assembly station [optional ~50%]

    When avg_steps is None or 0, each product gets a random depth of 3–8 steps
    (Variable Processing Time scenario).
    """
    cutting_m  = [m["name"] for m in machines if m.get("_role") == "cutting"]
    assembly_m = [m["name"] for m in machines if m.get("_role") == "assembly"]
    molding_m  = [m["name"] for m in machines if m.get("_role") == "molding"]

    if not cutting_m:  cutting_m  = [machines[0]["name"]]
    if not assembly_m: assembly_m = [machines[min(1, len(machines)-1)]["name"]]
    if not molding_m:  molding_m  = [machines[-1]["name"]]

    # Variable depth mode: each product independently picks a target length
    variable_depth = (avg_steps is None or avg_steps == 0)

    all_steps: list[dict] = []

    for product in products:
        item      = product["item"]
        connector = product["_connector"]
        category  = product["_wire_category"]
        steps: list[dict] = []
        step_num = 1

        # Per-product target when using variable depth
        target_base = rng.randint(3, 8) if variable_depth else avg_steps

        def add(op_cat, machine_pool, cycle_range, workers=0.5):
            nonlocal step_num
            steps.append(_build_step(
                op_cat, connector, rng.choice(machine_pool),
                step_num, item, cycle_range, workers, rng,
            ))
            step_num += 1

        # 1. Cut & Strip (always)
        add("cut_strip", cutting_m, (5.5, 9.5), workers=0.5)

        # 2. Crimp (always)
        add("crimp", cutting_m, (5.0, 9.0), workers=0.5)

        # 3. Tube / Grommet (probability by wire category)
        tube_prob = 0.65 if category in ("jacket", "shielded") else 0.25
        if rng.random() < tube_prob:
            add("tube_grommet", assembly_m, (9.0, 20.0), workers=1.0)

        # 4. Connector Assembly (always)
        add("assembly", assembly_m, (7.0, 14.0), workers=1.0)

        # 5. Wrap / Tape (probability by wire category)
        wrap_probs = {"twisted": 0.80, "coaxial": 0.60,
                      "jacket": 0.30, "shielded": 0.45, "single": 0.20}
        if rng.random() < wrap_probs.get(category, 0.30):
            add("wrap_tape", assembly_m, (7.5, 13.0), workers=0.5)

        # 6. Optional extra assembly step to hit target depth
        if len(steps) < target_base - 1:
            add("assembly", assembly_m, (8.0, 15.0), workers=1.0)

        # 7. Test step (50% chance)
        if rng.random() < 0.50:
            add("test", assembly_m, (4.0, 8.0), workers=0.5)

        # 8. Overmolding — always last
        add("mold", molding_m, (13.0, 22.0), workers=1.0)

        all_steps.extend(steps)

    return all_steps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_scenario_catalog() -> list[dict]:
    """Return a JSON-serialisable list of scenario profile metadata.

    Each item includes name, icon, color, subtitle, description, tags,
    params, and expected outcomes — suitable for the /api/scenario-profiles/
    endpoint.
    """
    result = []
    for name, profile in SCENARIO_CATALOG.items():
        result.append({
            "name":        name,
            "icon":        profile["icon"],
            "color":       profile["color"],
            "subtitle":    profile["subtitle"],
            "description": profile["description"],
            "tags":        profile["tags"],
            "params":      profile["params"],
            "expected":    profile["expected"],
        })
    return result


def generate_synthetic_dataset(
    num_products: int = 10,
    num_machines: int | None = None,
    steps_per_product: int | None = 5,
    seed: int | None = None,
    demand_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """
    Generate a complete synthetic production dataset.

    Parameters
    ----------
    num_products       : int        Products to generate (default 10)
    num_machines       : int|None   Machines to include  (default min(num_products+3, 15))
    steps_per_product  : int|None   Average routing depth per product.
                                    None or 0 → random 3–8 per product (variable depth).
    seed               : int|None   RNG seed; None → use current time (fresh each call)
    demand_range       : tuple|None (demand_min, demand_max) to override the default
                                    tiered demand ranges (80–4500).  When provided,
                                    the tiered ranges are scaled proportionally to fit
                                    within the supplied bounds.

    Returns
    -------
    dict with keys:
        "machines" : list[dict] — name, available_hours_per_day
        "products" : list[dict] — item, sap_tn, sap_pl, dcc_type, description,
                                  demand_2024, batch_size, num_batches
        "routing"  : list[dict] — item, machine, step, name, time, workers
        "metadata" : dict       — generation parameters
    """
    if seed is None:
        seed = int(time.time())
    rng = random.Random(seed)

    if num_machines is None:
        num_machines = min(num_products + 3, 15)
    num_machines = max(4, min(num_machines, 25))

    # Treat 0 as "random variable depth" (same as None)
    effective_steps = steps_per_product if (steps_per_product and steps_per_product > 0) else None

    machines  = _generate_machines(num_machines, rng)
    products  = _generate_products(num_products, rng, demand_range=demand_range)
    raw_steps = _generate_process_steps(products, machines, effective_steps, rng)

    clean_machines = [{k: v for k, v in m.items() if not k.startswith("_")} for m in machines]
    clean_products = [{k: v for k, v in p.items() if not k.startswith("_")} for p in products]

    routing = [
        {
            "item":    s["product_item"],
            "machine": s["machine_name"],
            "step":    s["step_number"],
            "name":    s["step_name"],
            "time":    s["cycle_time_seconds"],
            "workers": s["workers_required"],
        }
        for s in raw_steps
    ]

    steps_label = (
        effective_steps if effective_steps is not None else "random 3–8"
    )

    metadata = {
        "dataset_type":        "synthetic",
        "num_products":        num_products,
        "num_machines":        num_machines,
        "steps_per_product":   steps_label,
        "seed":                seed,
        "demand_range":        list(demand_range) if demand_range else None,
        "total_routing_steps": len(routing),
    }

    return {
        "machines": clean_machines,
        "products": clean_products,
        "routing":  routing,
        "metadata": metadata,
    }


def generate_scenario_dataset(
    scenario_name: str,
    seed_override: int | None = None,
) -> dict[str, Any]:
    """
    Generate a dataset for a named scenario profile.

    Parameters
    ----------
    scenario_name  : str  Key from SCENARIO_CATALOG
    seed_override  : int  Override the profile's default seed (optional)

    Returns
    -------
    Same dict as generate_synthetic_dataset(), with metadata["scenario"] set.
    """
    if scenario_name not in SCENARIO_CATALOG:
        valid = list(SCENARIO_CATALOG.keys())
        raise ValueError(
            f"Unknown scenario {scenario_name!r}. Valid profiles: {valid}"
        )

    profile = SCENARIO_CATALOG[scenario_name]
    p       = profile["params"]

    seed   = seed_override if seed_override is not None else p["seed"]
    steps  = p["steps_per_product"]

    dataset = generate_synthetic_dataset(
        num_products      = p["num_products"],
        num_machines      = p["num_machines"],
        steps_per_product = steps if steps > 0 else None,
        seed              = seed,
        demand_range      = (p["demand_min"], p["demand_max"]),
    )
    dataset["metadata"]["scenario"] = scenario_name
    return dataset


def insert_synthetic_dataset_into_db(dataset: dict, clear_existing: bool = True) -> dict:
    """
    Persist a synthetic dataset into the Django ORM models.

    Parameters
    ----------
    dataset        : dict  output of generate_synthetic_dataset() or generate_scenario_dataset()
    clear_existing : bool  wipe existing data before inserting (default True)

    Returns
    -------
    dict with counts: products_created, machines_created, process_steps_created, metadata
    """
    from api.models import Product, Machine, ProcessStep, ProductionSchedule  # noqa

    if clear_existing:
        ProductionSchedule.objects.all().delete()
        ProcessStep.objects.all().delete()
        Product.objects.all().delete()
        Machine.objects.all().delete()

    machine_objs: dict[str, Machine] = {}
    for m in dataset["machines"]:
        obj = Machine.objects.create(
            name                    = m["name"],
            available_hours_per_day = m["available_hours_per_day"],
        )
        machine_objs[m["name"]] = obj

    product_objs: dict[int, Product] = {}
    for p in dataset["products"]:
        obj = Product.objects.create(
            item        = p["item"],
            sap_tn      = p.get("sap_tn") or "",
            sap_pl      = p.get("sap_pl"),
            dcc_type    = p.get("dcc_type") or "",
            description = p.get("description") or "",
            demand_2024 = p["demand_2024"],
            batch_size  = p["batch_size"],
            num_batches = p["num_batches"],
        )
        product_objs[p["item"]] = obj

    steps_created = 0
    for s in dataset["routing"]:
        product = product_objs.get(s["item"])
        machine = machine_objs.get(s["machine"])
        if product and machine:
            ProcessStep.objects.create(
                product            = product,
                step_number        = s["step"],
                machine            = machine,
                step_name          = s["name"],
                cycle_time_seconds = s["time"],
                workers_required   = s["workers"],
            )
            steps_created += 1

    return {
        "products_created":      len(product_objs),
        "machines_created":      len(machine_objs),
        "process_steps_created": steps_created,
        "metadata":              dataset.get("metadata", {}),
    }
