"""
api/utils_folder/synthetic_generator.py

Simplified synthetic dataset generator.

Machines  : M1, M2, ..., Mn
Products  : P1, P2, ..., Pn  (description field)
Steps     : Step 1, Step 2, ..., Step N

Public API
----------
SCENARIO_CATALOG          : dict    – named scenario profiles (backend source of truth)
get_scenario_catalog()    : dict    – serialisable copy of SCENARIO_CATALOG
generate_synthetic_dataset()        – low-level generator (full parameter control)
generate_scenario_dataset()         – named-scenario shortcut
insert_synthetic_dataset_into_db()  – persist a generated dataset into Django ORM
"""
from __future__ import annotations

import random
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
# Internal constants
# ---------------------------------------------------------------------------

_DEMAND_WEIGHTS = [0.2, 0.5, 0.3]
_DEMAND_RANGES  = {"low": (80, 400), "mid": (400, 1500), "high": (1500, 4500)}
_BATCH_SIZES    = [50, 100, 150, 200, 250]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scaled_demand_ranges(demand_range: tuple[int, int]) -> dict[str, tuple[int, int]]:
    """
    Split a (min, max) demand range into low / mid / high tier sub-ranges
    using the same tier-weight proportions as _DEMAND_WEIGHTS [0.2, 0.5, 0.3].
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


def _generate_machines(n: int) -> list[dict]:
    """
    Generate n machines with sequential names M1, M2, ..., Mn.

    Roles and available hours:
      Cutting  (30%) — 24 h/day  (automated)
      Assembly (40%) — 16 h/day  (operator-dependent)
      Molding  (30%) — 24 h/day  (automated)
    """
    machines: list[dict] = []

    n_molding  = max(1, round(n * 0.30))
    n_cutting  = max(1, round(n * 0.30))
    n_stations = max(1, n - n_molding - n_cutting)

    idx = 1

    for _ in range(n_cutting):
        machines.append({
            "name":                    f"M{idx}",
            "available_hours_per_day": 24.0,
            "_role":                   "cutting",
        })
        idx += 1

    for _ in range(n_stations):
        machines.append({
            "name":                    f"M{idx}",
            "available_hours_per_day": 16.0,
            "_role":                   "assembly",
        })
        idx += 1

    for _ in range(n_molding):
        machines.append({
            "name":                    f"M{idx}",
            "available_hours_per_day": 24.0,
            "_role":                   "molding",
        })
        idx += 1

    return machines


def _generate_products(
    n: int,
    rng: random.Random,
    demand_range: tuple[int, int] | None = None,
) -> list[dict]:
    """
    Generate n products with sequential names P1, P2, ..., Pn.

    Demand is drawn from a tiered distribution:
      Low  (20%) — bottom 20% of the demand range
      Mid  (50%) — middle 50% of the demand range
      High (30%) — top 30% of the demand range
    """
    effective_ranges = (
        _scaled_demand_ranges(demand_range) if demand_range is not None else _DEMAND_RANGES
    )

    products: list[dict] = []

    for i in range(1, n + 1):
        tier   = rng.choices(["low", "mid", "high"], weights=_DEMAND_WEIGHTS)[0]
        demand = rng.randint(*effective_ranges[tier])
        batch  = rng.choice(_BATCH_SIZES)

        products.append({
            "item":        i,
            "sap_tn":      f"TN-{rng.randint(100000, 999999)}",
            "sap_pl":      f"PL-{rng.randint(1000, 9999)}" if rng.random() > 0.25 else None,
            "dcc_type":    f"Type-{i}",
            "description": f"P{i}",
            "demand_2024": demand,
            "batch_size":  batch,
            "num_batches": max(1, round(demand / batch)),
        })

    return products


def _generate_process_steps(
    products: list,
    machines: list,
    avg_steps: int | None,
    rng: random.Random,
) -> list[dict]:
    """
    Assign process steps following factory flow.

    Step names: Step 1, Step 2, ..., Step N  (sequential within each product)

    Routing structure (same probabilities and cycle-time ranges as before):
      1. Cutting step        — always
      2. Cutting step        — always
      3. Assembly step       — optional (~65%)
      4. Assembly step       — always
      5. Assembly step       — optional (~30%)
      6. Extra assembly step — added if below target depth
      7. Test step           — optional (~50%)
      8. Molding step        — always last

    When avg_steps is None or 0, each product gets a random depth of 3–8 steps.
    """
    cutting_m  = [m["name"] for m in machines if m.get("_role") == "cutting"]
    assembly_m = [m["name"] for m in machines if m.get("_role") == "assembly"]
    molding_m  = [m["name"] for m in machines if m.get("_role") == "molding"]

    if not cutting_m:  cutting_m  = [machines[0]["name"]]
    if not assembly_m: assembly_m = [machines[min(1, len(machines) - 1)]["name"]]
    if not molding_m:  molding_m  = [machines[-1]["name"]]

    variable_depth = (avg_steps is None or avg_steps == 0)

    all_steps: list[dict] = []

    for product in products:
        item     = product["item"]
        steps:   list[dict] = []
        step_num = 1

        target_base = rng.randint(3, 8) if variable_depth else avg_steps

        def add(machine_pool, cycle_range, workers=0.5):
            nonlocal step_num
            steps.append({
                "product_item":       item,
                "step_number":        step_num,
                "machine_name":       rng.choice(machine_pool),
                "step_name":          f"Step {step_num}",
                "cycle_time_seconds": round(rng.uniform(*cycle_range), 2),
                "workers_required":   workers,
            })
            step_num += 1

        # 1. Cutting step (always)
        add(cutting_m,  (5.5, 9.5),   workers=0.5)

        # 2. Cutting step (always)
        add(cutting_m,  (5.0, 9.0),   workers=0.5)

        # 3. Assembly step (optional ~65%)
        if rng.random() < 0.65:
            add(assembly_m, (9.0, 20.0), workers=1.0)

        # 4. Assembly step (always)
        add(assembly_m, (7.0, 14.0), workers=1.0)

        # 5. Assembly step (optional ~30%)
        if rng.random() < 0.30:
            add(assembly_m, (7.5, 13.0), workers=0.5)

        # 6. Extra assembly step to reach target depth
        if len(steps) < target_base - 1:
            add(assembly_m, (8.0, 15.0), workers=1.0)

        # 7. Test step (50% chance)
        if rng.random() < 0.50:
            add(assembly_m, (4.0, 8.0),  workers=0.5)

        # 8. Molding step (always last)
        add(molding_m,  (13.0, 22.0), workers=1.0)

        all_steps.extend(steps)

    return all_steps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_scenario_catalog() -> list[dict]:
    """Return a JSON-serialisable list of scenario profile metadata."""
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
                                    None or 0 → random 3–8 per product.
    seed               : int|None   RNG seed; None → use current time (fresh each call)
    demand_range       : tuple|None (demand_min, demand_max) overrides default tier ranges.

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

    effective_steps = steps_per_product if (steps_per_product and steps_per_product > 0) else None

    machines  = _generate_machines(num_machines)
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

    steps_label = effective_steps if effective_steps is not None else "random 3–8"

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

    seed  = seed_override if seed_override is not None else p["seed"]
    steps = p["steps_per_product"]

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
