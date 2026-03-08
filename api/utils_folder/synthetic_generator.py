"""
api/utils_folder/synthetic_generator.py

Synthetic dataset generator for the Production Analytics Suite.

Generates realistic Products, Machines, ProcessSteps, and optionally
a full ProductionSchedule without requiring real CSV uploads.

All random values are seeded so experiments are fully reproducible:
    dataset = generate_synthetic_dataset(num_products=30, seed=42)

The generated schema is identical to what initialize_data() produces
from real CSVs, so every downstream algorithm (ERT scheduler, batch
ILP, buffer LP, gap analysis) works without modification.
"""

from __future__ import annotations

import math
import random
import string
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Domain constants — kept close to real-world plausible values
# ---------------------------------------------------------------------------

# Machines that appear in the real routing (subset used for synthetic data)
_MACHINE_POOL: list[str] = [
    "Cutting Automation",
    "SKM DCPC Crimp",
    "SKM Seal",
    "ARBURG 375ST",
    "ARBURG 520S",
    "PUR-Tube Assembly",
    "Sigma Press",
    "Leak Test Bench",
    "Vision Inspection",
    "Ultrasonic Welder",
    "Laser Marker",
    "Final Assembly",
    "Packaging Line",
    "Functional Tester",
    "Heat Shrink Station",
]

_DCC_TYPES: list[str] = ["Type-A", "Type-B", "Type-C", "Type-D"]

_STEP_NAMES: list[str] = [
    "Raw Material Prep",
    "Cutting",
    "Crimping",
    "Sealing",
    "Injection Moulding",
    "Sub-Assembly",
    "Ultrasonic Welding",
    "Leak Testing",
    "Vision Inspection",
    "Laser Marking",
    "Final Assembly",
    "Functional Testing",
    "Heat Shrinking",
    "Packaging",
    "Quality Gate",
]

# Cycle time ranges (seconds) per machine class
_CYCLE_TIME_RANGES: dict[str, tuple[float, float]] = {
    "Cutting Automation":  (30.0,  120.0),
    "SKM DCPC Crimp":      (15.0,   60.0),
    "SKM Seal":            (20.0,   80.0),
    "ARBURG 375ST":        (45.0,  180.0),
    "ARBURG 520S":         (60.0,  240.0),
    "PUR-Tube Assembly":   (90.0,  360.0),
    "Sigma Press":         (25.0,  100.0),
    "Leak Test Bench":     (10.0,   40.0),
    "Vision Inspection":   (5.0,    20.0),
    "Ultrasonic Welder":   (8.0,    30.0),
    "Laser Marker":        (3.0,    15.0),
    "Final Assembly":      (120.0, 480.0),
    "Packaging Line":      (15.0,   60.0),
    "Functional Tester":   (20.0,   90.0),
    "Heat Shrink Station": (10.0,   45.0),
}

# Default cycle time range for any machine not in the table above
_DEFAULT_CYCLE_RANGE: tuple[float, float] = (20.0, 120.0)


# ---------------------------------------------------------------------------
# Data containers (plain dicts returned; dataclasses used internally)
# ---------------------------------------------------------------------------

@dataclass
class _SyntheticMachine:
    name: str
    available_hours_per_day: float = 24.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available_hours_per_day": self.available_hours_per_day,
        }


@dataclass
class _SyntheticProduct:
    item: int
    sap_tn: str
    sap_pl: str
    dcc_type: str
    description: str
    demand_2024: int
    batch_size: int
    num_batches: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "item":        self.item,
            "sap_tn":      self.sap_tn,
            "sap_pl":      self.sap_pl,
            "dcc_type":    self.dcc_type,
            "description": self.description,
            "demand_2024": self.demand_2024,
            "batch_size":  self.batch_size,
            "num_batches": self.num_batches,
        }


@dataclass
class _SyntheticStep:
    item: int           # FK → product item number
    step: int           # 1-based step number
    machine: str        # machine name
    time: float         # cycle_time_seconds
    name: str           # step display name
    workers: float      # workers_required

    def to_dict(self) -> dict[str, Any]:
        return {
            "item":    self.item,
            "step":    self.step,
            "machine": self.machine,
            "time":    round(self.time, 2),
            "name":    self.name,
            "workers": self.workers,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    num_products: int = 10,
    num_machines: int | None = None,
    steps_per_product: int | None = None,
    seed: int = 42,
    demand_range: tuple[int, int] = (500, 50_000),
) -> dict[str, Any]:
    """
    Generate a complete synthetic production dataset.

    Parameters
    ----------
    num_products       : int   number of synthetic products (default 10)
    num_machines       : int   machines to include (default min(num_products+3, 15))
    steps_per_product  : int   routing depth per product (default random 3–8)
    seed               : int   random seed for reproducibility (default 42)
    demand_range       : tuple (min_demand, max_demand) for 2024 annual demand

    Returns
    -------
    dict with keys:
        "machines"   : list[dict]  — machine records (name, available_hours_per_day)
        "products"   : list[dict]  — product records (item, sap_tn, …, num_batches)
        "routing"    : list[dict]  — process step records (item, step, machine, …)
        "metadata"   : dict        — generation parameters (for logging / thesis)

    The structure mirrors the output of process_frontpage_data() +
    process_routing_data() so initialize_data() can be called with it
    directly instead of with uploaded CSV files.
    """
    # ------------------------------------------------------------------
    # STEP 1: Seed all random generators for reproducibility
    # ------------------------------------------------------------------
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # STEP 2: Choose machine pool
    # ------------------------------------------------------------------
    if num_machines is None:
        num_machines = min(num_products + 3, len(_MACHINE_POOL))
    num_machines = max(2, min(num_machines, len(_MACHINE_POOL)))

    selected_machines = rng.sample(_MACHINE_POOL, num_machines)
    machines = [
        _SyntheticMachine(name=m, available_hours_per_day=24.0).to_dict()
        for m in selected_machines
    ]

    # ------------------------------------------------------------------
    # STEP 3: Generate products
    # ------------------------------------------------------------------
    products: list[dict] = []
    routing:  list[dict] = []

    for idx in range(num_products):
        item = 1000 + idx * 10   # e.g., 1000, 1010, 1020 …

        # Unique SAP identifiers
        sap_tn = rng.randint(700_000, 799_999)
        sap_pl = rng.randint(200_000, 299_999)

        dcc_type    = rng.choice(_DCC_TYPES)
        description = f"Synthetic Product {item} ({dcc_type})"

        # Demand: log-uniform so we get a realistic wide spread
        log_min = math.log(demand_range[0])
        log_max = math.log(demand_range[1])
        demand  = int(math.exp(np_rng.uniform(log_min, log_max)))

        # Initial batch split (same heuristic as real initialize_data)
        batch_size  = max(1, math.ceil(demand / 12))
        num_batches = 12

        products.append(
            _SyntheticProduct(
                item        = item,
                sap_tn      = str(sap_tn),
                sap_pl      = str(sap_pl),
                dcc_type    = dcc_type,
                description = description,
                demand_2024 = demand,
                batch_size  = batch_size,
                num_batches = num_batches,
            ).to_dict()
        )

        # --------------------------------------------------------------
        # STEP 4: Generate routing steps for this product
        # Each product visits a random ordered subset of machines.
        # Steps are ordered so each machine appears at most once per
        # product (no re-visit), mirroring real routing data.
        # --------------------------------------------------------------
        if steps_per_product is None:
            n_steps = rng.randint(3, min(8, num_machines))
        else:
            n_steps = max(1, min(steps_per_product, num_machines))

        step_machines = rng.sample(selected_machines, n_steps)
        step_names    = rng.sample(_STEP_NAMES, n_steps)

        for step_no, (mname, sname) in enumerate(zip(step_machines, step_names), start=1):
            lo, hi = _CYCLE_TIME_RANGES.get(mname, _DEFAULT_CYCLE_RANGE)
            cycle_time = round(rng.uniform(lo, hi), 2)

            routing.append(
                _SyntheticStep(
                    item    = item,
                    step    = step_no,
                    machine = mname,
                    time    = cycle_time,
                    name    = sname,
                    workers = 0.5,
                ).to_dict()
            )

    # ------------------------------------------------------------------
    # STEP 5: Return complete dataset + metadata
    # ------------------------------------------------------------------
    metadata = {
        "dataset_type":     "synthetic",
        "num_products":     num_products,
        "num_machines":     num_machines,
        "steps_per_product": steps_per_product or "random(3-8)",
        "seed":             seed,
        "demand_range":     demand_range,
        "total_routing_steps": len(routing),
    }

    return {
        "machines": machines,
        "products": products,
        "routing":  routing,
        "metadata": metadata,
    }


def insert_synthetic_dataset_into_db(dataset: dict, clear_existing: bool = True) -> dict:
    """
    Persist a synthetic dataset (from generate_synthetic_dataset) into the
    Django ORM models, mirroring what initialize_data() does for real CSVs.

    Must be called from within a Django application context (models available).

    Parameters
    ----------
    dataset        : dict  output of generate_synthetic_dataset()
    clear_existing : bool  wipe existing data before inserting (default True)

    Returns
    -------
    dict with counts: products_created, machines_created, process_steps_created
    """
    # Late import to avoid circular imports at module load time
    from api.models import Product, Machine, ProcessStep, ProductionSchedule  # noqa

    if clear_existing:
        ProductionSchedule.objects.all().delete()
        ProcessStep.objects.all().delete()
        Product.objects.all().delete()
        Machine.objects.all().delete()

    # ------------------------------------------------------------------
    # STEP 1: Create Machine records
    # ------------------------------------------------------------------
    machine_objs: dict[str, Machine] = {}
    for m in dataset["machines"]:
        obj = Machine.objects.create(
            name=m["name"],
            available_hours_per_day=m["available_hours_per_day"],
        )
        machine_objs[m["name"]] = obj

    # ------------------------------------------------------------------
    # STEP 2: Create Product records
    # ------------------------------------------------------------------
    product_objs: dict[int, Product] = {}
    for p in dataset["products"]:
        obj = Product.objects.create(
            item        = p["item"],
            sap_tn      = p["sap_tn"],
            sap_pl      = p["sap_pl"],
            dcc_type    = p["dcc_type"],
            description = p["description"],
            demand_2024 = p["demand_2024"],
            batch_size  = p["batch_size"],
            num_batches = p["num_batches"],
        )
        product_objs[p["item"]] = obj

    # ------------------------------------------------------------------
    # STEP 3: Create ProcessStep records
    # ------------------------------------------------------------------
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