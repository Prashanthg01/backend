# api/utils.py

import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from collections import defaultdict
from django.db.models import Sum
from .models import Product, ProcessStep, ProductionSchedule

from pulp import (
    LpProblem, LpMinimize, LpMaximize, LpVariable, LpInteger, LpContinuous, LpBinary,
    lpSum, LpStatus, value, PULP_CBC_CMD
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHIFT_LABELS = [
    'Shift 1', 'Shift 1 B', 'Shift 2', 'Shift 2 A', 'Shift 3', 'Shift 3 C',
    'Shift 4', 'Shift 4 B', 'Shift 5', 'Shift 5 A', 'Shift 6', 'Shift 6 C',
    'Shift 7', 'Shift 7 B', 'Shift 8', 'Shift 8 A', 'Shift 9', 'Shift 9 C',
    'Shift 10', 'Shift 10 B', 'Shift 11', 'Shift 11 A', 'Shift 12', 'Shift 12 C',
    'Shift 13', 'Shift 13 B', 'Shift 14', 'Shift 14 A', 'Shift 15', 'Shift 15 C',
    'Shift 16', 'Shift 16 B', 'Shift 17', 'Shift 17 A', 'Shift 18', 'Shift 18 C'
]

# Solver shared across the module — silent output
_SOLVER = PULP_CBC_CMD(msg=0)
_SOLVER_TIMED = PULP_CBC_CMD(msg=0, timeLimit=45)   # 45-second wall-clock cap per solve


# ===========================================================================
# 0.  LEFT-SHIFT COMPACTION  (gap elimination)                         ← NEW
# ===========================================================================
# After a greedy dispatch, every operation is slid as early as possible
# while respecting:
#   a) Machine non-overlap  – no two ops on the same machine overlap
#   b) Job precedence       – step N+1 can only start after step N ends
#
# WHY GAPS APPEAR IN THE GREEDY SCHEDULE
# ----------------------------------------
# 1. Precedence-induced starvation: a machine is free but the next operation
#    waiting for it can't start because its upstream step (on a DIFFERENT
#    machine) hasn't finished yet. The greedy dispatcher moves on, leaving
#    a hole.
# 2. Batch-interleaving gaps: different products share a machine; when
#    product A's batch finishes, product B's next batch isn't ready yet.
# 3. SPT ordering mismatches: sorting globally by duration causes batches
#    to be processed out of natural order, leaving holes while upstream
#    ops complete.
#
# THE FIX
# --------
# Iterate over all ops (sorted by start time) and for each op compute:
#   earliest_start = max(machine_free_time, predecessor_end_time)
# If earliest_start < op.current_start, slide the op left.
# Repeat until no op moves (converges in 2–3 passes, O(n²) total).

def left_shift_compaction(schedule: list, max_passes: int = 5) -> list:
    """
    Eliminate idle gaps in a greedy-dispatched schedule by sliding every
    operation as early as possible.

    Parameters
    ----------
    schedule : list[dict]
        Each dict is one scheduled operation and MUST have keys:
            'machine_name' (str)
            'batch_id'     (str)   – job identifier
            'step_number'  (int)   – 1-based step within the job
            'start_hrs'    (float) – hours from schedule epoch
            'end_hrs'      (float)
            'dur_hours'    (float)
            'start_dt'     (datetime)
            'end_dt'       (datetime)
    start_dt_epoch : not needed – we use start_hrs/end_hrs internally.

    Returns
    -------
    Same list with start_hrs / end_hrs / start_dt / end_dt mutated in-place.
    """
    if not schedule:
        return schedule

    # Determine epoch from the minimum start across all ops
    epoch_dt = min(op['start_dt'] for op in schedule)

    for pass_no in range(max_passes):
        moved = 0

        # Sort by current start time so earlier ops anchor first
        schedule.sort(key=lambda o: (o['start_hrs'], o['machine_name']))

        # Rebuild machine-timeline and job-end maps incrementally
        # machine_timeline[m] = sorted list of (start_hrs, end_hrs)
        machine_timeline: dict = {}
        # job_end[batch_id][step_number] = end_hrs
        job_end: dict = {}

        for op in schedule:
            m     = op['machine_name']
            job   = op['batch_id']
            step  = op['step_number']
            dur   = op['dur_hours']

            # Lower bound from job precedence (predecessor step must be done)
            pred_end = _predecessor_end_hrs(job_end, job, step)

            # Earliest gap on this machine that fits `dur` starting from pred_end
            earliest = _first_fit_gap_hrs(
                machine_timeline.get(m, []),
                pred_end,
                dur,
            )

            # Only move if we can genuinely shift earlier (> 1-second tolerance)
            if earliest < op['start_hrs'] - (1 / 3600):
                op['start_hrs'] = earliest
                op['end_hrs']   = earliest + dur
                op['start_dt']  = epoch_dt + timedelta(hours=earliest)
                op['end_dt']    = epoch_dt + timedelta(hours=earliest + dur)
                moved += 1

            # Register op in machine timeline
            if m not in machine_timeline:
                machine_timeline[m] = []
            machine_timeline[m].append((op['start_hrs'], op['end_hrs']))
            machine_timeline[m].sort()

            # Register job step completion
            if job not in job_end:
                job_end[job] = {}
            job_end[job][step] = op['end_hrs']

        if moved == 0:
            break   # converged

    return schedule


def _predecessor_end_hrs(job_end: dict, job: str, step: int) -> float:
    """Return end_hrs of step-1 for this job, or 0.0 if step == 1."""
    if step <= 1:
        return 0.0
    return job_end.get(job, {}).get(step - 1, 0.0)


def _first_fit_gap_hrs(
    timeline: list,          # sorted list of (start_hrs, end_hrs)
    earliest: float,
    duration: float,
) -> float:
    """
    Find the earliest start >= `earliest` where `duration` fits without
    overlapping any interval in `timeline`.
    """
    candidate = earliest
    for busy_start, busy_end in timeline:
        if busy_end <= candidate:
            continue                          # busy block entirely before us
        if candidate + duration <= busy_start:
            break                             # fits in gap before this block
        candidate = max(candidate, busy_end)  # pushed past this block
    return candidate


def count_schedule_gaps(schedule: list, min_gap_sec: int = 60) -> dict:
    """
    Count idle gaps remaining in a schedule after compaction.
    Useful for logging / returning in API responses.

    Returns
    -------
    dict with keys: total_gaps, total_idle_hours, per_machine
    """
    machine_ops: dict = defaultdict(list)
    for op in schedule:
        machine_ops[op['machine_name']].append(op)

    total_gaps = 0
    total_idle = 0.0
    per_machine = {}

    for m, mops in machine_ops.items():
        sorted_ops = sorted(mops, key=lambda o: o['start_hrs'])
        gaps  = 0
        idle  = 0.0
        for i in range(1, len(sorted_ops)):
            gap_h = sorted_ops[i]['start_hrs'] - sorted_ops[i-1]['end_hrs']
            if gap_h * 3600 > min_gap_sec:
                gaps  += 1
                idle  += gap_h
        total_gaps += gaps
        total_idle += idle
        per_machine[m] = {'gaps': gaps, 'idle_hours': round(idle, 3)}

    return {
        'total_gaps':       total_gaps,
        'total_idle_hours': round(total_idle, 2),
        'per_machine':      per_machine,
    }


# ===========================================================================
# 1.  BATCH-SIZE OPTIMISATION  (single-product ILP)
# ===========================================================================

def calculate_optimal_batch_size(
        demand: int,
        max_num_batches: int = 25,
        min_batch_size: int = 50,
        max_batch_size: int = 500,
) -> tuple[int, int, float]:
    """
    Solve a single-product batch-size ILP with PuLP.

    Returns
    -------
    (batch_size, num_batches, ideal_batch_size)

    Fix vs. original
    ----------------
    The original code used the raw user min/max as ILP bounds.
    For small demands (e.g., 121) this forced batch_size=500 with N=1,
    which is wrong (one batch of 500 when demand is only 121 wastes
    everything downstream).
    For large demands (e.g., 319,908) the ILP had NO feasible solution
    because ceil(D/n) > 500 for every n, so it always fell back to the
    naive heuristic.

    The fix: adaptively clamp the bounds to the demand-derived feasible
    range, keeping user preferences when they are achievable.
    """
    if demand <= 0:
        return 0, 0, 0.0

    ideal_batch_size = demand / max_num_batches

    # ── Adaptive bounds (THE FIX) ────────────────────────────────────────────
    eff_min, eff_max = _adaptive_bounds(
        demand, max_num_batches, min_batch_size, max_batch_size
    )

    candidates = list(range(1, max_num_batches + 1))

    prob = LpProblem("BatchSizeOptimization", LpMinimize)

    B = LpVariable("batch_size",  lowBound=eff_min, upBound=eff_max,  cat=LpInteger)
    N = LpVariable("num_batches", lowBound=1,        upBound=max_num_batches, cat=LpInteger)

    # Minimise number of batches (fewer set-ups)
    prob += N

    # Binary selector: exactly one candidate n is chosen
    y = {n: LpVariable(f"y_{n}", cat="Binary") for n in candidates}

    prob += lpSum(y[n] for n in candidates) == 1
    prob += N == lpSum(n * y[n] for n in candidates)

    M = eff_max  # big-M for linking constraint
    for n in candidates:
        required_B = math.ceil(demand / n)
        prob += B >= required_B - M * (1 - y[n])

    # Prune candidates whose required batch exceeds effective max
    for n in candidates:
        if math.ceil(demand / n) > eff_max:
            prob += y[n] == 0

    prob.solve(_SOLVER)

    if LpStatus[prob.status] == "Optimal":
        opt_B = max(eff_min, min(eff_max, int(round(value(B)))))
        opt_N = max(1, min(max_num_batches, int(round(value(N)))))
        return opt_B, opt_N, round(ideal_batch_size, 2)

    # ── Fallback: demand-aware heuristic ─────────────────────────────────────
    fallback_N = max(1, min(max_num_batches, math.ceil(demand / eff_min)))
    fallback_B = math.ceil(demand / fallback_N)
    fallback_B = max(eff_min, min(eff_max, fallback_B))
    return fallback_B, fallback_N, round(ideal_batch_size, 2)

# ===========================================================================
# 2.  JOINT MULTI-PRODUCT BATCH OPTIMISATION
# ===========================================================================

def optimize_product_batches_jointly(
        products,                    # QuerySet / list of Product ORM objects
        max_num_batches: int = 25,
        min_batch_size: int = 50,
        max_batch_size: int = 500,
        time_limit_seconds: int = 120,
) -> dict:
    """
    Joint multi-product batch optimization that balances machine loads.

    Instead of optimising each product independently (which can create
    bottleneck machines), this single MILP decides batch sizes for ALL
    products simultaneously, minimising the *makespan proxy* C — the
    maximum total processing hours on any single machine.

    Decision variables
    ------------------
    y_{p,n}   : Binary — product p uses n batches  (one per product)
    C         : Continuous — worst-case machine load (hours)

    Objective
    ---------
    Minimise C  (balance machine loads)

    Constraints
    -----------
    Per product p:
      Σ_n y_{p,n} = 1                          (exactly one n chosen)
      min_batch ≤ ceil(demand_p / n) ≤ max_batch  (feasibility filter)

    Per machine m:
      Σ_{p,n}  load_{p,m,n} * y_{p,n}  ≤  C   (load ≤ makespan proxy)

    where  load_{p,m,n} = cycle_time_{p,m} × ceil(demand_p/n) × n / 3600
                        = cycle_time × total_units / 3600  (hours)

    Returns
    -------
    dict with keys:
      status          : 'optimal' | 'fallback' | 'error'
      results         : list of {product_id, batch_size, num_batches,
                                 demand, ideal_batch_size, improvement}
      makespan_proxy  : C value (hours)
      machine_loads   : {machine_name: hours} after optimization
      products_updated: count of products whose DB records were saved
      message         : human-readable summary
    """
    from .models import ProcessStep  # avoid circular import at module level

    products_list = list(products)

    if not products_list:
        return {
            "status": "error",
            "message": "No products provided.",
            "results": [],
            "makespan_proxy": 0,
            "machine_loads": {},
            "products_updated": 0,
        }

    # ── Collect process routing data ──────────────────────────────────────────
    # cycle_times[product_pk][machine_name] = cycle_time_seconds (float)
    cycle_times: dict[int, dict[str, float]] = {}
    all_machine_names: set[str] = set()

    for product in products_list:
        steps = ProcessStep.objects.filter(product=product).select_related("machine")
        cycle_times[product.pk] = {}
        for step in steps:
            mname = step.machine.name
            # If a product visits the same machine multiple times, sum the times
            cycle_times[product.pk][mname] = (
                cycle_times[product.pk].get(mname, 0.0) + step.cycle_time_seconds
            )
            all_machine_names.add(mname)

    all_machines = sorted(all_machine_names)

    # ── Build the ILP ─────────────────────────────────────────────────────────
    prob = LpProblem("JointBatchOptimization", LpMinimize)

    # Makespan proxy: the worst machine load (hours)
    C = LpVariable("makespan_proxy", lowBound=0, cat=LpContinuous)
    prob += C  # objective: minimise C

    candidates = list(range(1, max_num_batches + 1))

    # Per-product binary selectors
    # Y[pk][n] = 1  iff product pk uses n batches
    Y: dict[int, dict[int, LpVariable]] = {}

    # Pre-compute effective bounds and feasible load per (product, n, machine)
    # load_table[pk][n][machine] = hours on that machine if n batches chosen
    load_table: dict[int, dict[int, dict[str, float]]] = {}

    for product in products_list:
        pk     = product.pk
        demand = int(product.demand_2024) if product.demand_2024 else 0

        if demand <= 0:
            # Skip zero-demand products in the joint model
            continue

        eff_min, eff_max = _adaptive_bounds(
            demand, max_num_batches, min_batch_size, max_batch_size
        )

        Y[pk] = {}
        load_table[pk] = {}

        feasible_ns = []
        for n in candidates:
            batch_sz = math.ceil(demand / n)
            if eff_min <= batch_sz <= eff_max:
                feasible_ns.append(n)

        if not feasible_ns:
            # If no n is feasible with user bounds, open the range fully
            eff_min, eff_max = math.ceil(demand / max_num_batches), demand
            feasible_ns = candidates[:]

        for n in candidates:
            Y[pk][n] = LpVariable(f"y_{pk}_{n}", cat="Binary")

            # Compute load on each machine for this (product, n) choice
            batch_sz    = math.ceil(demand / n)
            total_units = batch_sz * n  # total units produced (≥ demand)
            load_table[pk][n] = {}

            for mname in all_machines:
                ct = cycle_times[pk].get(mname, 0.0)  # seconds per unit
                hours = (ct * total_units) / 3600.0
                load_table[pk][n][mname] = hours

        # Constraint: exactly one n chosen
        prob += lpSum(Y[pk][n] for n in candidates) == 1

        # Constraint: prune infeasible n values
        for n in candidates:
            batch_sz = math.ceil(demand / n)
            if not (eff_min <= batch_sz <= eff_max):
                prob += Y[pk][n] == 0

    # ── Machine load constraints ──────────────────────────────────────────────
    for mname in all_machines:
        machine_load_expr = []
        for product in products_list:
            pk     = product.pk
            demand = int(product.demand_2024) if product.demand_2024 else 0
            if demand <= 0 or pk not in Y:
                continue
            for n in candidates:
                hours = load_table[pk][n].get(mname, 0.0)
                if hours > 0:
                    machine_load_expr.append(hours * Y[pk][n])

        if machine_load_expr:
            prob += lpSum(machine_load_expr) <= C

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = PULP_CBC_CMD(msg=0, timeLimit=time_limit_seconds)
    prob.solve(solver)

    status = LpStatus[prob.status]
    solved_optimally = status == "Optimal"

    # ── Extract results ───────────────────────────────────────────────────────
    results       = []
    machine_loads = {m: 0.0 for m in all_machines}
    products_updated = 0

    for product in products_list:
        pk     = product.pk
        demand = int(product.demand_2024) if product.demand_2024 else 0

        # Old baseline (as in DB initialization)
        old_batch_size  = math.ceil(demand / 12) if demand > 0 else 1
        old_num_batches = 12

        ideal_batch = round(demand / max_num_batches, 2) if demand > 0 else 0

        if demand <= 0 or pk not in Y:
            results.append({
                "product_id":      pk,
                "item":            product.item,
                "description":     (product.description or "")[:50],
                "demand":          demand,
                "old_batch_size":  old_batch_size,
                "old_num_batches": old_num_batches,
                "new_batch_size":  old_batch_size,
                "new_num_batches": old_num_batches,
                "ideal_batch_size": ideal_batch,
                "improvement":     "0%",
                "source":          "skipped",
            })
            continue

        if solved_optimally:
            # Find which n was selected
            chosen_n = None
            for n in candidates:
                if Y[pk][n].varValue is not None and Y[pk][n].varValue > 0.5:
                    chosen_n = n
                    break

            if chosen_n is not None:
                new_batch = math.ceil(demand / chosen_n)
                new_n     = chosen_n
            else:
                # Safety fallback
                eff_min, eff_max = _adaptive_bounds(
                    demand, max_num_batches, min_batch_size, max_batch_size
                )
                new_n     = max(1, min(max_num_batches, math.ceil(demand / eff_min)))
                new_batch = math.ceil(demand / new_n)
        else:
            # Full fallback when solver didn't find optimum
            eff_min, eff_max = _adaptive_bounds(
                demand, max_num_batches, min_batch_size, max_batch_size
            )
            new_n     = max(1, min(max_num_batches, math.ceil(demand / eff_min)))
            new_batch = math.ceil(demand / new_n)
            new_batch = max(eff_min, min(eff_max, new_batch))

        # Accumulate actual machine loads after decision
        for mname in all_machines:
            if chosen_n if solved_optimally else True:
                n_used = chosen_n if solved_optimally and chosen_n else new_n
                ct     = cycle_times[pk].get(mname, 0.0)
                total  = new_batch * new_n
                machine_loads[mname] += (ct * total) / 3600.0

        # Improvement metric
        if old_num_batches != new_n:
            improvement_pct = (old_num_batches - new_n) / old_num_batches * 100
            improvement     = f"{improvement_pct:.1f}%"
        else:
            improvement = "0%"

        results.append({
            "product_id":      pk,
            "item":            product.item,
            "description":     (product.description or "")[:50],
            "demand":          demand,
            "old_batch_size":  old_batch_size,
            "old_num_batches": old_num_batches,
            "new_batch_size":  new_batch,
            "new_num_batches": new_n,
            "ideal_batch_size": ideal_batch,
            "improvement":     improvement,
            "source":          "joint_ilp" if solved_optimally else "fallback",
        })

        # Save to DB
        if product.batch_size != new_batch or product.num_batches != new_n:
            product.batch_size  = new_batch
            product.num_batches = new_n
            product.save(update_fields=["batch_size", "num_batches"])
            products_updated += 1

    makespan_proxy = round(value(C), 4) if solved_optimally and value(C) is not None else 0.0

    return {
        "status":           "optimal" if solved_optimally else "fallback",
        "message":          (
            f"Joint optimization {'succeeded' if solved_optimally else 'used fallback'}. "
            f"{products_updated} products updated. "
            f"Makespan proxy: {makespan_proxy:.1f}h"
        ),
        "results":          results,
        "makespan_proxy":   makespan_proxy,
        "machine_loads":    {m: round(v, 2) for m, v in machine_loads.items()},
        "products_updated": products_updated,
    }


# ===========================================================================
# 3.  BUFFER ALLOCATION OPTIMISATION
# ===========================================================================

def pulp_optimize_buffers(machine_buffer_data, total_budget):
    """Allocate a finite buffer budget across machines via PuLP LP."""
    if not machine_buffer_data or total_budget <= 0:
        for m in machine_buffer_data:
            m['allocated_buffer'] = 0.0
            m['shortfall']        = m.get('required_buffer', 0)
        return machine_buffer_data

    machines = [m['machine'] for m in machine_buffer_data]

    prob  = LpProblem("BufferAllocation", LpMinimize)
    alloc = {m: LpVariable(f"alloc_{m}", lowBound=0, cat=LpContinuous) for m in machines}
    slack = {m: LpVariable(f"slack_{m}", lowBound=0, cat=LpContinuous) for m in machines}

    req  = {d['machine']: d['required_buffer'] for d in machine_buffer_data}
    util = {d['machine']: d['utilization']     for d in machine_buffer_data}

    prob += lpSum(util[m] * slack[m] for m in machines)

    for m in machines:
        prob += alloc[m] <= req[m]
        prob += slack[m] == req[m] - alloc[m]

    prob += lpSum(alloc[m] for m in machines) <= total_budget

    prob.solve(_SOLVER)

    for d in machine_buffer_data:
        m = d['machine']
        if LpStatus[prob.status] == "Optimal":
            d['allocated_buffer'] = round(value(alloc[m]), 2)
            d['shortfall']        = round(value(slack[m]),  2)
        else:
            share                 = req[m] / max(sum(req.values()), 1e-9)
            d['allocated_buffer'] = round(share * total_budget, 2)
            d['shortfall']        = round(max(req[m] - d['allocated_buffer'], 0), 2)

    return machine_buffer_data


# ===========================================================================
# 4.  JOB-SHOP SCHEDULER  (Greedy SPT → Left-Shift Compaction → PuLP MILP)
# ===========================================================================

def run_job_shop_scheduler(
    products,
    start_dt: datetime,
    objective: str = "makespan",
    batch_override: dict = None,
    local_opt_machines: int = 5,
    enable_compaction: bool = True,        # ← NEW: gap elimination flag
    progress_callback=None,
) -> list:
    """
    Build a job-shop schedule for *products* starting at *start_dt*.

    Phase 1a  –  Greedy SPT dispatcher      (always runs)
    Phase 1b  –  Left-shift compaction      (runs when enable_compaction=True)
    Phase 2   –  PuLP local MILP            (runs when local_opt_machines > 0)

    Returns
    -------
    list[dict]  – one dict per scheduled operation.
    """
    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    _progress(5, "Extracting process routing…")

    # ── 1. Build routing data ─────────────────────────────────────────
    routing = {}
    for p in products:
        steps = (
            ProcessStep.objects
            .filter(product=p, cycle_time_seconds__gt=0)
            .select_related('machine')
            .order_by('step_number')
        )
        if steps.exists():
            routing[p.pk] = [
                {
                    'step':       s.step_number,
                    'machine':    s.machine.name,
                    'cycle_sec':  s.cycle_time_seconds,
                    'step_name':  s.step_name,
                }
                for s in steps
            ]

    _progress(15, "Building job list…")

    # ── 2. LPT product ordering + job expansion ──────────────────────────
    # Sort products by total processing hours DESCENDING (Longest Processing
    # Time first).  This ensures the heaviest product's upstream steps are
    # dispatched first, so its downstream machines (e.g. SKM Seal, ARBURG)
    # unblock early and lighter products can fill them without waiting.
    #
    # Without LPT ordering:
    #   P1 (large) locks SKM DCPC Crimp for a long stretch → P2 arrives at
    #   SKM Seal late → the visible gap on the right of the SKM Seal row.
    #   On ARBURG 6-10: P3 fills early slots, then machine idles waiting for
    #   P2's upstream steps that are still blocked behind P1.
    # With LPT ordering:
    #   Heaviest product clears all upstream steps first; lighter products
    #   find those machines free and interleave into the gaps → gaps close.
    def _total_work_hrs(p):
        if p.pk not in routing:
            return 0.0
        bs = p.batch_size  if p.batch_size  > 0 else 1
        nb = p.num_batches if p.num_batches > 0 else 1
        if batch_override and p.pk in batch_override:
            bs, nb = batch_override[p.pk]
        return sum(s['cycle_sec'] for s in routing[p.pk]) * bs * nb / 3600.0

    products_ordered = sorted(products, key=_total_work_hrs, reverse=True)

    jobs    = []
    job_ops = []

    for p in products_ordered:
        if p.pk not in routing:
            continue

        if batch_override and p.pk in batch_override:
            b_size, n_batches = batch_override[p.pk]
        else:
            b_size    = p.batch_size  if p.batch_size  > 0 else 1
            n_batches = p.num_batches if p.num_batches > 0 else 1

        steps = routing[p.pk]

        for b in range(1, n_batches + 1):
            job_id = f"{p.item}_B{b:03d}"
            jobs.append({
                'job_id':       job_id,
                'product_pk':   p.pk,
                'product_item': p.item,
                'batch_num':    b,
                'batch_size':   b_size,
                'steps':        steps,
            })

            for k, s in enumerate(steps):
                job_ops.append({
                    'job_id':     job_id,
                    'product_pk': p.pk,
                    'batch_num':  b,
                    'batch_size': b_size,
                    'step_idx':   k,
                    'step':       s['step'],
                    'machine':    s['machine'],
                    'dur_hours':  (s['cycle_sec'] * b_size) / 3600.0,
                    'step_name':  s['step_name'],
                })

    n_machines = len(set(o['machine'] for o in job_ops))
    _progress(25, f"Scheduling {len(jobs)} jobs across {n_machines} machines…")

    # ── 3. Phase 1a: Greedy SPT dispatcher ────────────────────────────
    schedule = _greedy_dispatch(jobs, job_ops, start_dt, progress_callback)
    _progress(55, f"Greedy dispatch complete: {len(schedule):,} operations")

    # ── 4. Phase 1b: Left-shift compaction (gap elimination) ──────────
    if enable_compaction and schedule:
        _progress(60, "Eliminating idle gaps (left-shift compaction)…")

        gaps_before = count_schedule_gaps(schedule)
        schedule    = left_shift_compaction(schedule, max_passes=5)
        gaps_after  = count_schedule_gaps(schedule)

        _progress(70, (
            f"Gap elimination done — "
            f"{gaps_before['total_gaps']} → {gaps_after['total_gaps']} gaps, "
            f"{gaps_before['total_idle_hours']:.1f}h → "
            f"{gaps_after['total_idle_hours']:.1f}h idle"
        ))

    # ── 5. Phase 2: PuLP local optimiser on bottleneck machines ───────
    if local_opt_machines > 0:
        _progress(75, f"PuLP optimisation on {local_opt_machines} bottleneck machines…")
        schedule = _local_pulp_optimise(schedule, job_ops, start_dt, local_opt_machines)

    _progress(95, "Finalising schedule…")
    return schedule


# ---------------------------------------------------------------------------
# Phase 1a: Event-Driven Earliest-Release-Time (ERT) dispatcher
# ---------------------------------------------------------------------------
# WHY THE OLD SPT ROUND-BY-ROUND APPROACH LEFT GAPS
# --------------------------------------------------
# The old dispatcher processed ops step-by-step (all step-1 ops, then all
# step-2 ops …) sorted by duration within each round.  This caused two gap
# patterns visible in the Gantt:
#
#   1. ARBURG 375ST gap  — P3 batch fills ARBURG at step 1, then ARBURG
#      sits idle because P1's step 1 (on Sigma/PUR-Tube) hasn't finished
#      yet when the "step-2 round" starts.  The SPT sort picks short jobs
#      that aren't ready on ARBURG, pushing longer-but-earlier-available
#      jobs to later slots.
#
#   2. SKM Seal gap  — P2 batches depend on SKM DCPC Crimp finishing.
#      The round-by-round approach schedules P1 and P3 on SKM Seal first
#      (they win SPT in rounds 2–4), then P2 arrives late — even though
#      P2 could have been interleaved much earlier if we looked globally.
#
# THE FIX: GLOBAL EVENT-DRIVEN ERT DISPATCHER
# --------------------------------------------
# At every "event" (= a machine becomes free OR a job step completes):
#   1. Find all operations whose job is ready (predecessor step done)
#   2. Among those, assign each to its machine at earliest_start =
#      max(machine_free[machine], job_ready[job])
#   3. Pick the one with the LOWEST earliest_start (Earliest Release Time)
#      and schedule it — this is the operation that would actually start
#      soonest, filling the machine with no idle wait.
#   4. Update machine_free and job_ready, repeat until all ops scheduled.
#
# This is O(n log n) — faster than the old approach for large instances
# and produces schedules with significantly fewer cross-product gaps.

def _greedy_dispatch(jobs, job_ops, start_dt, progress_callback=None):
    """
    Event-driven ERT dispatcher with interleaved batch tie-breaking.

    WHY TIE-BREAKING MATTERS
    ------------------------
    At t=0 every job's first step has release time 0.0.  Python's heapq
    breaks ties on the second heap element, which was previously the job_id
    string (e.g. "1234_B001").  This made the product with the lowest item
    number win EVERY tie, filling ALL shared machines before other products
    even get a look-in — causing the large SKM Seal gap where P2 (red)
    arrives much later than P1/P3 because every upstream machine was
    monopolised by P1 first.

    THE FIX: ROUND-ROBIN TIE-BREAK
    --------------------------------
    We assign each job an interleave_rank = (batch_number - 1) * n_products
                                           + product_rank
    so the dispatch order at equal start times becomes:
      P1b1, P2b1, P3b1, P1b2, P2b2, P3b2, P1b3, ...

    This interleaves batches across products on every shared machine,
    so P2 and P3 start their upstream steps (PUR-Tube, SKM DCPC) after
    just ONE P1 batch instead of ALL P1 batches — dramatically reducing
    the release-time gap at SKM Seal.

    ARBURG 6-10 initial gap remains: P3's process routing has upstream
    steps (Sigma, Connector Assembly) before ARBURG, so ARBURG must wait
    for those to complete.  This is an irreducible critical-path constraint
    imposed by the product's own routing — no dispatcher can eliminate it.
    """
    import heapq

    machine_free: dict = defaultdict(float)
    job_ready:    dict = defaultdict(float)

    ops_by_job: dict = defaultdict(list)
    for op in job_ops:
        ops_by_job[op['job_id']].append(op)
    for jid in ops_by_job:
        ops_by_job[jid].sort(key=lambda o: o['step_idx'])

    # ── Build interleave_rank for round-robin tie-breaking ─────────────
    # Group jobs by product_pk, then assign ranks in round-robin order:
    #   batch 1 of each product, then batch 2 of each product, etc.
    # Within the same batch number, use the LPT product order (heaviest first).
    product_rank: dict = {}   # product_pk -> rank (0 = heaviest = first)
    seen_products = []
    for op in job_ops:
        pk = op['product_pk']
        if pk not in product_rank:
            product_rank[pk] = len(seen_products)
            seen_products.append(pk)

    n_products = max(len(seen_products), 1)

    def _interleave_rank(jid: str, batch_num: int, product_pk: int) -> int:
        # rank = (batch_index * n_products) + product_position
        # → all batch-1 ops come before all batch-2 ops, within each
        #   batch group products appear in LPT order (same as seen_products)
        return (batch_num - 1) * n_products + product_rank.get(product_pk, 0)

    # Pre-compute ranks for all jobs
    job_rank: dict = {}
    for op in job_ops:
        jid = op['job_id']
        if jid not in job_rank:
            job_rank[jid] = _interleave_rank(jid, op['batch_num'], op['product_pk'])

    scheduled      = []
    job_next_step  = {jid: 0 for jid in ops_by_job}

    # Initial heap: all jobs' first steps at t=0
    # Heap entry: (est_start, interleave_rank, jid, step_idx, op)
    heap = []
    for jid, steps in ops_by_job.items():
        if steps:
            heapq.heappush(heap, (0.0, job_rank[jid], jid, 0, steps[0]))

    while heap:
        est, rank, jid, step_idx, op = heapq.heappop(heap)

        machine  = op['machine']
        dur      = op['dur_hours']
        earliest = max(machine_free[machine], job_ready[jid])

        # Stale entry: re-push with updated time (keep same rank for
        # consistency — we only care about rank at equal start times)
        if earliest > est + 1e-9:
            heapq.heappush(heap, (earliest, rank, jid, step_idx, op))
            continue

        start_hrs = earliest
        end_hrs   = start_hrs + dur

        machine_free[machine]  = end_hrs
        job_ready[jid]         = end_hrs
        job_next_step[jid]     = step_idx + 1

        scheduled.append({
            'job_id':       jid,
            'product_pk':   op['product_pk'],
            'batch_num':    op['batch_num'],
            'batch_size':   op['batch_size'],
            'step_number':  op['step'],
            'step_name':    op['step_name'],
            'machine_name': machine,
            'start_hrs':    start_hrs,
            'end_hrs':      end_hrs,
            'dur_hours':    dur,
            'start_dt':     start_dt + timedelta(hours=start_hrs),
            'end_dt':       start_dt + timedelta(hours=end_hrs),
            'batch_id':     jid,
        })

        next_idx = step_idx + 1
        if next_idx < len(ops_by_job[jid]):
            next_op  = ops_by_job[jid][next_idx]
            next_est = max(machine_free[next_op['machine']], end_hrs)
            heapq.heappush(heap, (next_est, rank, jid, next_idx, next_op))

    return scheduled


# ---------------------------------------------------------------------------
# Phase 2: Bounded PuLP optimisation on bottleneck machines  (unchanged)
# ---------------------------------------------------------------------------

def _local_pulp_optimise(schedule, job_ops, start_dt, k_machines):
    """Re-optimise the K most-loaded machines using single-machine MILP."""
    machine_load = defaultdict(float)
    for row in schedule:
        machine_load[row['machine_name']] += row['dur_hours']

    top_machines = sorted(machine_load, key=machine_load.get, reverse=True)[:k_machines]

    by_machine = defaultdict(list)
    for i, row in enumerate(schedule):
        by_machine[row['machine_name']].append(i)

    for machine in top_machines:
        indices = by_machine[machine]
        if len(indices) < 2:
            continue
        if len(indices) > 200:
            indices.sort(key=lambda i: schedule[i]['start_hrs'])
            continue
        _reoptimise_machine(schedule, indices, machine, start_dt)

    return schedule


def _reoptimise_machine(schedule, indices, machine_name, start_dt):
    """Single-machine MILP: minimise makespan for one machine's operations."""
    n   = len(indices)
    ops = [schedule[i] for i in indices]

    lb        = [max(0.0, op['start_hrs'] - op['dur_hours']) for op in ops]
    ub_global = max(op['end_hrs'] for op in ops) * 1.5
    durations = [op['dur_hours'] for op in ops]

    prob = LpProblem(f"SingleMachine_{machine_name}", LpMinimize)
    S    = [LpVariable(f"S_{i}", lowBound=lb[i], upBound=ub_global) for i in range(n)]
    Cmax = LpVariable("Cmax", lowBound=0)

    prob += Cmax

    for i in range(n):
        prob += Cmax >= S[i] + durations[i]

    M = ub_global
    for i in range(n):
        for j in range(i + 1, n):
            y = LpVariable(f"y_{i}_{j}", cat=LpBinary)
            prob += S[j] >= S[i] + durations[i] - M * (1 - y)
            prob += S[i] >= S[j] + durations[j] - M * y

    prob.solve(_SOLVER_TIMED)

    if LpStatus[prob.status] == "Optimal":
        for idx_in_ops, orig_idx in enumerate(indices):
            new_start = value(S[idx_in_ops])
            new_end   = new_start + durations[idx_in_ops]
            schedule[orig_idx]['start_hrs'] = new_start
            schedule[orig_idx]['end_hrs']   = new_end
            schedule[orig_idx]['start_dt']  = start_dt + timedelta(hours=new_start)
            schedule[orig_idx]['end_dt']    = start_dt + timedelta(hours=new_end)


# ---------------------------------------------------------------------------
# KPI helpers
# ---------------------------------------------------------------------------

def compute_schedule_kpis(schedule_rows, makespan_hours):
    """Compute summary KPIs from a list of schedule dicts."""
    if not schedule_rows:
        return {}

    machine_used = defaultdict(float)
    machine_ops  = defaultdict(int)

    for row in schedule_rows:
        machine_used[row['machine_name']] += row['dur_hours']
        machine_ops[row['machine_name']]  += 1

    utilisation = {
        m: round(machine_used[m] / makespan_hours * 100, 2) if makespan_hours > 0 else 0
        for m in machine_used
    }

    bottleneck = max(utilisation, key=utilisation.get) if utilisation else None

    return {
        'makespan_hours':     round(makespan_hours, 2),
        'makespan_days':      round(makespan_hours / 24, 2),
        'total_operations':   len(schedule_rows),
        'machines_used':      len(machine_used),
        'utilisation':        utilisation,
        'bottleneck_machine': bottleneck,
        'bottleneck_util':    utilisation.get(bottleneck, 0) if bottleneck else 0,
    }


# ===========================================================================
# 5.  LEGACY HELPERS  (unchanged)
# ===========================================================================

def optimize_product_batches(products, max_num_batches, min_batch_size, max_batch_size):
    """Per-product batch optimisation (calls the PuLP single-product ILP)."""
    batch_optimization_log = []
    for product in products:
        batch_size, num_batches, ideal_batch = calculate_optimal_batch_size(
            product.demand_2024, max_num_batches, min_batch_size, max_batch_size
        )
        product.batch_size  = batch_size
        product.num_batches = num_batches
        product.save()
        batch_optimization_log.append({
            'item':             product.item,
            'demand':           product.demand_2024,
            'batch_size':       batch_size,
            'num_batches':      num_batches,
            'ideal_batch_size': round(ideal_batch, 2)
        })
    return batch_optimization_log


def calculate_kpis(schedule_records, machine_availability):
    """Calculate high-level scheduling KPIs."""
    if not schedule_records:
        return {}

    max_end        = max(s.end_time   for s in schedule_records)
    min_start      = min(s.start_time for s in schedule_records)
    makespan_hours = (max_end - min_start).total_seconds() / 3600
    makespan_days  = makespan_hours / 24

    machine_stats = {}
    for machine_name in machine_availability:
        used_hours = ProductionSchedule.objects.filter(
            machine__name=machine_name
        ).aggregate(total=Sum('duration_hours'))['total'] or 0
        utilization = (used_hours / makespan_hours * 100) if makespan_hours > 0 else 0
        machine_stats[machine_name] = {
            'used_hours':  round(used_hours, 2),
            'utilization': round(utilization, 2)
        }

    total_units = Product.objects.filter(demand_2024__gt=0).aggregate(
        total=Sum('demand_2024')
    )['total'] or 0

    return {
        'total_makespan_hours':     round(makespan_hours, 2),
        'total_makespan_days':      round(makespan_days, 2),
        'machine_utilization':      machine_stats,
        'total_operations':         len(schedule_records),
        'throughput_units_per_day': round(total_units / makespan_days if makespan_days > 0 else 0, 2),
        'total_units_scheduled':    total_units
    }


def get_batch_params(request):
    """Extract batch optimisation parameters from the API request."""
    return (
        int(request.data.get('max_num_batches',  25)),
        int(request.data.get('min_batch_size',   50)),
        int(request.data.get('max_batch_size',  500)),
    )


# ===========================================================================
# 6.  CSV / DATAFRAME HELPERS  (completely unchanged)
# ===========================================================================

def build_summary(df):
    return {
        "Metric": ["Target Headcount", "Actual Headcount", "Current Backlog", "Currently Open"],
        "Finished Goods": [
            f"{df[df['Step']=='F']['Planned'].sum():,.0f}",
            f"{df[df['Step']=='F']['Realized'].sum():,.0f}",
            f"{df[(df['Step']=='F') & (df['Backlog']>0)]['Backlog'].sum():,.0f}",
            f"{df[df['Step']=='F']['Open'].sum():,.0f}",
        ],
        "Connectors": [
            f"{df[df['Area']=='Assembly']['Planned'].sum():,.0f}",
            f"{df[df['Area']=='Assembly']['Realized'].sum():,.0f}",
            f"{df[(df['Area']=='Assembly') & (df['Backlog']>0)]['Backlog'].sum():,.0f}",
            f"{df[df['Area']=='Assembly']['Open'].sum():,.0f}",
        ],
    }


def calculate_production_outputs(df, finished_filter, connector_filter):
    fg_output   = {}
    conn_output = {}
    for i, shift in enumerate(SHIFT_LABELS):
        col_idx = 14 + i
        if col_idx < 50:
            fg   = df.iloc[:, col_idx][finished_filter].sum()
            conn = df.iloc[:, col_idx][connector_filter].sum()
        else:
            fg = conn = 0
        fg_output[shift]   = f"{fg:,.0f}"   if fg   > 0 else "0"
        conn_output[shift] = f"{conn:,.0f}" if conn > 0 else "0"
    return fg_output, conn_output


def calculate_backlog(df):
    backlog_cols   = range(95, 113)
    backlog_values = []
    for idx in backlog_cols:
        total = df.iloc[2:264, idx]
        total = total.loc[total > 0].sum()
        backlog_values.append(total if total > 0 else 0)
    result = []
    for val in backlog_values:
        result.extend([val, 0])
    return (result + [0] * len(SHIFT_LABELS))[:len(SHIFT_LABELS)]


def calculate_efficiency(df, num_shifts):
    efficiency_list = []
    if 'STD' not in df.columns:
        return [0] * 36

    std_col = pd.to_numeric(
        df.loc[2:1003, 'STD'].astype(str)
        .str.replace(r'[^\d\.\-]', '', regex=True),
        errors='coerce'
    )

    shift_time_hours = 7.67
    available_time   = shift_time_hours * num_shifts
    quantity_columns = list(range(15, 52, 2))

    for col_idx in quantity_columns:
        quantity = pd.to_numeric(
            df.iloc[2:1003, col_idx].astype(str)
            .str.replace(r'[^\d\.\-]', '', regex=True),
            errors='coerce'
        )
        valid           = quantity.notna() & std_col.notna()
        planned_minutes = (quantity[valid] * std_col[valid]).sum()
        planned_hours   = planned_minutes / 60
        efficiency      = (planned_hours / available_time) * 100 if available_time else 0
        efficiency_list.append(efficiency)

    result = []
    for val in efficiency_list:
        result.extend([val, 0])
    return (result + [0] * 36)[:36]


def apply_filters(df, filters):
    for column, val in filters.items():
        if val != "All" and column in df.columns:
            df = df[df[column] == val]
    return df


def clean_numeric_columns(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '').str.strip(),
                errors='coerce'
            )


def clean_text_columns(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()


def clean_shift_columns(df, column_ranges):
    for start, end in column_ranges:
        for idx in range(start, end):
            df.iloc[:, idx] = pd.to_numeric(
                df.iloc[:, idx].astype(str)
                .str.replace(r'[^\d\.\-]', '', regex=True),
                errors='coerce'
            )


def process_frontpage_data(frontpage_df):
    frontpage_df = frontpage_df.head(3)
    df = frontpage_df.rename(columns={
        'SAP TN': 'SAP_TN', 'SAP PL': 'SAP_PL', 'DCC Type': 'DCC_Type'
    })
    df = df[['Item', 'SAP_TN', 'SAP_PL', 'DCC_Type', 'Description', '2024']]
    df = df.rename(columns={'2024': 'Demand_2024'})

    int_cols = ['Item', 'SAP_TN', 'SAP_PL', 'Demand_2024']
    for col in int_cols:
        df[col] = (
            df[col].astype(str)
            .str.replace(',', '', regex=False)
            .replace('None', pd.NA)
            .replace('nan',  pd.NA)
            .pipe(pd.to_numeric, errors='coerce')
            .astype('Int64')
        )

    df = df.where(pd.notna(df), None)
    return df.to_dict(orient='list')


def process_routing_data(process_df):
    process_routing = []

    machines = (
        process_df.iloc[2, 4:].fillna('').astype(str).str.strip().tolist()
    )
    process_steps = (
        process_df.iloc[3, 4:].fillna('').astype(str)
        .str.replace(r'\s+', ' ', regex=True).str.strip().tolist()
    )

    data_df = process_df.iloc[4:].copy()

    for _, row in data_df.iterrows():
        if not str(row.iloc[0]).replace('.0', '').isdigit():
            continue
        try:
            item = int(float(row.iloc[0]))
        except (ValueError, TypeError):
            continue

        for idx in range(len(process_steps)):
            raw_val  = row.iloc[idx + 4]
            time_val = pd.to_numeric(raw_val, errors='coerce')

            if pd.notna(time_val) and time_val > 0:
                process_routing.append({
                    'item':    item,
                    'step':    idx + 1,
                    'machine': machines[idx],
                    'time':    round(float(time_val), 2),
                    'name':    process_steps[idx],
                    'workers': 0.5
                })

    machines_list = list(set(m for m in machines if m))
    return process_routing, machines_list


def _adaptive_bounds(demand: int,
                     max_num_batches: int,
                     user_min: int,
                     user_max: int) -> tuple[int, int]:
    """
    Compute effective [min_batch, max_batch] for this demand level.

    The user-supplied min/max are treated as *hints*.  If they are
    infeasible for this particular demand (e.g., demand=319,908 but
    user_max=500) we fall back to the demand-derived feasible range
    so the ILP always has a solution.

    Feasible range:
      true_min = ceil(demand / max_num_batches)   # smallest batch when N=max
      true_max = demand                            # one big batch (N=1)

    Effective bounds = intersection(user, true).
    If intersection is empty, use the true range (demand-derived).
    """
    if demand <= 0:
        return 1, 1

    true_min = max(1, math.ceil(demand / max_num_batches))
    true_max = demand  # absolute maximum is one batch of the entire demand

    # Intersection with user preferences
    eff_min = max(user_min, true_min)
    eff_max = min(user_max, true_max)

    if eff_min > eff_max:
        # User bounds are completely outside feasible range → use demand-derived
        eff_min = true_min
        eff_max = true_max

    return eff_min, eff_max

def _build_joint_summary(result: dict) -> dict:
    """Build a summary dict from joint optimization result."""
    results = result.get("results", [])
    if not results:
        return {}

    batch_sizes   = [r["new_batch_size"]  for r in results if r["demand"] > 0]
    num_batches   = [r["new_num_batches"] for r in results if r["demand"] > 0]
    improvements  = []
    for r in results:
        try:
            pct = float(r["improvement"].replace("%", ""))
            improvements.append(pct)
        except (ValueError, AttributeError):
            improvements.append(0.0)

    ml = result.get("machine_loads", {})
    most_loaded   = max(ml, key=ml.get) if ml else "—"
    least_loaded  = min(ml, key=ml.get) if ml else "—"
    load_vals     = list(ml.values())
    load_balance  = round(
        (1 - (max(load_vals) - min(load_vals)) / (max(load_vals) + 1e-9)) * 100, 1
    ) if load_vals else 0

    return {
        "total_products":    len(results),
        "total_demand":      sum(r["demand"] for r in results),
        "total_batches":     sum(num_batches),
        "avg_batch_size":    round(np.mean(batch_sizes), 1) if batch_sizes else 0,
        "avg_improvement":   round(np.mean(improvements), 1) if improvements else 0,
        "most_loaded_machine":  most_loaded,
        "least_loaded_machine": least_loaded,
        "load_balance_score": load_balance,
        "makespan_proxy_hours": result.get("makespan_proxy", 0),
        "solver_status":     result.get("status", "unknown"),
    }


def _joint_preview_no_save(products, max_num_batches, min_batch_size, max_batch_size):
    """
    Run joint optimization without writing to DB.
    We do this by temporarily overriding the save step.
    """
    from .utils import (
        optimize_product_batches_jointly as _joint,
        _adaptive_bounds,
    )
    import math
    from .models import ProcessStep
    from pulp import (
        LpProblem, LpMinimize, LpVariable, LpContinuous,
        lpSum, value, LpStatus, PULP_CBC_CMD,
    )

    products_list = list(products)
    candidates    = list(range(1, max_num_batches + 1))

    # Collect cycle times
    cycle_times: dict = {}
    all_machine_names: set = set()

    for product in products_list:
        steps = ProcessStep.objects.filter(product=product).select_related("machine")
        cycle_times[product.pk] = {}
        for step in steps:
            mname = step.machine.name
            cycle_times[product.pk][mname] = (
                cycle_times[product.pk].get(mname, 0.0) + step.cycle_time_seconds
            )
            all_machine_names.add(mname)

    all_machines = sorted(all_machine_names)

    prob = LpProblem("JointBatchOptimizationPreview", LpMinimize)
    C    = LpVariable("makespan_proxy", lowBound=0, cat=LpContinuous)
    prob += C

    Y: dict = {}
    load_table: dict = {}

    for product in products_list:
        pk     = product.pk
        demand = int(product.demand_2024) if product.demand_2024 else 0
        if demand <= 0:
            continue

        eff_min, eff_max = _adaptive_bounds(
            demand, max_num_batches, min_batch_size, max_batch_size
        )
        Y[pk] = {}
        load_table[pk] = {}

        for n in candidates:
            Y[pk][n] = LpVariable(f"yp_{pk}_{n}", cat="Binary")
            batch_sz    = math.ceil(demand / n)
            total_units = batch_sz * n
            load_table[pk][n] = {}
            for mname in all_machines:
                ct = cycle_times[pk].get(mname, 0.0)
                load_table[pk][n][mname] = (ct * total_units) / 3600.0

        prob += lpSum(Y[pk][n] for n in candidates) == 1

        for n in candidates:
            batch_sz = math.ceil(demand / n)
            if not (eff_min <= batch_sz <= eff_max):
                prob += Y[pk][n] == 0

    for mname in all_machines:
        expr = []
        for product in products_list:
            pk     = product.pk
            demand = int(product.demand_2024) if product.demand_2024 else 0
            if demand <= 0 or pk not in Y:
                continue
            for n in candidates:
                h = load_table[pk][n].get(mname, 0.0)
                if h > 0:
                    expr.append(h * Y[pk][n])
        if expr:
            prob += lpSum(expr) <= C

    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=120))

    solved = LpStatus[prob.status] == "Optimal"
    machine_loads = {m: 0.0 for m in all_machines}
    results = []

    for product in products_list:
        pk     = product.pk
        demand = int(product.demand_2024) if product.demand_2024 else 0
        old_bs = product.batch_size
        old_nb = product.num_batches
        ideal  = round(demand / max_num_batches, 2) if demand > 0 else 0

        if demand <= 0 or pk not in Y:
            results.append({
                "item": product.item, "description": (product.description or "")[:50],
                "demand": demand, "old_batch_size": old_bs, "old_num_batches": old_nb,
                "new_batch_size": old_bs, "new_num_batches": old_nb,
                "ideal_batch_size": ideal, "improvement": "0%", "source": "skipped",
            })
            continue

        chosen_n = None
        if solved:
            for n in candidates:
                if Y[pk][n].varValue is not None and Y[pk][n].varValue > 0.5:
                    chosen_n = n
                    break

        if chosen_n:
            new_batch = math.ceil(demand / chosen_n)
            new_n     = chosen_n
        else:
            eff_min, eff_max = _adaptive_bounds(
                demand, max_num_batches, min_batch_size, max_batch_size
            )
            new_n     = max(1, min(max_num_batches, math.ceil(demand / eff_min)))
            new_batch = math.ceil(demand / new_n)

        for mname in all_machines:
            ct    = cycle_times[pk].get(mname, 0.0)
            total = new_batch * new_n
            machine_loads[mname] += (ct * total) / 3600.0

        impr = (
            f"{((old_nb - new_n) / old_nb * 100):.1f}%"
            if old_nb and old_nb != new_n else "0%"
        )

        results.append({
            "item": product.item, "description": (product.description or "")[:50],
            "demand": demand, "old_batch_size": old_bs, "old_num_batches": old_nb,
            "new_batch_size": new_batch, "new_num_batches": new_n,
            "ideal_batch_size": ideal, "improvement": impr,
            "source": "joint_ilp" if solved else "fallback",
        })

    proxy = round(value(C), 4) if solved and value(C) is not None else 0.0

    return {
        "status":         "optimal" if solved else "fallback",
        "results":        results,
        "machine_loads":  {m: round(v, 2) for m, v in machine_loads.items()},
        "makespan_proxy": proxy,
        "products_updated": 0,
        "message": f"Preview only — no DB writes. Status: {'optimal' if solved else 'fallback'}",
    }


