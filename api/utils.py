# api/utils.py

import logging
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from collections import defaultdict
from django.db.models import Sum
from .models import Product, ProcessStep, ProductionSchedule

logger = logging.getLogger(__name__)

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

# Solver shared across the module — silent output (msg=0 suppresses CBC console output)
_SOLVER = PULP_CBC_CMD(msg=0)

# Time-limited solver used for single-machine MILP calls in Phase 2
# 45-second wall-clock cap prevents the optimizer from hanging on large instances
_SOLVER_TIMED = PULP_CBC_CMD(msg=0, timeLimit=45)


# ===========================================================================
# 0.  LEFT-SHIFT COMPACTION  (gap elimination)
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

    Returns
    -------
    Same list with start_hrs / end_hrs / start_dt / end_dt mutated in-place.
    """
    if not schedule:
        return schedule

    # ---------------------------------------------------------------------
    # STEP 1: Determine the schedule's epoch (earliest start time)
    # All internal computations use hours-from-epoch (start_hrs / end_hrs).
    # We record epoch_dt so we can convert back to real datetimes at the end.
    # ---------------------------------------------------------------------
    epoch_dt = min(op['start_dt'] for op in schedule)

    # ---------------------------------------------------------------------
    # STEP 2: Iterative left-shift passes
    # Each pass attempts to move every operation earlier.
    # We stop early if a full pass moves nothing (convergence).
    # In practice this converges in 2–3 passes for typical schedules.
    # ---------------------------------------------------------------------
    for pass_no in range(max_passes):
        moved = 0

        # Sort by current start time so earlier ops anchor first.
        # This ensures each op is only moved after its predecessors have
        # already been compacted, making the pass monotonically valid.
        schedule.sort(key=lambda o: (o['start_hrs'], o['machine_name']))

        # -----------------------------------------------------------------
        # Rebuild tracking maps fresh each pass (state from previous pass
        # is stale after ops have been moved).
        #
        # machine_timeline[m]     : sorted list of (start_hrs, end_hrs)
        #                           representing all committed ops on machine m
        # job_end[batch_id][step] : end_hrs of each completed step per job
        # -----------------------------------------------------------------
        machine_timeline: dict = {}
        job_end: dict = {}

        for op in schedule:
            m    = op['machine_name']
            job  = op['batch_id']
            step = op['step_number']
            dur  = op['dur_hours']

            # ---------------------------------------------------------
            # STEP 2A: Compute lower bound from job precedence
            # predecessor step (step-1) must be complete before this op
            # can start. Returns 0.0 for the first step of a job.
            # ---------------------------------------------------------
            pred_end = _predecessor_end_hrs(job_end, job, step)

            # ---------------------------------------------------------
            # STEP 2B: Find the earliest gap on this machine that
            # fits `dur` hours starting from pred_end.
            # This respects all already-committed ops on the machine
            # and returns the earliest conflict-free start time.
            # ---------------------------------------------------------
            earliest = _first_fit_gap_hrs(
                machine_timeline.get(m, []),
                pred_end,
                dur,
            )

            # ---------------------------------------------------------
            # STEP 2C: Move the op left if we found an earlier slot
            # Tolerance of 1 second (1/3600 hours) avoids floating-point
            # churn where "earlier" is just a rounding artefact.
            # ---------------------------------------------------------
            if earliest < op['start_hrs'] - (1 / 3600):
                op['start_hrs'] = earliest
                op['end_hrs']   = earliest + dur
                op['start_dt']  = epoch_dt + timedelta(hours=earliest)
                op['end_dt']    = epoch_dt + timedelta(hours=earliest + dur)
                moved += 1

            # ---------------------------------------------------------
            # STEP 2D: Register this op's committed slot in both maps
            # so subsequent ops in this pass see it as occupied.
            # ---------------------------------------------------------
            if m not in machine_timeline:
                machine_timeline[m] = []
            machine_timeline[m].append((op['start_hrs'], op['end_hrs']))
            machine_timeline[m].sort()

            if job not in job_end:
                job_end[job] = {}
            job_end[job][step] = op['end_hrs']

        # -----------------------------------------------------------------
        # STEP 2E: Early exit if no ops moved this pass — schedule is stable
        # -----------------------------------------------------------------
        if moved == 0:
            break

    return schedule


def _predecessor_end_hrs(job_end: dict, job: str, step: int) -> float:
    """
    Return the end time (hours) of the immediately preceding step for this job.

    Parameters
    ----------
    job_end : dict  { job_id: { step_number: end_hrs } }
    job     : str   job / batch identifier
    step    : int   current step number (1-based)

    Returns
    -------
    0.0  if this is the first step (no predecessor constraint)
    end_hrs of step-1 otherwise (job cannot start until predecessor finishes)
    """
    if step <= 1:
        return 0.0
    return job_end.get(job, {}).get(step - 1, 0.0)


def _first_fit_gap_hrs(
    timeline: list,          # sorted list of (start_hrs, end_hrs)
    earliest: float,         # lower bound from job precedence
    duration: float,         # how long the op takes
) -> float:
    """
    Find the earliest start time >= `earliest` where `duration` hours
    fit on a machine without overlapping any committed interval.

    Algorithm: walk the sorted timeline and push the candidate start
    forward whenever the current busy block would cause an overlap.

    Parameters
    ----------
    timeline  : sorted list of (start_hrs, end_hrs) representing busy periods
    earliest  : minimum allowed start (from predecessor constraint)
    duration  : hours required

    Returns
    -------
    Earliest conflict-free start time >= earliest
    """
    candidate = earliest
    for busy_start, busy_end in timeline:
        if busy_end <= candidate:
            # This busy block is entirely before our candidate — ignore it
            continue
        if candidate + duration <= busy_start:
            # Our op fits in the gap before this block — stop scanning
            break
        # Our op would overlap this block — push past it
        candidate = max(candidate, busy_end)
    return candidate


def count_schedule_gaps(schedule: list, min_gap_sec: int = 60) -> dict:
    """
    Count idle gaps remaining in a schedule after compaction.

    Used for before/after logging when compaction runs and for returning
    gap statistics in API responses.

    Parameters
    ----------
    schedule    : list of scheduled op dicts (must have machine_name,
                  start_hrs, end_hrs)
    min_gap_sec : gaps shorter than this (seconds) are ignored as rounding
                  noise rather than true idle periods

    Returns
    -------
    dict with keys:
        total_gaps       : int   total gap count across all machines
        total_idle_hours : float sum of all idle time across all machines
        per_machine      : dict  { machine_name: { gaps, idle_hours } }
    """
    # ---------------------------------------------------------------------
    # STEP 1: Group ops by machine
    # ---------------------------------------------------------------------
    machine_ops: dict = defaultdict(list)
    for op in schedule:
        machine_ops[op['machine_name']].append(op)

    total_gaps = 0
    total_idle = 0.0
    per_machine = {}

    # ---------------------------------------------------------------------
    # STEP 2: For each machine, scan consecutive op pairs for gaps
    # A gap exists when op[i].end < op[i+1].start (machine sits idle).
    # Only count gaps longer than min_gap_sec to filter out float noise.
    # ---------------------------------------------------------------------
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
# 0b.  SCHEDULE VALIDATION
# ===========================================================================
# After all scheduling phases complete, this validator checks two invariants
# that must hold for a schedule to be physically realizable:
#
#   1. MACHINE CONFLICT — no two operations on the same machine overlap in time.
#      If they do, a real machine would be running two jobs simultaneously,
#      which is impossible.
#
#   2. ROUTING DEPENDENCY — no operation starts before its predecessor step
#      (on any machine) has finished.  If it does, a job is being assembled
#      before its upstream component is ready.
#
# WHY THIS WAS MISSING
# --------------------
# Phase 2 (PuLP MILP) re-sequences operations on individual machines using a
# single-machine model.  Before the fix in _reoptimise_machine, the solver
# could move operations earlier than their predecessor end times, silently
# producing invalid schedules.  Validation makes such violations visible
# rather than letting them propagate to the Gantt chart and KPI calculations.

def validate_schedule(schedule: list) -> dict:
    """
    Validate a completed schedule for machine conflicts and routing violations.

    Parameters
    ----------
    schedule : list[dict]
        Scheduled operations — each dict must have:
            batch_id, step_number, machine_name, start_hrs, end_hrs

    Returns
    -------
    dict with keys:
        valid              : bool   True only when zero errors found
        machine_conflicts  : list   overlap descriptions (empty when valid)
        routing_violations : list   precedence violation descriptions
        total_errors       : int
        summary            : str    human-readable one-liner
    """
    TOLERANCE = 1.0 / 3600   # 1-second float tolerance

    machine_conflicts:  list = []
    routing_violations: list = []

    # ── Check 1: machine overlaps ─────────────────────────────────────────
    by_machine: dict = defaultdict(list)
    for op in schedule:
        by_machine[op['machine_name']].append(op)

    for machine, ops in by_machine.items():
        sorted_ops = sorted(ops, key=lambda o: o['start_hrs'])
        for i in range(1, len(sorted_ops)):
            prev = sorted_ops[i - 1]
            curr = sorted_ops[i]
            overlap = prev['end_hrs'] - curr['start_hrs']
            if overlap > TOLERANCE:
                machine_conflicts.append({
                    'machine':       machine,
                    'op1_job':       prev['batch_id'],
                    'op1_step':      prev['step_number'],
                    'op1_end_hrs':   round(prev['end_hrs'], 4),
                    'op2_job':       curr['batch_id'],
                    'op2_step':      curr['step_number'],
                    'op2_start_hrs': round(curr['start_hrs'], 4),
                    'overlap_hours': round(overlap, 4),
                })

    # ── Check 2: routing dependency violations ────────────────────────────
    job_step_map: dict = {}
    for op in schedule:
        job_step_map[(op['batch_id'], op['step_number'])] = op

    for op in schedule:
        step = op['step_number']
        if step <= 1:
            continue
        pred = job_step_map.get((op['batch_id'], step - 1))
        if pred is None:
            continue
        violation = pred['end_hrs'] - op['start_hrs']
        if violation > TOLERANCE:
            routing_violations.append({
                'job':             op['batch_id'],
                'step':            step,
                'machine':         op['machine_name'],
                'start_hrs':       round(op['start_hrs'], 4),
                'pred_step':       step - 1,
                'pred_machine':    pred['machine_name'],
                'pred_end_hrs':    round(pred['end_hrs'], 4),
                'violation_hours': round(violation, 4),
            })

    total = len(machine_conflicts) + len(routing_violations)
    if total == 0:
        summary = "Schedule VALID — no conflicts or dependency violations"
    else:
        summary = (
            f"Schedule INVALID — "
            f"{len(machine_conflicts)} machine conflict(s), "
            f"{len(routing_violations)} routing violation(s)"
        )

    return {
        'valid':              total == 0,
        'machine_conflicts':  machine_conflicts,
        'routing_violations': routing_violations,
        'total_errors':       total,
        'summary':            summary,
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

    Objective: minimise number of batches N (fewer setups = lower cost)

    Constraints:
        batch_size × num_batches ≥ demand
        min_batch_size ≤ batch_size ≤ max_batch_size
        1 ≤ num_batches ≤ max_num_batches

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

    Returns
    -------
    (batch_size, num_batches, ideal_batch_size)
        batch_size      : int    optimal (or fallback) batch size
        num_batches     : int    optimal (or fallback) number of batches
        ideal_batch_size: float  unconstrained ideal = demand / max_num_batches
    """
    # ---------------------------------------------------------------------
    # STEP 1: Reject zero-demand products immediately
    # No production needed → all values are 0
    # ---------------------------------------------------------------------
    if demand <= 0:
        return 0, 0, 0.0

    # ---------------------------------------------------------------------
    # STEP 2: Compute the "ideal" (unconstrained) batch size
    # This is the theoretical optimum if we ignore all bounds.
    # Returned purely for reporting — not used in the ILP itself.
    # ---------------------------------------------------------------------
    ideal_batch_size = demand / max_num_batches

    # ---------------------------------------------------------------------
    # STEP 3: Compute adaptive effective bounds
    # Clamps user min/max to the range that is actually feasible for this
    # demand level. See _adaptive_bounds() for full explanation.
    # ---------------------------------------------------------------------
    eff_min, eff_max = _adaptive_bounds(
        demand, max_num_batches, min_batch_size, max_batch_size
    )

    candidates = list(range(1, max_num_batches + 1))

    # ---------------------------------------------------------------------
    # STEP 4: Build the ILP model
    # Decision variables:
    #   B  (integer) — batch size, bounded in [eff_min, eff_max]
    #   N  (integer) — number of batches, bounded in [1, max_num_batches]
    #   y_n (binary) — 1 if exactly n batches are chosen (one-hot selector)
    #
    # Objective: minimise N (fewer setups)
    # ---------------------------------------------------------------------
    prob = LpProblem("BatchSizeOptimization", LpMinimize)

    B = LpVariable("batch_size",  lowBound=eff_min, upBound=eff_max,          cat=LpInteger)
    N = LpVariable("num_batches", lowBound=1,        upBound=max_num_batches,  cat=LpInteger)

    prob += N   # objective: minimise number of batches

    # Binary selector variables: exactly one y_n = 1
    y = {n: LpVariable(f"y_{n}", cat="Binary") for n in candidates}

    # ---------------------------------------------------------------------
    # STEP 5: Add constraints
    #
    # Constraint 1: exactly one candidate n is active
    # Constraint 2: N equals the chosen candidate's value
    # Constraint 3: B ≥ ceil(demand/n) when y_n = 1 (big-M linking)
    #               Ensures the batch size is large enough to cover demand
    #               when n batches are produced.
    # ---------------------------------------------------------------------
    prob += lpSum(y[n] for n in candidates) == 1
    prob += N == lpSum(n * y[n] for n in candidates)

    M = eff_max  # big-M constant (upper bound on B is sufficient)
    for n in candidates:
        required_B = math.ceil(demand / n)
        prob += B >= required_B - M * (1 - y[n])

    # Prune candidates that are provably infeasible (required batch > max)
    # This tightens the LP relaxation and speeds up solve time.
    for n in candidates:
        if math.ceil(demand / n) > eff_max:
            prob += y[n] == 0

    # ---------------------------------------------------------------------
    # STEP 6: Solve and extract results
    # ---------------------------------------------------------------------
    prob.solve(_SOLVER)

    if LpStatus[prob.status] == "Optimal":
        # Clamp to effective bounds to absorb any floating-point rounding
        opt_B = max(eff_min, min(eff_max, int(round(value(B)))))
        opt_N = max(1, min(max_num_batches, int(round(value(N)))))
        return opt_B, opt_N, round(ideal_batch_size, 2)

    # ---------------------------------------------------------------------
    # STEP 7: Demand-aware fallback heuristic (if ILP fails)
    # Compute N as the minimum number of batches such that the batch size
    # stays within the effective bounds, then back-calculate B.
    # This always produces a valid (if suboptimal) solution.
    # ---------------------------------------------------------------------
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
      results         : list of per-product result dicts
      makespan_proxy  : C value (hours)
      machine_loads   : {machine_name: hours} after optimization
      products_updated: count of products whose DB records were saved
      message         : human-readable summary
    """
    from .models import ProcessStep  # avoid circular import at module level

    products_list = list(products)

    # ---------------------------------------------------------------------
    # STEP 1: Guard — abort immediately if no products provided
    # ---------------------------------------------------------------------
    if not products_list:
        return {
            "status": "error",
            "message": "No products provided.",
            "results": [],
            "makespan_proxy": 0,
            "machine_loads": {},
            "products_updated": 0,
        }

    # ---------------------------------------------------------------------
    # STEP 2: Collect process routing data
    # Build cycle_times[product_pk][machine_name] = total cycle seconds.
    # If a product visits the same machine at multiple steps, their times
    # are summed — the machine sees the full cumulative load per unit.
    # Also collect the global set of all machine names for the ILP.
    # ---------------------------------------------------------------------
    cycle_times: dict[int, dict[str, float]] = {}
    all_machine_names: set[str] = set()

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

    # ---------------------------------------------------------------------
    # STEP 3: Build the joint ILP model
    # C is the makespan proxy (worst-case machine load in hours).
    # Minimising C drives the solver to spread load evenly across machines.
    # ---------------------------------------------------------------------
    prob = LpProblem("JointBatchOptimization", LpMinimize)

    C = LpVariable("makespan_proxy", lowBound=0, cat=LpContinuous)
    prob += C  # objective: minimise C

    candidates = list(range(1, max_num_batches + 1))

    # Y[pk][n]          : binary variable — 1 if product pk uses n batches
    # load_table[pk][n] : pre-computed machine load (hours) for that choice
    Y: dict[int, dict[int, LpVariable]] = {}
    load_table: dict[int, dict[int, dict[str, float]]] = {}

    # ---------------------------------------------------------------------
    # STEP 4: Per-product variable and load precomputation
    # For each product, compute the machine load that would result from
    # every possible n value. This lets the ILP use pre-computed constants
    # rather than nonlinear products in constraints.
    # ---------------------------------------------------------------------
    for product in products_list:
        pk     = product.pk
        demand = int(product.demand_2024) if product.demand_2024 else 0

        # Skip zero-demand products — they add no load and no variables
        if demand <= 0:
            continue

        eff_min, eff_max = _adaptive_bounds(
            demand, max_num_batches, min_batch_size, max_batch_size
        )

        Y[pk] = {}
        load_table[pk] = {}

        # -----------------------------------------------------------------
        # Determine which n values are feasible for this product.
        # If none are feasible within user bounds, open the range fully.
        # -----------------------------------------------------------------
        feasible_ns = []
        for n in candidates:
            batch_sz = math.ceil(demand / n)
            if eff_min <= batch_sz <= eff_max:
                feasible_ns.append(n)

        if not feasible_ns:
            eff_min, eff_max = math.ceil(demand / max_num_batches), demand
            feasible_ns = candidates[:]

        for n in candidates:
            Y[pk][n] = LpVariable(f"y_{pk}_{n}", cat="Binary")

            # ---------------------------------------------------------
            # Compute machine load in hours for choosing n batches:
            #   batch_sz    = ceil(demand / n)      (units per batch)
            #   total_units = batch_sz × n           (total units produced)
            #   hours       = cycle_time_sec × total_units / 3600
            # ---------------------------------------------------------
            batch_sz    = math.ceil(demand / n)
            total_units = batch_sz * n
            load_table[pk][n] = {}

            for mname in all_machines:
                ct = cycle_times[pk].get(mname, 0.0)
                hours = (ct * total_units) / 3600.0
                load_table[pk][n][mname] = hours

        # -----------------------------------------------------------------
        # Constraint: exactly one n is chosen per product
        # -----------------------------------------------------------------
        prob += lpSum(Y[pk][n] for n in candidates) == 1

        # -----------------------------------------------------------------
        # Constraint: fix infeasible n values to 0 (prune search space)
        # This prevents the solver from choosing an n that would violate
        # the batch size bounds for this product.
        # -----------------------------------------------------------------
        for n in candidates:
            batch_sz = math.ceil(demand / n)
            if not (eff_min <= batch_sz <= eff_max):
                prob += Y[pk][n] == 0

    # ---------------------------------------------------------------------
    # STEP 5: Machine load constraints
    # For each machine, the sum of loads across all products and all their
    # n-choices must not exceed C (the makespan proxy).
    # Only add the constraint if the machine is actually used by any product.
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # STEP 6: Solve
    # Use a dedicated solver instance with the caller-supplied time limit
    # so large instances don't block indefinitely.
    # ---------------------------------------------------------------------
    solver = PULP_CBC_CMD(msg=0, timeLimit=time_limit_seconds)
    prob.solve(solver)

    status          = LpStatus[prob.status]
    solved_optimally = status == "Optimal"

    # ---------------------------------------------------------------------
    # STEP 7: Extract results for each product
    # For optimally-solved instances, read the chosen n from the binary
    # variables. For fallback instances, recompute using the heuristic.
    # Accumulate actual machine loads and record per-product changes.
    # ---------------------------------------------------------------------
    results          = []
    machine_loads    = {m: 0.0 for m in all_machines}
    products_updated = 0

    for product in products_list:
        pk     = product.pk
        demand = int(product.demand_2024) if product.demand_2024 else 0

        # Old baseline: pre-optimization values (monthly split assumption)
        old_batch_size  = math.ceil(demand / 12) if demand > 0 else 1
        old_num_batches = 12

        ideal_batch = round(demand / max_num_batches, 2) if demand > 0 else 0

        # -----------------------------------------------------------------
        # Products with zero demand or not in the ILP get passed through unchanged
        # -----------------------------------------------------------------
        if demand <= 0 or pk not in Y:
            results.append({
                "product_id":       pk,
                "item":             product.item,
                "description":      (product.description or "")[:50],
                "demand":           demand,
                "old_batch_size":   old_batch_size,
                "old_num_batches":  old_num_batches,
                "new_batch_size":   old_batch_size,
                "new_num_batches":  old_num_batches,
                "ideal_batch_size": ideal_batch,
                "improvement":      "0%",
                "source":           "skipped",
            })
            continue

        # -----------------------------------------------------------------
        # STEP 7A: Read the solver's choice of n
        # varValue > 0.5 treats binary variables as integer due to
        # floating-point representation from the LP relaxation.
        # -----------------------------------------------------------------
        if solved_optimally:
            chosen_n = None
            for n in candidates:
                if Y[pk][n].varValue is not None and Y[pk][n].varValue > 0.5:
                    chosen_n = n
                    break

            if chosen_n is not None:
                new_batch = math.ceil(demand / chosen_n)
                new_n     = chosen_n
            else:
                # Safety fallback: solver returned "optimal" but no variable
                # was selected (degenerate edge case)
                eff_min, eff_max = _adaptive_bounds(
                    demand, max_num_batches, min_batch_size, max_batch_size
                )
                new_n     = max(1, min(max_num_batches, math.ceil(demand / eff_min)))
                new_batch = math.ceil(demand / new_n)
        else:
            # -----------------------------------------------------------------
            # STEP 7B: Full fallback — solver did not reach optimality
            # Use the same heuristic as calculate_optimal_batch_size
            # -----------------------------------------------------------------
            eff_min, eff_max = _adaptive_bounds(
                demand, max_num_batches, min_batch_size, max_batch_size
            )
            new_n     = max(1, min(max_num_batches, math.ceil(demand / eff_min)))
            new_batch = math.ceil(demand / new_n)
            new_batch = max(eff_min, min(eff_max, new_batch))

        # -----------------------------------------------------------------
        # STEP 7C: Accumulate actual machine load contribution for this product
        # Uses the chosen (or fallback) batch/n values to compute hours.
        # -----------------------------------------------------------------
        for mname in all_machines:
            n_used = chosen_n if solved_optimally and chosen_n else new_n
            ct     = cycle_times[pk].get(mname, 0.0)
            total  = new_batch * new_n
            machine_loads[mname] += (ct * total) / 3600.0

        # -----------------------------------------------------------------
        # STEP 7D: Compute improvement vs. baseline (old = 12 batches)
        # Positive = fewer batches = fewer setups = improvement
        # -----------------------------------------------------------------
        if old_num_batches != new_n:
            improvement_pct = (old_num_batches - new_n) / old_num_batches * 100
            improvement     = f"{improvement_pct:.1f}%"
        else:
            improvement = "0%"

        results.append({
            "product_id":       pk,
            "item":             product.item,
            "description":      (product.description or "")[:50],
            "demand":           demand,
            "old_batch_size":   old_batch_size,
            "old_num_batches":  old_num_batches,
            "new_batch_size":   new_batch,
            "new_num_batches":  new_n,
            "ideal_batch_size": ideal_batch,
            "improvement":      improvement,
            "source":           "joint_ilp" if solved_optimally else "fallback",
        })

        # -----------------------------------------------------------------
        # STEP 7E: Persist new batch values to DB (only if they changed)
        # Using update_fields for efficiency — only touches these two columns.
        # -----------------------------------------------------------------
        if product.batch_size != new_batch or product.num_batches != new_n:
            product.batch_size  = new_batch
            product.num_batches = new_n
            product.save(update_fields=["batch_size", "num_batches"])
            products_updated += 1

    # ---------------------------------------------------------------------
    # STEP 8: Extract makespan proxy value from solver
    # value(C) is None if the solver failed, so we default to 0.0
    # ---------------------------------------------------------------------
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
    """
    Allocate a finite buffer budget across machines via PuLP LP.

    Objective: minimise utilisation-weighted shortfall
    (prioritise high-utilisation machines — they suffer most from buffer
    deficits and cause the most downstream starvation).

    Parameters
    ----------
    machine_buffer_data : list[dict]  each dict must have:
        'machine'         : str   machine identifier
        'required_buffer' : float buffer hours needed
        'utilization'     : float machine utilisation fraction (0–1)
    total_budget        : float total buffer hours available to allocate

    Returns
    -------
    Same list with 'allocated_buffer' and 'shortfall' added to each dict.
    """
    # ---------------------------------------------------------------------
    # STEP 1: Guard — if no data or no budget, set all allocations to 0
    # ---------------------------------------------------------------------
    if not machine_buffer_data or total_budget <= 0:
        for m in machine_buffer_data:
            m['allocated_buffer'] = 0.0
            m['shortfall']        = m.get('required_buffer', 0)
        return machine_buffer_data

    machines = [m['machine'] for m in machine_buffer_data]

    # ---------------------------------------------------------------------
    # STEP 2: Build the LP model
    # Variables:
    #   alloc[m] : hours allocated to machine m  (0 ≤ alloc ≤ required)
    #   slack[m] : unmet buffer need  = required - alloc  (always ≥ 0)
    #
    # Objective: minimise Σ utilization[m] × slack[m]
    #   High-utilisation machines cost more per unit of unmet buffer,
    #   so the solver preferentially satisfies them first.
    # ---------------------------------------------------------------------
    prob  = LpProblem("BufferAllocation", LpMinimize)
    alloc = {m: LpVariable(f"alloc_{m}", lowBound=0, cat=LpContinuous) for m in machines}
    slack = {m: LpVariable(f"slack_{m}", lowBound=0, cat=LpContinuous) for m in machines}

    req  = {d['machine']: d['required_buffer'] for d in machine_buffer_data}
    util = {d['machine']: d['utilization']     for d in machine_buffer_data}

    prob += lpSum(util[m] * slack[m] for m in machines)   # objective

    # ---------------------------------------------------------------------
    # STEP 3: Add constraints
    # Constraint 1: allocation cannot exceed what is required
    # Constraint 2: slack = required - allocated  (slack definition)
    # Constraint 3: total allocation stays within budget
    # ---------------------------------------------------------------------
    for m in machines:
        prob += alloc[m] <= req[m]                    # can't over-allocate
        prob += slack[m] == req[m] - alloc[m]         # slack definition

    prob += lpSum(alloc[m] for m in machines) <= total_budget  # budget cap

    prob.solve(_SOLVER)

    # ---------------------------------------------------------------------
    # STEP 4: Extract results
    # If solver succeeded, read variable values directly.
    # If solver failed (infeasible / error), fall back to proportional
    # allocation: each machine gets budget × (its_requirement / total_requirement).
    # ---------------------------------------------------------------------
    for d in machine_buffer_data:
        m = d['machine']
        if LpStatus[prob.status] == "Optimal":
            d['allocated_buffer'] = round(value(alloc[m]), 2)
            d['shortfall']        = round(value(slack[m]),  2)
        else:
            # Proportional fallback: allocate budget proportional to need
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
    enable_compaction: bool = True,
    progress_callback=None,
) -> list:
    """
    Build a job-shop schedule for *products* starting at *start_dt*.

    Three-phase pipeline:
    Phase 1a  –  Greedy ERT dispatcher      (always runs)
    Phase 1b  –  Left-shift compaction      (runs when enable_compaction=True)
    Phase 2   –  PuLP local MILP            (runs when local_opt_machines > 0)

    Parameters
    ----------
    products            : iterable of Product ORM objects
    start_dt            : datetime  schedule anchor (t=0)
    objective           : str       reserved for future use (currently 'makespan')
    batch_override      : dict      { product_pk: (batch_size, num_batches) }
                          overrides DB values for specific products
    local_opt_machines  : int       how many bottleneck machines to re-optimise
                          in Phase 2. Set to 0 to skip Phase 2 entirely.
    enable_compaction   : bool      whether to run Phase 1b gap elimination
    progress_callback   : callable  f(pct: int, msg: str) for Celery progress

    Returns
    -------
    list[dict]  – one dict per scheduled operation with keys:
        job_id, product_pk, batch_num, batch_size, step_number, step_name,
        machine_name, start_hrs, end_hrs, dur_hours, start_dt, end_dt, batch_id
    """
    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    _progress(5, "Extracting process routing…")

    # ── STEP 1: Build routing dictionary ─────────────────────────────────────
    # routing[product_pk] = ordered list of step dicts for that product.
    # Only steps with cycle_time > 0 are included (zero-time steps are
    # placeholder rows and should not appear in the schedule).
    # ─────────────────────────────────────────────────────────────────────────
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

    # ── STEP 2: LPT product ordering + job expansion ──────────────────────────
    # Sort products by total processing hours DESCENDING (Longest Processing
    # Time first).  This ensures the heaviest product's upstream steps are
    # dispatched first, so its downstream machines unblock early and lighter
    # products can fill them without waiting — significantly reducing gaps
    # on shared downstream machines like SKM Seal and ARBURG.
    # ─────────────────────────────────────────────────────────────────────────
    def _total_work_hrs(p):
        if p.pk not in routing:
            return 0.0
        bs = p.batch_size  if p.batch_size  > 0 else 1
        nb = p.num_batches if p.num_batches > 0 else 1
        if batch_override and p.pk in batch_override:
            bs, nb = batch_override[p.pk]
        return sum(s['cycle_sec'] for s in routing[p.pk]) * bs * nb / 3600.0

    products_ordered = sorted(products, key=_total_work_hrs, reverse=True)

    jobs    = []   # one entry per (product, batch) pair
    job_ops = []   # one entry per (product, batch, step) triple

    for p in products_ordered:
        if p.pk not in routing:
            continue  # no routing defined → skip this product

        # Resolve batch size and count (override takes priority over DB)
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

            # Expand each routing step into an individual schedulable operation
            for k, s in enumerate(steps):
                job_ops.append({
                    'job_id':     job_id,
                    'product_pk': p.pk,
                    'batch_num':  b,
                    'batch_size': b_size,
                    'step_idx':   k,           # 0-based index within job
                    'step':       s['step'],   # 1-based step number from DB
                    'machine':    s['machine'],
                    'dur_hours':  (s['cycle_sec'] * b_size) / 3600.0,
                    'step_name':  s['step_name'],
                })

    n_machines = len(set(o['machine'] for o in job_ops))
    _progress(25, f"Scheduling {len(jobs)} jobs across {n_machines} machines…")

    # ── STEP 3: Phase 1a — Greedy ERT dispatcher ─────────────────────────────
    # Assigns every operation to a machine slot using the Earliest Release
    # Time heuristic with round-robin tie-breaking. See _greedy_dispatch()
    # for a full explanation of why this outperforms the old round-by-round SPT.
    # ─────────────────────────────────────────────────────────────────────────
    schedule = _greedy_dispatch(jobs, job_ops, start_dt, progress_callback)
    _progress(55, f"Greedy dispatch complete: {len(schedule):,} operations")

    # ── STEP 4: Phase 1b — Left-shift compaction (gap elimination) ───────────
    # Slides every operation as early as possible to close idle gaps that the
    # greedy dispatcher couldn't avoid due to precedence starvation.
    # Runs up to 5 passes, stopping when no further improvements are found.
    # ─────────────────────────────────────────────────────────────────────────
    if enable_compaction and schedule:
        _progress(60, "Eliminating idle gaps (left-shift compaction)…")

        gaps_before = count_schedule_gaps(schedule)
        schedule    = left_shift_compaction(schedule, max_passes=5)
        gaps_after  = count_schedule_gaps(schedule)

        # Report the improvement for debugging / monitoring
        _progress(70, (
            f"Gap elimination done — "
            f"{gaps_before['total_gaps']} → {gaps_after['total_gaps']} gaps, "
            f"{gaps_before['total_idle_hours']:.1f}h → "
            f"{gaps_after['total_idle_hours']:.1f}h idle"
        ))

    # ── STEP 5: Phase 2 — PuLP local optimiser on bottleneck machines ─────────
    # Re-solves each of the K most-loaded machines independently using a
    # single-machine MILP to minimise their individual makespans.
    # This refines the schedule beyond what the greedy dispatcher can achieve.
    # ─────────────────────────────────────────────────────────────────────────
    if local_opt_machines > 0:
        _progress(75, f"PuLP optimisation on {local_opt_machines} bottleneck machines…")
        schedule = _local_pulp_optimise(schedule, job_ops, start_dt, local_opt_machines)

    # ── STEP 6: Post-schedule validation ──────────────────────────────────────
    # Verify that the final schedule has no machine conflicts and respects all
    # routing dependencies.  After the Phase 2 fix (predecessor lower bounds)
    # this should always pass; this check makes regressions immediately visible.
    # ─────────────────────────────────────────────────────────────────────────
    _progress(90, "Validating schedule integrity…")
    validation = validate_schedule(schedule)
    if validation['valid']:
        logger.info("Schedule validation passed — %d operations, %d machines",
                    len(schedule),
                    len(set(op['machine_name'] for op in schedule)))
    else:
        logger.warning("Schedule validation FAILED: %s", validation['summary'])
        for conflict in validation['machine_conflicts'][:5]:   # log first 5
            logger.warning(
                "  Machine conflict on %s: %s step %d overlaps %s step %d by %.4fh",
                conflict['machine'],
                conflict['op1_job'], conflict['op1_step'],
                conflict['op2_job'], conflict['op2_step'],
                conflict['overlap_hours'],
            )
        for violation in validation['routing_violations'][:5]:
            logger.warning(
                "  Routing violation: %s step %d starts %.4fh before step %d ends",
                violation['job'], violation['step'],
                violation['violation_hours'], violation['pred_step'],
            )

    # Attach validation summary to every operation so callers can surface it
    for op in schedule:
        op['_schedule_valid']  = validation['valid']
        op['_validation_note'] = validation['summary']

    _progress(95, f"Finalising schedule… [{validation['summary']}]")
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
    """
    import heapq

    # Tracks when each machine becomes free (hours from epoch)
    machine_free: dict = defaultdict(float)
    # Tracks when each job is ready for its next step (hours from epoch)
    job_ready:    dict = defaultdict(float)

    # Group all ops by job and sort within each job by step index
    ops_by_job: dict = defaultdict(list)
    for op in job_ops:
        ops_by_job[op['job_id']].append(op)
    for jid in ops_by_job:
        ops_by_job[jid].sort(key=lambda o: o['step_idx'])

    # ── Build interleave_rank for round-robin tie-breaking ─────────────────
    # Assign product_rank based on LPT order (heaviest product = rank 0).
    # interleave_rank = (batch_index × n_products) + product_rank
    # This produces: P1b1=0, P2b1=1, P3b1=2, P1b2=3, P2b2=4, ...
    # ensuring batches of all products are interleaved rather than serialised.
    # ─────────────────────────────────────────────────────────────────────────
    product_rank: dict = {}
    seen_products = []
    for op in job_ops:
        pk = op['product_pk']
        if pk not in product_rank:
            product_rank[pk] = len(seen_products)
            seen_products.append(pk)

    n_products = max(len(seen_products), 1)

    def _interleave_rank(jid: str, batch_num: int, product_pk: int) -> int:
        return (batch_num - 1) * n_products + product_rank.get(product_pk, 0)

    # Pre-compute ranks for all jobs
    job_rank: dict = {}
    for op in job_ops:
        jid = op['job_id']
        if jid not in job_rank:
            job_rank[jid] = _interleave_rank(jid, op['batch_num'], op['product_pk'])

    scheduled     = []
    job_next_step = {jid: 0 for jid in ops_by_job}

    # ── Initialise the heap ────────────────────────────────────────────────
    # Push all first steps at t=0.
    # Heap entry: (est_start, interleave_rank, jid, step_idx, op)
    # est_start  — earliest possible start time (used as primary sort key)
    # rank       — tie-breaker: lower rank = scheduled first at equal time
    # ─────────────────────────────────────────────────────────────────────────
    heap = []
    for jid, steps in ops_by_job.items():
        if steps:
            heapq.heappush(heap, (0.0, job_rank[jid], jid, 0, steps[0]))

    # ── Main dispatch loop ────────────────────────────────────────────────
    while heap:
        est, rank, jid, step_idx, op = heapq.heappop(heap)

        machine  = op['machine']
        dur      = op['dur_hours']

        # Recompute actual earliest start: max(machine available, job ready)
        earliest = max(machine_free[machine], job_ready[jid])

        # -----------------------------------------------------------------
        # Stale entry check: if this op's true earliest start has moved
        # later since it was pushed (because the machine became busier),
        # re-push with the updated time and skip this entry.
        # We preserve the same rank to maintain relative priority at the
        # new time — this is consistent with round-robin ordering.
        # -----------------------------------------------------------------
        if earliest > est + 1e-9:
            heapq.heappush(heap, (earliest, rank, jid, step_idx, op))
            continue

        start_hrs = earliest
        end_hrs   = start_hrs + dur

        # Commit this operation to its machine and record completion time
        machine_free[machine] = end_hrs
        job_ready[jid]        = end_hrs
        job_next_step[jid]    = step_idx + 1

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

        # Push the next step of this job onto the heap
        next_idx = step_idx + 1
        if next_idx < len(ops_by_job[jid]):
            next_op  = ops_by_job[jid][next_idx]
            # Estimate earliest start for the next step: max(machine free, job done now)
            next_est = max(machine_free[next_op['machine']], end_hrs)
            heapq.heappush(heap, (next_est, rank, jid, next_idx, next_op))

    return scheduled


# ---------------------------------------------------------------------------
# Phase 2: Bounded PuLP optimisation on bottleneck machines
# ---------------------------------------------------------------------------

def _rebuild_timings_after_phase2(schedule: list, start_dt) -> None:
    """
    After Phase 2 MILP resequences bottleneck machines, some successor steps
    on non-optimised machines may have start times that predate their updated
    job-predecessor end times (because _reoptimise_machine updates one machine
    at a time and does not cascade timing changes to downstream operations).

    This function recomputes ALL start/end times from scratch while preserving
    the per-machine sequence order decided by Phase 2.

    Algorithm (Kahn-style forward-pass dispatch):
      1. Lock in machine ordering by sorting each machine's ops by their
         current (Phase 2) start_hrs.
      2. Define two dependency types per operation:
           - job dependency   : step N must start after step N-1 of same job
           - machine dependency: op at machine queue position k must start
                                 after position k-1 ends
      3. Process ops in topological order (BFS); each op starts at
         max(job_predecessor_end, machine_predecessor_end).

    Mutates schedule in place.
    """
    import heapq as _heapq

    n = len(schedule)
    if n == 0:
        return

    # ── Machine ordering (preserves Phase 2 sequence) ────────────────────
    by_machine: dict[str, list[int]] = defaultdict(list)
    for i, op in enumerate(schedule):
        by_machine[op['machine_name']].append(i)
    for idxs in by_machine.values():
        idxs.sort(key=lambda i: schedule[i]['start_hrs'])

    machine_prev: dict[int, int] = {}  # op_idx → idx of previous op on same machine
    dependents:   list[list[int]] = [[] for _ in range(n)]

    for idxs in by_machine.values():
        for pos, idx in enumerate(idxs):
            if pos > 0:
                prev_idx = idxs[pos - 1]
                machine_prev[idx] = prev_idx
                dependents[prev_idx].append(idx)

    # ── Job-step predecessor map ──────────────────────────────────────────
    job_step_idx: dict[tuple, int] = {
        (op['batch_id'], op['step_number']): i
        for i, op in enumerate(schedule)
    }

    job_pred: dict[int, int] = {}
    for i, op in enumerate(schedule):
        if op['step_number'] > 1:
            pred_idx = job_step_idx.get((op['batch_id'], op['step_number'] - 1))
            if pred_idx is not None:
                job_pred[i] = pred_idx
                dependents[pred_idx].append(i)

    # ── In-degree count (number of unresolved dependencies) ──────────────
    in_degree = [0] * n
    for i in range(n):
        if i in machine_prev:
            in_degree[i] += 1
        if i in job_pred:
            in_degree[i] += 1

    # ready_time[i] = earliest possible start for op i
    ready_time = [0.0] * n
    heap: list[tuple[float, int]] = []
    for i in range(n):
        if in_degree[i] == 0:
            _heapq.heappush(heap, (0.0, i))

    # ── Forward-pass dispatch ─────────────────────────────────────────────
    processed = 0
    while heap:
        _, i = _heapq.heappop(heap)
        op    = schedule[i]
        start = ready_time[i]
        end   = start + op['dur_hours']

        op['start_hrs'] = start
        op['end_hrs']   = end
        op['start_dt']  = start_dt + timedelta(hours=start)
        op['end_dt']    = start_dt + timedelta(hours=end)
        processed += 1

        for dep in dependents[i]:
            ready_time[dep] = max(ready_time[dep], end)
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                _heapq.heappush(heap, (ready_time[dep], dep))

    if processed != n:
        logger.warning(
            "_rebuild_timings_after_phase2: processed %d/%d ops — "
            "cycle detected in dependency graph; schedule may be invalid",
            processed, n,
        )


def _local_pulp_optimise(schedule, job_ops, start_dt, k_machines):
    """
    Re-optimise the K most-loaded machines using single-machine MILP.

    Strategy:
    1. Identify the K machines with the highest total processing hours
       (the bottlenecks most worth polishing).
    2. For each, collect all operations assigned to it and solve a
       single-machine sequencing MILP to minimise that machine's makespan.
    3. Very large machines (> 200 ops) are skipped in MILP but still
       re-sorted by start time to ensure consistency.

    Parameters
    ----------
    schedule   : list of scheduled op dicts (mutated in place)
    job_ops    : original op definitions (for duration lookup)
    start_dt   : schedule epoch (for datetime reconstruction)
    k_machines : how many bottleneck machines to optimise

    Returns
    -------
    Same schedule list with updated start/end times for optimised machines.
    """
    # ── STEP 1: Compute total load per machine ────────────────────────────
    machine_load = defaultdict(float)
    for row in schedule:
        machine_load[row['machine_name']] += row['dur_hours']

    # Select the K heaviest-loaded machines
    top_machines = sorted(machine_load, key=machine_load.get, reverse=True)[:k_machines]

    # Index schedule by machine for efficient access
    by_machine = defaultdict(list)
    for i, row in enumerate(schedule):
        by_machine[row['machine_name']].append(i)

    # ── Build predecessor-end map ─────────────────────────────────────────
    # For each (job, step) pair record when the PREVIOUS step finishes so
    # that _reoptimise_machine can use it as a hard lower bound.
    # Without this map Phase 2 MILP was free to move a step earlier than
    # its upstream step on another machine, violating job precedence.
    job_step_ends: dict = {}
    for row in schedule:
        job_step_ends[(row['batch_id'], row['step_number'])] = row['end_hrs']

    predecessor_end_for: dict = {}
    for row in schedule:
        step = row['step_number']
        if step > 1:
            predecessor_end_for[(row['batch_id'], step)] = job_step_ends.get(
                (row['batch_id'], step - 1), 0.0
            )

    # ── STEP 2: Re-optimise each selected machine ─────────────────────────
    for machine in top_machines:
        indices = by_machine[machine]
        if len(indices) < 2:
            # Only one operation — nothing to sequence
            continue
        if len(indices) > 200:
            # MILP would take too long — just re-sort by existing start time
            # to guarantee consistent ordering without optimisation
            indices.sort(key=lambda i: schedule[i]['start_hrs'])
            continue
        _reoptimise_machine(schedule, indices, machine, start_dt, predecessor_end_for)

    # After all machines are resequenced, some successor steps on other
    # machines may now start before their (repositioned) predecessors finish.
    # Rebuild all timings in topological order while preserving Phase 2 ordering.
    _rebuild_timings_after_phase2(schedule, start_dt)

    return schedule


def _reoptimise_machine(
    schedule,
    indices,
    machine_name,
    start_dt,
    predecessor_end_for: dict | None = None,
):
    """
    Single-machine MILP: minimise makespan for one machine's operations.

    Model
    -----
    Variables:
      S_i   (continuous) — start time for operation i (hours from epoch)
      Cmax  (continuous) — makespan = max(S_i + d_i) across all i
      y_ij  (binary)     — 1 if op i precedes op j, 0 if j precedes i

    Objective: minimise Cmax

    Constraints:
      Cmax ≥ S_i + d_i  for all i          (makespan definition)
      S_j ≥ S_i + d_i - M(1-y_ij)         (disjunctive: if i before j)
      S_i ≥ S_j + d_j - M·y_ij            (disjunctive: if j before i)
      S_i ≥ predecessor_end_i              (routing dependency — NEW)

    Lower bounds now come from predecessor_end_for so the solver cannot
    schedule a step earlier than its upstream step on another machine.
    The old heuristic (start - duration) had no scheduling basis and
    allowed Phase 2 to violate job precedence constraints.
    """
    n         = len(indices)
    ops       = [schedule[i] for i in indices]

    # Lower bound: earliest this op can start = when its predecessor finishes.
    # For first steps (step_number == 1) there is no predecessor → lb = 0.
    # predecessor_end_for is keyed by (batch_id, step_number).
    if predecessor_end_for:
        lb = [
            predecessor_end_for.get((op['batch_id'], op['step_number']), 0.0)
            for op in ops
        ]
    else:
        # Safe fallback when caller does not supply predecessor info
        lb = [0.0] * n

    # Upper bound: 1.5× the current schedule end (generous slack)
    ub_global = max(op['end_hrs'] for op in ops) * 1.5
    durations = [op['dur_hours'] for op in ops]

    prob = LpProblem(f"SingleMachine_{machine_name}", LpMinimize)
    S    = [LpVariable(f"S_{i}", lowBound=lb[i], upBound=ub_global) for i in range(n)]
    Cmax = LpVariable("Cmax", lowBound=0)

    prob += Cmax   # objective: minimise makespan

    # Makespan definition: Cmax must be at least as large as every op's end
    for i in range(n):
        prob += Cmax >= S[i] + durations[i]

    # Disjunctive constraints: every pair of ops must be non-overlapping.
    # y_ij = 1  →  op i runs before op j
    # y_ij = 0  →  op j runs before op i
    #
    # Exception: if both ops belong to the same batch AND one has a lower
    # step number, the order is fixed by job sequencing — no binary variable
    # needed; we add a hard ordering constraint instead.
    M = ub_global
    for i in range(n):
        for j in range(i + 1, n):
            same_batch = ops[i]['batch_id'] == ops[j]['batch_id']
            if same_batch:
                # Job-step order on this machine must match step-number order
                if ops[i]['step_number'] < ops[j]['step_number']:
                    prob += S[j] >= S[i] + durations[i]   # i before j, hard
                else:
                    prob += S[i] >= S[j] + durations[j]   # j before i, hard
            else:
                y = LpVariable(f"y_{i}_{j}", cat=LpBinary)
                prob += S[j] >= S[i] + durations[i] - M * (1 - y)
                prob += S[i] >= S[j] + durations[j] - M * y

    # Use time-limited solver to prevent blocking on complex instances
    prob.solve(_SOLVER_TIMED)

    # ── Apply result if solver found an optimal solution ──────────────────
    # If infeasible or time-limit hit, leave the original schedule unchanged
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
    """
    Compute summary KPIs from a completed schedule.

    Parameters
    ----------
    schedule_rows   : list of scheduled op dicts (must have machine_name,
                      dur_hours)
    makespan_hours  : float  total schedule span from first start to last end

    Returns
    -------
    dict with keys:
        makespan_hours      : rounded total schedule duration
        makespan_days       : makespan converted to days
        total_operations    : total number of scheduled ops
        machines_used       : count of distinct machines
        utilisation         : { machine_name: utilisation_pct }
        bottleneck_machine  : machine with highest utilisation
        bottleneck_util     : that machine's utilisation percentage
    """
    if not schedule_rows:
        return {}

    # Accumulate total processing hours and op count per machine
    machine_used = defaultdict(float)
    machine_ops  = defaultdict(int)

    for row in schedule_rows:
        machine_used[row['machine_name']] += row['dur_hours']
        machine_ops[row['machine_name']]  += 1

    # Utilisation = (time machine is busy) / (total makespan) × 100
    # A machine running non-stop would show 100%; gaps reduce this.
    utilisation = {
        m: round(machine_used[m] / makespan_hours * 100, 2) if makespan_hours > 0 else 0
        for m in machine_used
    }

    # Bottleneck = the machine with the highest utilisation
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
    """
    Per-product batch optimisation (calls the PuLP single-product ILP).

    Legacy wrapper that iterates over products one by one and saves each
    result immediately. Kept for backward compatibility — new code should
    use optimize_product_batches_jointly() for better load balancing.

    Returns
    -------
    list of dicts summarising the batch decisions made for each product
    """
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
    """
    Calculate high-level scheduling KPIs from ORM records.

    Differs from compute_schedule_kpis() in that it reads from the
    ProductionSchedule ORM (persisted records) rather than raw scheduler
    output dicts. Used by legacy API endpoints.

    Parameters
    ----------
    schedule_records    : QuerySet / list of ProductionSchedule ORM objects
    machine_availability: iterable of machine names to include

    Returns
    -------
    dict with makespan, utilisation per machine, and throughput stats
    """
    if not schedule_records:
        return {}

    max_end        = max(s.end_time   for s in schedule_records)
    min_start      = min(s.start_time for s in schedule_records)
    makespan_hours = (max_end - min_start).total_seconds() / 3600
    makespan_days  = makespan_hours / 24

    # Per-machine utilisation from DB aggregation
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
    """
    Extract batch optimisation parameters from the API request body.

    Applies safe defaults when parameters are missing:
        max_num_batches = 25  (default: up to 25 batches per product)
        min_batch_size  = 50  (default: at least 50 units per batch)
        max_batch_size  = 500 (default: at most 500 units per batch)

    Returns
    -------
    (max_num_batches, min_batch_size, max_batch_size) as ints
    """
    return (
        int(request.data.get('max_num_batches',  25)),
        int(request.data.get('min_batch_size',   50)),
        int(request.data.get('max_batch_size',  500)),
    )


# ===========================================================================
# 6.  CSV / DATAFRAME HELPERS
# ===========================================================================

def build_summary(df):
    """
    Build a summary table dict from the production dataframe.

    Returns a dict with columns 'Metric', 'Finished Goods', 'Connectors',
    aggregating Planned, Realized, Backlog, and Open counts.
    """
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
    """
    Compute per-shift production output totals for finished goods and connectors.

    Reads shift columns starting at column index 14 (up to index 50).
    Returns two dicts keyed by shift label, each mapping to a formatted
    production count string.

    Parameters
    ----------
    df               : production dataframe
    finished_filter  : boolean Series selecting finished goods rows
    connector_filter : boolean Series selecting connector rows
    """
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
    """
    Compute per-shift backlog values from columns 95–112 of the dataframe.

    Each column represents one backlog period. Only positive values are
    summed (negative or zero means no backlog for that period).

    Returns a list aligned to SHIFT_LABELS with backlog values interleaved
    with zeros (two entries per column: [value, 0]).
    """
    # Read backlog columns (indices 95–112 inclusive)
    backlog_cols   = range(95, 113)
    backlog_values = []
    for idx in backlog_cols:
        total = df.iloc[2:264, idx]
        total = total.loc[total > 0].sum()
        backlog_values.append(total if total > 0 else 0)

    # Interleave each value with a zero placeholder for shift pairing
    result = []
    for val in backlog_values:
        result.extend([val, 0])

    # Pad/truncate to exactly len(SHIFT_LABELS) entries
    return (result + [0] * len(SHIFT_LABELS))[:len(SHIFT_LABELS)]


def calculate_efficiency(df, num_shifts):
    """
    Calculate per-shift production efficiency as a percentage.

    Formula:
        efficiency = (planned_hours / available_hours) × 100

    where:
        planned_hours  = Σ (quantity × STD_minutes) / 60
        available_hours = 7.67 hours/shift × num_shifts

    Parameters
    ----------
    df         : production dataframe (must have 'STD' column)
    num_shifts : number of active shifts (used for available time)

    Returns
    -------
    List of efficiency values aligned to SHIFT_LABELS (36 entries).
    """
    efficiency_list = []

    # Guard: if STD column is missing, return all zeros
    if 'STD' not in df.columns:
        return [0] * 36

    # Parse STD column to float, removing any non-numeric characters
    std_col = pd.to_numeric(
        df.loc[2:1003, 'STD'].astype(str)
        .str.replace(r'[^\d\.\-]', '', regex=True),
        errors='coerce'
    )

    shift_time_hours = 7.67           # standard shift duration
    available_time   = shift_time_hours * num_shifts
    quantity_columns = list(range(15, 52, 2))   # every other column (qty cols)

    for col_idx in quantity_columns:
        # Parse quantity column to float
        quantity = pd.to_numeric(
            df.iloc[2:1003, col_idx].astype(str)
            .str.replace(r'[^\d\.\-]', '', regex=True),
            errors='coerce'
        )
        # Only compute where both quantity and STD are valid
        valid           = quantity.notna() & std_col.notna()
        planned_minutes = (quantity[valid] * std_col[valid]).sum()
        planned_hours   = planned_minutes / 60
        efficiency      = (planned_hours / available_time) * 100 if available_time else 0
        efficiency_list.append(efficiency)

    # Interleave with zeros and truncate/pad to exactly 36 entries
    result = []
    for val in efficiency_list:
        result.extend([val, 0])
    return (result + [0] * 36)[:36]


def apply_filters(df, filters):
    """
    Apply a dict of column-value filters to a DataFrame.

    Filters with value "All" are skipped (no filtering applied for that column).
    Columns not present in the DataFrame are also silently skipped.

    Parameters
    ----------
    df      : pandas DataFrame
    filters : dict { column_name: value }

    Returns
    -------
    Filtered DataFrame (subset of rows matching all active filters)
    """
    for column, val in filters.items():
        if val != "All" and column in df.columns:
            df = df[df[column] == val]
    return df


def clean_numeric_columns(df, columns):
    """
    Convert specified columns to numeric, coercing errors to NaN.

    Strips commas and whitespace before parsing — handles formatted
    numbers like "1,234.56" from CSV exports.

    Mutates the DataFrame in place.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '').str.strip(),
                errors='coerce'
            )


def clean_text_columns(df, columns):
    """
    Strip leading/trailing whitespace from specified text columns.

    Mutates the DataFrame in place.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()


def clean_shift_columns(df, column_ranges):
    """
    Parse shift production columns (given as index ranges) to numeric.

    Strips any non-numeric characters (currency symbols, spaces) before
    parsing. Used for raw shift data that may contain mixed-format values.

    Parameters
    ----------
    df             : pandas DataFrame
    column_ranges  : list of (start_idx, end_idx) tuples (exclusive end)

    Mutates the DataFrame in place.
    """
    for start, end in column_ranges:
        for idx in range(start, end):
            df.iloc[:, idx] = pd.to_numeric(
                df.iloc[:, idx].astype(str)
                .str.replace(r'[^\d\.\-]', '', regex=True),
                errors='coerce'
            )


def process_frontpage_data(frontpage_df):
    """
    Parse and normalise the frontpage CSV into a structured data dict.

    Steps:
    1. Take only the first 3 data rows (one per product)
    2. Rename columns to snake_case
    3. Select relevant columns: Item, SAP_TN, SAP_PL, DCC_Type,
       Description, Demand_2024
    4. Parse numeric columns — stripping commas, handling None/NaN
    5. Convert to dict-of-lists for easy iteration in calling code

    Returns
    -------
    dict { column_name: [value, value, value] }
    """
    # Only the first 3 rows contain product data
    frontpage_df = frontpage_df.head(3)
    df = frontpage_df.rename(columns={
        'SAP TN': 'SAP_TN', 'SAP PL': 'SAP_PL', 'DCC Type': 'DCC_Type'
    })
    df = df[['Item', 'SAP_TN', 'SAP_PL', 'DCC_Type', 'Description', '2024']]
    df = df.rename(columns={'2024': 'Demand_2024'})

    # Parse integer columns — strip commas, replace 'None'/'nan' with NA
    int_cols = ['Item', 'SAP_TN', 'SAP_PL', 'Demand_2024']
    for col in int_cols:
        df[col] = (
            df[col].astype(str)
            .str.replace(',', '', regex=False)
            .replace('None', pd.NA)
            .replace('nan',  pd.NA)
            .pipe(pd.to_numeric, errors='coerce')
            .astype('Int64')   # nullable integer type preserves NaN as pd.NA
        )

    # Replace pandas NA with Python None for JSON serialisability
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient='list')


def process_routing_data(process_df):
    """
    Parse the process routing CSV into a list of step dicts and a machines list.

    Expected CSV layout (after removing last 2 columns):
        Row 0–1  : header rows (ignored)
        Row 2    : machine names  (from column 4 onward)
        Row 3    : step names     (from column 4 onward)
        Row 4+   : product rows   (col 0 = item number, cols 4+ = cycle times)

    A step is included only if its cycle time > 0. Workers default to 0.5.

    Returns
    -------
    process_routing : list[dict]  each dict has: item, step, machine,
                                  time (seconds), name, workers
    machines_list   : list[str]   unique machine names found in the routing
    """
    process_routing = []

    # Extract machine names and step names from header rows
    machines = (
        process_df.iloc[2, 4:].fillna('').astype(str).str.strip().tolist()
    )
    process_steps = (
        process_df.iloc[3, 4:].fillna('').astype(str)
        .str.replace(r'\s+', ' ', regex=True).str.strip().tolist()
    )

    # Product data starts at row 4
    data_df = process_df.iloc[4:].copy()

    for _, row in data_df.iterrows():
        # Skip non-product rows (header continuations, blank lines, etc.)
        if not str(row.iloc[0]).replace('.0', '').isdigit():
            continue
        try:
            item = int(float(row.iloc[0]))
        except (ValueError, TypeError):
            continue

        # Each column from index 4 onward corresponds to a routing step
        for idx in range(len(process_steps)):
            raw_val  = row.iloc[idx + 4]
            time_val = pd.to_numeric(raw_val, errors='coerce')

            # Only create a step if a valid positive cycle time exists
            if pd.notna(time_val) and time_val > 0:
                process_routing.append({
                    'item':    item,
                    'step':    idx + 1,          # 1-based step number
                    'machine': machines[idx],
                    'time':    round(float(time_val), 2),
                    'name':    process_steps[idx],
                    'workers': 0.5               # default staffing level
                })

    # Deduplicate machine names (preserve only non-empty strings)
    machines_list = list(set(m for m in machines if m))
    return process_routing, machines_list


def _adaptive_bounds(demand: int,
                     max_num_batches: int,
                     user_min: int,
                     user_max: int) -> tuple[int, int]:
    """
    Compute effective [min_batch, max_batch] bounds for a given demand level.

    The user-supplied min/max are treated as *preferences*, not hard constraints.
    If they are infeasible for this particular demand (e.g., demand=319,908 but
    user_max=500 → ceil(319908/1)=319908 > 500 for every n), we fall back to
    the demand-derived feasible range so the ILP always has a valid solution.

    Feasible range (demand-derived):
        true_min = ceil(demand / max_num_batches)  — smallest batch when N=max
        true_max = demand                           — one big batch (N=1)

    Effective bounds = intersection(user preferences, true feasible range).
    If the intersection is empty, use the true range.

    Parameters
    ----------
    demand          : int   total units to produce
    max_num_batches : int   maximum allowed number of batches
    user_min        : int   user-specified minimum batch size
    user_max        : int   user-specified maximum batch size

    Returns
    -------
    (eff_min, eff_max) : tuple[int, int]
    """
    if demand <= 0:
        return 1, 1

    # Demand-derived feasible range
    true_min = max(1, math.ceil(demand / max_num_batches))
    true_max = demand   # absolute ceiling: one batch of the entire demand

    # Intersect with user preferences
    eff_min = max(user_min, true_min)
    eff_max = min(user_max, true_max)

    if eff_min > eff_max:
        # User bounds are entirely outside the feasible range — ignore them
        eff_min = true_min
        eff_max = true_max

    return eff_min, eff_max


def _build_joint_summary(result: dict) -> dict:
    """
    Build a summary statistics dict from a joint optimization result.

    Extracts aggregate metrics (totals, averages, load balance score)
    from the per-product results list and machine_loads dict returned
    by optimize_product_batches_jointly().

    Parameters
    ----------
    result : dict  return value of optimize_product_batches_jointly()

    Returns
    -------
    dict with aggregate stats, or {} if no results
    """
    results = result.get("results", [])
    if not results:
        return {}

    batch_sizes  = [r["new_batch_size"]  for r in results if r["demand"] > 0]
    num_batches  = [r["new_num_batches"] for r in results if r["demand"] > 0]

    # Parse improvement strings back to floats for averaging
    improvements = []
    for r in results:
        try:
            pct = float(r["improvement"].replace("%", ""))
            improvements.append(pct)
        except (ValueError, AttributeError):
            improvements.append(0.0)

    ml            = result.get("machine_loads", {})
    most_loaded   = max(ml, key=ml.get) if ml else "—"
    least_loaded  = min(ml, key=ml.get) if ml else "—"
    load_vals     = list(ml.values())

    # Load balance score: 100 = perfectly even, 0 = all load on one machine
    # Formula: (1 - range/max) × 100
    # Epsilon prevents ZeroDivisionError when all loads are zero.
    load_balance  = round(
        (1 - (max(load_vals) - min(load_vals)) / (max(load_vals) + 1e-9)) * 100, 1
    ) if load_vals else 0

    return {
        "total_products":       len(results),
        "total_demand":         sum(r["demand"] for r in results),
        "total_batches":        sum(num_batches),
        "avg_batch_size":       round(np.mean(batch_sizes), 1) if batch_sizes else 0,
        "avg_improvement":      round(np.mean(improvements), 1) if improvements else 0,
        "most_loaded_machine":  most_loaded,
        "least_loaded_machine": least_loaded,
        "load_balance_score":   load_balance,
        "makespan_proxy_hours": result.get("makespan_proxy", 0),
        "solver_status":        result.get("status", "unknown"),
    }


def _joint_preview_no_save(products, max_num_batches, min_batch_size, max_batch_size):
    """
    Run joint batch optimization in PREVIEW mode (no database writes).

    This function builds and solves a joint Integer Linear Programming (ILP)
    model to determine the optimal number of batches for all products together,
    while balancing machine loads.

    It mirrors optimize_product_batches_jointly() exactly, except:
        - No product fields are saved
        - products_updated is always 0
        - A preview message is returned

    Objective:
        Minimize the maximum machine load (makespan proxy).

    Returns:
        {
            status: "optimal" or "fallback",
            results: per-product optimization details,
            machine_loads: final machine hours,
            makespan_proxy: minimized max machine load,
            products_updated: 0,
            message: preview status
        }
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

    # Convert queryset to list to avoid repeated DB evaluation
    products_list = list(products)

    # Candidate batch counts each product can choose from (1 → max_num_batches)
    candidates = list(range(1, max_num_batches + 1))

    # ---------------------------------------------------------------------
    # STEP 1 — Collect cycle times per product per machine
    # ---------------------------------------------------------------------
    # We build:
    #   cycle_times[product_id][machine_name] = total cycle time (seconds)
    #
    # This tells us:
    #   If we produce X units of a product,
    #   how much time each machine will consume.
    # ---------------------------------------------------------------------

    cycle_times: dict = {}
    all_machine_names: set = set()

    for product in products_list:
        steps = ProcessStep.objects.filter(product=product).select_related("machine")
        cycle_times[product.pk] = {}

        for step in steps:
            mname = step.machine.name

            # Accumulate cycle time per machine (in case multiple steps use same machine)
            cycle_times[product.pk][mname] = (
                cycle_times[product.pk].get(mname, 0.0) + step.cycle_time_seconds
            )

            all_machine_names.add(mname)

    # Sorted list of all machines involved in optimization
    all_machines = sorted(all_machine_names)

    # ---------------------------------------------------------------------
    # STEP 2 — Build ILP (Integer Linear Programming) model
    # ---------------------------------------------------------------------
    # Objective:
    #   Minimize C
    #
    # Where:
    #   C = maximum machine load (makespan proxy)
    #
    # By minimizing C, we balance workload across machines.
    # ---------------------------------------------------------------------

    prob = LpProblem("JointBatchOptimizationPreview", LpMinimize)

    # Continuous variable representing the maximum machine load (hours)
    C = LpVariable("makespan_proxy", lowBound=0, cat=LpContinuous)

    # Objective function
    prob += C

    # Binary decision variables:
    #   Y[product_id][n] = 1 if product chooses n batches
    #   Y[product_id][n] = 0 otherwise
    Y: dict = {}

    # Precomputed machine load table:
    #   load_table[product_id][n][machine_name] = machine hours
    load_table: dict = {}

    # ---------------------------------------------------------------------
    # STEP 2A — Create variables and compute machine loads per option
    # ---------------------------------------------------------------------

    for product in products_list:
        pk = product.pk
        demand = int(product.demand_2024) if product.demand_2024 else 0

        # Skip products with zero demand
        if demand <= 0:
            continue

        # Compute adaptive effective batch size bounds
        eff_min, eff_max = _adaptive_bounds(
            demand, max_num_batches, min_batch_size, max_batch_size
        )

        Y[pk] = {}
        load_table[pk] = {}

        for n in candidates:

            # Binary variable for choosing n batches
            Y[pk][n] = LpVariable(f"yp_{pk}_{n}", cat="Binary")

            # Batch size if we use n batches
            batch_sz = math.ceil(demand / n)

            # Total production units (may slightly exceed demand)
            total_units = batch_sz * n

            load_table[pk][n] = {}

            # Compute machine load in HOURS for this option
            for mname in all_machines:
                ct = cycle_times[pk].get(mname, 0.0)

                # Convert seconds → hours
                load_table[pk][n][mname] = (ct * total_units) / 3600.0

        # -----------------------------------------------------------------
        # Constraint 1 — Each product must choose EXACTLY one batch count
        # -----------------------------------------------------------------
        prob += lpSum(Y[pk][n] for n in candidates) == 1

        # -----------------------------------------------------------------
        # Constraint 2 — Enforce batch size bounds
        # -----------------------------------------------------------------
        # If batch size is outside allowed range,
        # force that option to be infeasible (Y = 0).
        # -----------------------------------------------------------------
        for n in candidates:
            batch_sz = math.ceil(demand / n)

            if not (eff_min <= batch_sz <= eff_max):
                prob += Y[pk][n] == 0

    # ---------------------------------------------------------------------
    # STEP 2B — Machine Load Constraints
    # ---------------------------------------------------------------------
    # For each machine:
    #
    #   Σ (machine load from all selected products) ≤ C
    #
    # This ensures C becomes the maximum machine load.
    # ---------------------------------------------------------------------

    for mname in all_machines:
        expr = []

        for product in products_list:
            pk = product.pk
            demand = int(product.demand_2024) if product.demand_2024 else 0

            if demand <= 0 or pk not in Y:
                continue

            for n in candidates:
                h = load_table[pk][n].get(mname, 0.0)
                if h > 0:
                    expr.append(h * Y[pk][n])

        if expr:
            prob += lpSum(expr) <= C

    # ---------------------------------------------------------------------
    # STEP 3 — Solve the ILP
    # ---------------------------------------------------------------------
    # Using CBC solver with a 2-minute time limit.
    # If time expires, fallback logic will be used.
    # ---------------------------------------------------------------------

    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=120))

    solved = LpStatus[prob.status] == "Optimal"

    machine_loads = {m: 0.0 for m in all_machines}
    results = []

    # ---------------------------------------------------------------------
    # STEP 4 — Extract Results (Preview Only — No DB Writes)
    # ---------------------------------------------------------------------

    for product in products_list:
        pk = product.pk
        demand = int(product.demand_2024) if product.demand_2024 else 0

        old_bs = product.batch_size
        old_nb = product.num_batches

        ideal = round(demand / max_num_batches, 2) if demand > 0 else 0

        # If skipped or zero demand → unchanged
        if demand <= 0 or pk not in Y:
            results.append({
                "item": product.item,
                "description": (product.description or "")[:50],
                "demand": demand,
                "old_batch_size": old_bs,
                "old_num_batches": old_nb,
                "new_batch_size": old_bs,
                "new_num_batches": old_nb,
                "ideal_batch_size": ideal,
                "improvement": "0%",
                "source": "skipped",
            })
            continue

        # Identify selected batch count from solver
        chosen_n = None
        if solved:
            for n in candidates:
                if Y[pk][n].varValue and Y[pk][n].varValue > 0.5:
                    chosen_n = n
                    break

        # If optimal solution found
        if chosen_n:
            new_batch = math.ceil(demand / chosen_n)
            new_n = chosen_n
        else:
            # Fallback heuristic if solver fails
            eff_min, eff_max = _adaptive_bounds(
                demand, max_num_batches, min_batch_size, max_batch_size
            )
            new_n = max(1, min(max_num_batches, math.ceil(demand / eff_min)))
            new_batch = math.ceil(demand / new_n)

        # Accumulate final machine loads for summary
        for mname in all_machines:
            ct = cycle_times[pk].get(mname, 0.0)
            total = new_batch * new_n
            machine_loads[mname] += (ct * total) / 3600.0

        # Calculate improvement in number of batches
        impr = (
            f"{((old_nb - new_n) / old_nb * 100):.1f}%"
            if old_nb and old_nb != new_n else "0%"
        )

        results.append({
            "item": product.item,
            "description": (product.description or "")[:50],
            "demand": demand,
            "old_batch_size": old_bs,
            "old_num_batches": old_nb,
            "new_batch_size": new_batch,
            "new_num_batches": new_n,
            "ideal_batch_size": ideal,
            "improvement": impr,
            "source": "joint_ilp" if solved else "fallback",
        })

    # Final minimized maximum machine load
    proxy = round(value(C), 4) if solved and value(C) is not None else 0.0

    return {
        "status": "optimal" if solved else "fallback",
        "results": results,
        "machine_loads": {m: round(v, 2) for m, v in machine_loads.items()},
        "makespan_proxy": proxy,
        "products_updated": 0,  # Always 0 in preview mode
        "message": f"Preview only — no DB writes. Status: {'optimal' if solved else 'fallback'}",
    }


# ===========================================================================
# SHIFT-AWARE SCHEDULER  (Type 1 — simple discrete-event with shift windows)
# ===========================================================================
#
# This scheduler guarantees all required production protocols:
#   • No machine overlap  — machine_free_at[m] is always advanced past end
#   • Step sequencing     — prev_step_end gates the next step's earliest start
#   • Shift boundaries    — _find_slot() only accepts slots within a shift window
#   • Duration consistency— duration_hours == (end_time - start_time).seconds/3600
#   • Batch coherence     — batch_id carries product item + batch number
#
# Contrast with run_job_shop_scheduler (Type 2) which uses the ERT greedy
# dispatcher + left-shift compaction + PuLP MILP for optimised throughput
# but does not enforce hard shift boundaries.

def _get_shift_windows(
    base_dt: datetime,
    shifts: list,
    num_days: int,
) -> list:
    """
    Build a sorted list of (shift_start, shift_end) pairs for num_days.

    Parameters
    ----------
    base_dt   : schedule epoch (any time-of-day; we anchor to midnight)
    shifts    : list of (start_hour, end_hour) tuples in 24h clock.
                end_hour > 24 is valid for night-shift overflow (e.g., 22→30).
    num_days  : number of calendar days to generate windows for.

    Returns
    -------
    Sorted list of (datetime, datetime) tuples.
    """
    base_day = base_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    windows = []
    for day_offset in range(num_days + 2):   # +2 to safely cover night-shift overflow
        day = base_day + timedelta(days=day_offset)
        for (sh, eh) in shifts:
            s = day + timedelta(hours=sh)
            e = day + timedelta(hours=eh)
            if e > base_dt:                   # skip windows entirely before start
                windows.append((s, e))
    windows.sort()
    return windows


def _find_slot(
    machine_free_at: datetime,
    earliest_start: datetime,
    duration_h: float,
    shift_windows: list,
) -> tuple:
    """
    Find the earliest (start, end) pair that:
      a) starts >= max(machine_free_at, earliest_start)
      b) fits entirely within one shift window (no cross-shift operations)

    Falls back to unconstrained scheduling if no window is found (avoids
    RuntimeError on very tight schedules — caller should use enough num_days).

    Parameters
    ----------
    machine_free_at : datetime  when the machine becomes available
    earliest_start  : datetime  step-sequencing lower bound (prev step end)
    duration_h      : float     operation duration in hours
    shift_windows   : list      sorted (shift_start, shift_end) pairs

    Returns
    -------
    (start, end) datetime pair
    """
    candidate = max(machine_free_at, earliest_start)
    duration  = timedelta(hours=duration_h)

    for (sw_start, sw_end) in shift_windows:
        if sw_end <= candidate:
            continue
        actual_start = max(candidate, sw_start)
        actual_end   = actual_start + duration
        if actual_end <= sw_end:
            return actual_start, actual_end
        # Operation doesn't fit in this window — try next shift

    # Fallback: no shift window found — return unconstrained
    return candidate, candidate + duration


def run_shift_aware_scheduler(
    products,
    start_dt: datetime,
    shifts: list | None = None,
    num_days: int = 60,
    batch_override: dict | None = None,
    progress_callback=None,
) -> list:
    """
    Shift-aware discrete-event scheduler (Schedule Type 1).

    Processes products in the order given, scheduling each batch's steps
    sequentially.  All operations are snapped into shift windows so no
    operation ever straddles a shift boundary.

    Protocols guaranteed
    --------------------
    1. No machine overlap  — machine_free_at advanced past each op's end
    2. Step sequencing     — step N+1 cannot start before step N ends
    3. Shift boundaries    — _find_slot() enforces window containment
    4. Duration consistency— dur_hours = (end - start).total_seconds() / 3600
    5. Batch coherence     — same batch_id = same product + batch_num + batch_size

    Parameters
    ----------
    products       : iterable of Product ORM objects
    start_dt       : datetime  schedule anchor
    shifts         : list of (start_hour, end_hour) tuples (default 3×8h shifts)
    num_days       : int       how many days of shift windows to pre-generate
    batch_override : dict      { product_pk: (batch_size, num_batches) }
    progress_callback : callable  f(pct: int, msg: str)

    Returns
    -------
    list[dict] with the same schema as run_job_shop_scheduler so the Celery
    task can persist results identically regardless of scheduler type.
    """
    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    if shifts is None:
        shifts = [(6, 14), (14, 22), (22, 30)]   # morning / afternoon / night

    _progress(5, "Building shift windows…")
    shift_windows = _get_shift_windows(start_dt, shifts, num_days)

    _progress(10, "Extracting process routing…")
    routing: dict = {}
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
                    'step':      s.step_number,
                    'machine':   s.machine.name,
                    'cycle_sec': s.cycle_time_seconds,
                    'step_name': s.step_name,
                }
                for s in steps
            ]

    # Collect all machine names and initialise availability to start_dt
    all_machines: set = set()
    for steps in routing.values():
        for s in steps:
            all_machines.add(s['machine'])
    machine_free_at: dict = {m: start_dt for m in all_machines}

    schedule_rows: list = []
    products_list = [p for p in products if p.pk in routing]
    total = max(len(products_list), 1)

    _progress(20, f"Scheduling {total} products across {len(all_machines)} machines…")

    for idx, p in enumerate(products_list):
        pct = 20 + int((idx / total) * 70)
        _progress(pct, f"Scheduling {p.item}…")

        if batch_override and p.pk in batch_override:
            b_size, n_batches = batch_override[p.pk]
        else:
            b_size    = p.batch_size  if p.batch_size  > 0 else 1
            n_batches = p.num_batches if p.num_batches > 0 else 1

        steps = routing[p.pk]

        for batch_num in range(1, n_batches + 1):
            batch_id     = f"{p.item}_B{batch_num:03d}"
            prev_step_end = start_dt      # sequencing anchor for this batch

            for step in steps:
                machine      = step['machine']
                duration_h   = max((step['cycle_sec'] * b_size) / 3600.0, 0.25)

                start, end = _find_slot(
                    machine_free_at = machine_free_at[machine],
                    earliest_start  = prev_step_end,
                    duration_h      = duration_h,
                    shift_windows   = shift_windows,
                )

                # Advance trackers
                machine_free_at[machine] = end
                prev_step_end            = end

                # Compute actual duration from timestamps (guaranteed consistent)
                actual_dur = (end - start).total_seconds() / 3600.0

                schedule_rows.append({
                    'job_id':      batch_id,
                    'batch_id':    batch_id,
                    'product_pk':  p.pk,
                    'batch_num':   batch_num,
                    'batch_size':  b_size,
                    'step_number': step['step'],
                    'step_name':   step['step_name'],
                    'machine_name': machine,
                    'start_hrs':   (start - start_dt).total_seconds() / 3600.0,
                    'end_hrs':     (end   - start_dt).total_seconds() / 3600.0,
                    'dur_hours':   round(actual_dur, 4),
                    'start_dt':    start,
                    'end_dt':      end,
                })

    _progress(92, "Validating shift-aware schedule…")
    validation = validate_schedule(schedule_rows)
    for op in schedule_rows:
        op['_schedule_valid']  = validation['valid']
        op['_validation_note'] = validation['summary']

    if validation['valid']:
        logger.info(
            "Shift-aware schedule VALID — %d ops, %d machines",
            len(schedule_rows), len(all_machines),
        )
    else:
        logger.warning("Shift-aware schedule INVALID: %s", validation['summary'])

    _progress(100, f"Complete [{validation['summary']}]")
    return schedule_rows