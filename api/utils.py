import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from django.db.models import Sum
from .models import Product, ProcessStep, ProductionSchedule

from pulp import (
    LpProblem, LpMinimize, LpMaximize, LpVariable, LpInteger, LpContinuous,
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


# ===========================================================================
# 1.  BATCH-SIZE OPTIMISATION  (single-product ILP)
# ===========================================================================
# Previous approach: simple ceil-division heuristic with manual clamping.
# PuLP model: choose (batch_size, num_batches) that covers demand exactly,
# stays inside [min_batch_size, max_batch_size], uses ≤ max_num_batches,
# and MINIMISES num_batches (fewer set-ups → less downtime).

def calculate_optimal_batch_size(demand, max_num_batches=25, min_batch_size=50, max_batch_size=500):
    """
    Solve a single-product batch-size ILP with PuLP.

    Decision variables
    ------------------
    B  (integer) – size of every batch
    N  (integer) – number of batches produced

    Objective
    ---------
    Minimise N  (fewer batches = fewer machine set-ups)

    Constraints
    -----------
    B × N  ≥  demand          (full demand coverage)
    min_batch_size  ≤  B  ≤  max_batch_size
    1  ≤  N  ≤  max_num_batches

    Returns
    -------
    tuple: (batch_size, num_batches, ideal_batch_size_float)
    """
    if demand <= 0:
        return 0, 0, 0.0

    ideal_batch_size = demand / max_num_batches          # kept for reporting

    # ------------------------------------------------------------------
    # Build the ILP
    # ------------------------------------------------------------------
    prob = LpProblem("BatchSizeOptimization", LpMinimize)

    B = LpVariable("batch_size", lowBound=min_batch_size, upBound=max_batch_size, cat=LpInteger)
    N = LpVariable("num_batches", lowBound=1,            upBound=max_num_batches, cat=LpInteger)

    # Objective: minimise number of batches
    prob += N

    # Coverage: B * N >= demand
    # PuLP cannot handle a product of two decision variables directly (that
    # would be quadratic).  We linearise by noting that for every candidate
    # value of N in [1, max_num_batches] the required B is ceil(demand/N).
    # We therefore introduce a binary selector y_n for each possible N and
    # rewrite the problem as a selection model.
    # ------------------------------------------------------------------
    # Reformulation as a binary-selection (big-M) model
    # ------------------------------------------------------------------
    candidates = list(range(1, max_num_batches + 1))               # possible N values
    y = {n: LpVariable(f"y_{n}", cat="Binary") for n in candidates}

    # Exactly one candidate must be chosen
    prob += lpSum(y[n] for n in candidates) == 1

    # Link N to the chosen candidate
    prob += N == lpSum(n * y[n] for n in candidates)

    # Link B: for the chosen candidate n, B must be ≥ ceil(demand / n)
    # Using big-M: B ≥ ceil(demand/n) - M*(1 - y_n)   for every n
    M = max_batch_size                                  # safe big-M upper bound
    for n in candidates:
        required_B = int(np.ceil(demand / n))
        prob += B >= required_B - M * (1 - y[n])

    # Upper-bound B to max_batch_size is already in the variable definition.
    # If ceil(demand / n) > max_batch_size for a candidate n, that candidate
    # is naturally infeasible (B can't satisfy the lower bound).  We can
    # tighten by explicitly forbidding it:
    for n in candidates:
        if int(np.ceil(demand / n)) > max_batch_size:
            prob += y[n] == 0

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    prob.solve(_SOLVER)

    if LpStatus[prob.status] == "Optimal":
        return int(value(B)), int(value(N)), round(ideal_batch_size, 2)

    # ── Fallback (should never fire for valid inputs) ──────────────────
    fallback_N = max(1, min(max_num_batches, int(np.ceil(demand / min_batch_size))))
    fallback_B = int(np.ceil(demand / fallback_N))
    return fallback_B, fallback_N, round(ideal_batch_size, 2)


# ===========================================================================
# 2.  JOINT MULTI-PRODUCT BATCH OPTIMISATION  (new)
# ===========================================================================
# When many products share the same machines, optimising each product in
# isolation can create unbalanced machine loads.  This model minimises the
# MAXIMUM total processing hours across all machines (i.e. the makespan
# bottleneck) while respecting per-product batch constraints.
#
# Decision variables
# ------------------
# B_i  (integer) – batch size for product i
# N_i  (integer) – number of batches for product i
# C     (continuous) – the makespan proxy (max machine load)
#
# Constraints per product i
# -------------------------
# B_i * N_i >= demand_i               (linearised via binary selectors, same
#                                       technique as calculate_optimal_batch_size)
# min_batch ≤ B_i ≤ max_batch
# 1        ≤ N_i ≤ max_num_batches
#
# Per-machine load constraint
# ---------------------------
# For each machine m:   Σ (cycle_time_{i,m} * B_i * N_i)  ≤  C
# (again linearised using the binary selectors already present)
#
# Objective: minimise C

def optimize_product_batches_jointly(products, max_num_batches, min_batch_size, max_batch_size):
    """
    Multi-product joint batch optimisation via PuLP.

    Parameters
    ----------
    products  : QuerySet[Product]  – products with demand > 0
    max_num_batches, min_batch_size, max_batch_size – global bounds

    Returns
    -------
    list[dict]  – one log entry per product (same shape as the old
                  optimize_product_batches return value)
    """
    # ── Pre-fetch process steps grouped by product & machine ──────────
    # structure: { product.pk: { machine_name: cycle_time_seconds } }
    step_map = {}
    all_machines = set()
    for product in products:
        steps = ProcessStep.objects.filter(product=product, cycle_time_seconds__gt=0)
        step_map[product.pk] = {}
        for s in steps:
            step_map[product.pk][s.machine.name] = s.cycle_time_seconds
            all_machines.add(s.machine.name)

    product_list = list(products)                        # materialise QuerySet once

    # ── Build the ILP ─────────────────────────────────────────────────
    prob = LpProblem("JointBatchOptimization", LpMinimize)

    # Upper bound on total processing hours used as big-M / C bound
    # Worst case: every product at max_batch_size * max_num_batches
    total_max_hours = sum(
        max_batch_size * max_num_batches * sum(step_map.get(p.pk, {}).values()) / 3600
        for p in product_list
    )
    C = LpVariable("makespan_proxy", lowBound=0, upBound=total_max_hours, cat=LpContinuous)

    prob += C                                            # objective: minimise C

    # ── Per-product variables & demand-coverage constraints ──────────
    # For each product we use the same binary-selector trick as above.
    candidates = list(range(1, max_num_batches + 1))

    B   = {}          # B[pk]  – batch size
    N   = {}          # N[pk]  – num batches
    Y   = {}          # Y[pk]  – dict of binary selectors  { n: var }
    # We also store, for each product, the *effective demand* contributed
    # to each machine per candidate: demand_i / n_i is already rounded up
    # to B_i = ceil(demand_i / n_i), so total units = B_i * n_i.
    # We keep a helper: for candidate n, total_units = ceil(demand / n) * n

    for p in product_list:
        pk   = p.pk
        dem  = p.demand_2024
        tag  = f"p{pk}"

        B[pk] = LpVariable(f"B_{tag}", lowBound=min_batch_size, upBound=max_batch_size, cat=LpInteger)
        N[pk] = LpVariable(f"N_{tag}", lowBound=1,              upBound=max_num_batches, cat=LpInteger)

        Y[pk] = {}
        for n in candidates:
            Y[pk][n] = LpVariable(f"y_{tag}_{n}", cat="Binary")

        # Exactly one candidate
        prob += lpSum(Y[pk][n] for n in candidates) == 1

        # Link N
        prob += N[pk] == lpSum(n * Y[pk][n] for n in candidates)

        # Link B (big-M lower bounds)
        M = max_batch_size
        for n in candidates:
            req = int(np.ceil(dem / n))
            prob += B[pk] >= req - M * (1 - Y[pk][n])
            if req > max_batch_size:
                prob += Y[pk][n] == 0          # infeasible candidate

    # ── Machine-load ≤ C  (linearised) ────────────────────────────────
    # For machine m: total_hours_m = Σ_i  cycle_time_i_m * B_i * N_i / 3600
    # B_i * N_i is NOT a single variable, but when candidate n is active:
    #   B_i = ceil(demand_i / n),  N_i = n   →  B_i * N_i = ceil(demand_i/n)*n
    # So we substitute using the binary selectors:
    #   B_i * N_i  ≈  Σ_n  [ ceil(demand_i/n) * n ] * y_{i,n}

    for m in all_machines:
        machine_load = []
        for p in product_list:
            pk        = p.pk
            dem       = p.demand_2024
            cyc_time  = step_map.get(pk, {}).get(m, 0)
            if cyc_time == 0:
                continue
            for n in candidates:
                total_units = int(np.ceil(dem / n)) * n
                hours       = cyc_time * total_units / 3600
                machine_load.append(hours * Y[pk][n])

        if machine_load:
            prob += lpSum(machine_load) <= C

    # ── Solve ─────────────────────────────────────────────────────────
    prob.solve(_SOLVER)

    # ── Extract results ──────────────────────────────────────────────
    log = []
    if LpStatus[prob.status] == "Optimal":
        for p in product_list:
            pk          = p.pk
            batch_size  = int(value(B[pk]))
            num_batches = int(value(N[pk]))

            p.batch_size  = batch_size
            p.num_batches = num_batches
            p.save()

            log.append({
                'item':             p.item,
                'demand':           p.demand_2024,
                'batch_size':       batch_size,
                'num_batches':      num_batches,
                'ideal_batch_size': round(p.demand_2024 / max_num_batches, 2)
            })
    else:
        # Fallback: solve each product independently
        log = optimize_product_batches(products, max_num_batches, min_batch_size, max_batch_size)

    return log


# ===========================================================================
# 3.  JOB-SHOP SCHEDULING  (PuLP makespan minimisation)
# ===========================================================================
# The old greedy scheduler assigned start times sequentially in product
# order, which can leave machines idle while a later product could fill gaps.
#
# PuLP model (classic job-shop with big-M disjunctive constraints):
#
# Decision variables
# ------------------
# S_{i,b,k}   (continuous) – start time (hours from epoch) of job (product i,
#                              batch b, step k)
# C_max       (continuous) – makespan (objective)
# Z_{i,b,k, j,c,l}  (binary) – 1 if operation (i,b,k) runs BEFORE (j,c,l)
#                                on the same machine
#
# Constraints
# -----------
# Precedence:  S_{i,b,k} ≥ S_{i,b,k-1} + duration_{i,b,k-1}
# No-overlap:  for every pair sharing a machine, exactly one ordering holds
#              S_{i,b,k} ≥ S_{j,c,l} + dur_{j,c,l} - M*(1-Z)    if Z=1
#              S_{j,c,l} ≥ S_{i,b,k} + dur_{i,b,k} - M*(1-Z')   if Z'=1-Z
# Makespan:    C_max ≥ S_{i,b,k} + duration_{i,b,k}   for all ops
#
# Objective: minimise C_max

def generate_production_schedule(products, use_pulp=None, time_limit_seconds=60):
    """
    Hybrid job-shop scheduler with intelligent PuLP usage.

    Strategy
    --------
    - For SMALL problems (<100 operations): Use PuLP job-shop optimizer
    - For LARGE problems (≥100 operations): Use improved greedy heuristic
    - User can force PuLP with use_pulp=True (not recommended for large problems)

    Parameters
    ----------
    products : QuerySet[Product]
    use_pulp : bool or None
        - None (default): Auto-decide based on problem size
        - True: Force PuLP (may be slow)
        - False: Force greedy
    time_limit_seconds : int
        Maximum time for PuLP solver (default 60 seconds)

    Returns
    -------
    tuple:
        - list[ProductionSchedule]  – ORM records (created in DB)
        - dict                      – machine_name → final availability datetime
    """
    # ── Collect all operations ────────────────────────────────────────
    operations = []
    machine_ops = {}

    for product in products:
        steps = ProcessStep.objects.filter(
            product=product, cycle_time_seconds__gt=0
        ).order_by('step_number')

        for batch_num in range(1, product.num_batches + 1):
            prev_step_number = None
            for step in steps:
                dur_hours = (step.cycle_time_seconds * product.batch_size) / 3600
                op = {
                    'idx':              len(operations),
                    'product':          product,
                    'batch_num':        batch_num,
                    'step':             step,
                    'step_number':      step.step_number,
                    'machine_name':     step.machine.name,
                    'machine':          step.machine,
                    'duration_hours':   dur_hours,
                    'batch_id':         f"Item{product.item}_B{batch_num}",
                    'prev_op_idx':      None,
                }
                if prev_step_number is not None:
                    op['prev_op_idx'] = len(operations) - 1

                operations.append(op)
                prev_step_number = step.step_number
                machine_ops.setdefault(step.machine.name, []).append(op['idx'])

    if not operations:
        return [], {}

    # ── Decide whether to use PuLP ────────────────────────────────────
    num_operations = len(operations)
    num_binary_vars = sum(
        len(ops) * (len(ops) - 1) // 2 
        for ops in machine_ops.values()
    )
    
    # Auto-decide: use PuLP only for small problems
    if use_pulp is None:
        use_pulp = (num_operations <= 100 and num_binary_vars <= 500)
    
    print(f"📊 Schedule size: {num_operations} operations, {num_binary_vars} disjunctive pairs")
    print(f"🔧 Using: {'PuLP optimizer' if use_pulp else 'Improved greedy heuristic'}")

    start_date = datetime.now()

    # ── Path 1: PuLP job-shop (for small problems) ───────────────────
    # if use_pulp:
    schedule_records, machine_availability = _pulp_job_shop_schedule(
        operations, machine_ops, start_date, time_limit_seconds
    )
    if schedule_records:  # Success
        return schedule_records, machine_availability
        # Fall through to greedy if PuLP failed

    # ── Path 2: Improved greedy heuristic ─────────────────────────────
    # return _improved_greedy_schedule(operations, machine_ops, start_date)


def _pulp_job_shop_schedule(operations, machine_ops, start_date, time_limit):
    """PuLP job-shop optimizer with timeout."""
    horizon = sum(op['duration_hours'] for op in operations)
    prob = LpProblem("JobShopScheduling", LpMinimize)

    S = {op['idx']: LpVariable(f"S_{op['idx']}", lowBound=0, upBound=horizon, cat=LpContinuous)
         for op in operations}
    C_max = LpVariable("C_max", lowBound=0, upBound=horizon, cat=LpContinuous)
    prob += C_max

    # Precedence constraints
    for op in operations:
        if op['prev_op_idx'] is not None:
            prev = operations[op['prev_op_idx']]
            prob += S[op['idx']] >= S[prev['idx']] + prev['duration_hours']

    # Makespan bounds
    for op in operations:
        prob += C_max >= S[op['idx']] + op['duration_hours']

    # No-overlap (disjunctive) constraints
    for m_name, op_indices in machine_ops.items():
        for a in range(len(op_indices)):
            for b in range(a + 1, len(op_indices)):
                i = op_indices[a]
                j = op_indices[b]
                dur_i = operations[i]['duration_hours']
                dur_j = operations[j]['duration_hours']
                Z = LpVariable(f"Z_{i}_{j}", cat="Binary")
                prob += S[j] >= S[i] + dur_i - horizon * (1 - Z)
                prob += S[i] >= S[j] + dur_j - horizon * Z

    # Solve with timeout
    solver = PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)

    # Extract solution
    if LpStatus[prob.status] == "Optimal":
        schedule_records = []
        machine_availability = {}
        
        for op in operations:
            start_hours = value(S[op['idx']])
            start_time  = start_date + timedelta(hours=start_hours)
            end_time    = start_time + timedelta(hours=op['duration_hours'])

            machine_availability[op['machine_name']] = max(
                machine_availability.get(op['machine_name'], start_date),
                end_time
            )

            schedule_records.append(
                ProductionSchedule.objects.create(
                    machine=op['machine'],
                    product=op['product'],
                    process_step=op['step'],
                    batch_id=op['batch_id'],
                    batch_num=op['batch_num'],
                    batch_size=op['product'].batch_size,
                    start_time=start_time,
                    end_time=end_time,
                    duration_hours=round(op['duration_hours'], 4)
                )
            )
        return schedule_records, machine_availability
    
    print(f"⚠️  PuLP solver status: {LpStatus[prob.status]} - falling back to greedy")
    return [], {}


# def _improved_greedy_schedule(operations, machine_ops, start_date):
#     """
#     Improved greedy scheduler with priority-based dispatching.
    
#     Enhancement over original: 
#     - Sorts operations by 'earliest due date' (end of precedence chain)
#     - Within same due date, prioritizes shorter operations (SPT rule)
#     """
#     # Calculate priority for each operation
#     for op in operations:
#         # Find the latest step in this batch's chain
#         current = op
#         chain_length = 0
#         while current is not None:
#             chain_length += current['duration_hours']
#             next_idx = None
#             # Find if this op is a predecessor to another
#             for other in operations:
#                 if other.get('prev_op_idx') == current['idx']:
#                     next_idx = other['idx']
#                     break
#             current = operations[next_idx] if next_idx is not None else None
        
#         op['priority'] = -chain_length  # Negative so longer chains = higher priority
#         op['spt'] = op['duration_hours']  # Shortest processing time

#     # Sort by priority (longest remaining chain first), then SPT
#     sorted_ops = sorted(operations, key=lambda x: (x['priority'], x['spt']))

#     machine_availability = {}
#     batch_completion = {}
#     schedule_records = []

#     for op in sorted_ops:
#         machine_key = op['machine_name']
#         machine_ready = machine_availability.get(machine_key, start_date)

#         # Wait for previous step in batch
#         if op['prev_op_idx'] is not None:
#             prev_op = operations[op['prev_op_idx']]
#             prev_key = f"{prev_op['batch_id']}_Step{prev_op['step_number']}"
#             prev_done = batch_completion.get(prev_key, start_date)
#         else:
#             prev_done = start_date

#         start_time = max(machine_ready, prev_done)
#         end_time = start_time + timedelta(hours=op['duration_hours'])

#         machine_availability[machine_key] = end_time
#         batch_completion[f"{op['batch_id']}_Step{op['step_number']}"] = end_time

#         schedule_records.append(
#             ProductionSchedule.objects.create(
#                 machine=op['machine'],
#                 product=op['product'],
#                 process_step=op['step'],
#                 batch_id=op['batch_id'],
#                 batch_num=op['batch_num'],
#                 batch_size=op['product'].batch_size,
#                 start_time=start_time,
#                 end_time=end_time,
#                 duration_hours=round(op['duration_hours'], 4)
#             )
#         )

#     return schedule_records, machine_availability



# ===========================================================================
# 4.  BUFFER ALLOCATION OPTIMISATION  (new)
# ===========================================================================
# The old endpoint computed buffers per-machine independently with a fixed
# formula.  When a facility has a TOTAL BUFFER BUDGET (physical space or
# WIP capital), we need to decide how to distribute that budget.
#
# Model: minimise weighted risk  =  Σ_m  utilisation_m * slack_m
#   where slack_m = (required_buffer_m − allocated_buffer_m)   (≥ 0)
#
# Subject to:
#   allocated_m  ≤  required_m                    (can't allocate more than needed)
#   Σ_m  allocated_m  ≤  total_budget            (budget constraint)
#   allocated_m  ≥  0
#
# This ensures high-utilisation machines get buffer first.

def pulp_optimize_buffers(machine_buffer_data, total_budget):
    """
    Allocate a finite buffer budget across machines via PuLP LP.

    Parameters
    ----------
    machine_buffer_data : list[dict]
        Each dict must have keys: 'machine', 'required_buffer', 'utilization'
        (utilization is 0-100 float).
    total_budget : float
        Maximum total buffer units available across all machines.

    Returns
    -------
    list[dict]  – same dicts enriched with 'allocated_buffer' and 'shortfall'.
    """
    if not machine_buffer_data or total_budget <= 0:
        for m in machine_buffer_data:
            m['allocated_buffer'] = 0.0
            m['shortfall']        = m.get('required_buffer', 0)
        return machine_buffer_data

    machines = [m['machine'] for m in machine_buffer_data]

    prob = LpProblem("BufferAllocation", LpMinimize)

    # Decision variables
    alloc   = {m: LpVariable(f"alloc_{m}", lowBound=0, cat=LpContinuous) for m in machines}
    slack   = {m: LpVariable(f"slack_{m}", lowBound=0, cat=LpContinuous) for m in machines}

    # Lookup helpers
    req   = {d['machine']: d['required_buffer'] for d in machine_buffer_data}
    util  = {d['machine']: d['utilization']     for d in machine_buffer_data}

    # Objective: minimise utilisation-weighted shortfall
    prob += lpSum(util[m] * slack[m] for m in machines)

    # Constraints
    for m in machines:
        # alloc ≤ required
        prob += alloc[m] <= req[m]
        # slack = required - alloc  (shortfall)
        prob += slack[m] == req[m] - alloc[m]

    # Budget
    prob += lpSum(alloc[m] for m in machines) <= total_budget

    prob.solve(_SOLVER)

    # ── Write results back ────────────────────────────────────────────
    for d in machine_buffer_data:
        m = d['machine']
        if LpStatus[prob.status] == "Optimal":
            d['allocated_buffer'] = round(value(alloc[m]), 2)
            d['shortfall']        = round(value(slack[m]),  2)
        else:                                                # proportional fallback
            share                 = req[m] / max(sum(req.values()), 1e-9)
            d['allocated_buffer'] = round(share * total_budget, 2)
            d['shortfall']        = round(max(req[m] - d['allocated_buffer'], 0), 2)

    return machine_buffer_data


# ===========================================================================
# 5.  LEGACY HELPERS  (unchanged – used by optimize_product_batches fallback
#                       and other parts of the project)
# ===========================================================================

def optimize_product_batches(products, max_num_batches, min_batch_size, max_batch_size):
    """
    Per-product batch optimisation (calls the PuLP single-product ILP).
    Kept as the fallback when the joint model fails.
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
    Calculate high-level scheduling KPIs.  (Unchanged logic.)
    """
    if not schedule_records:
        return {}

    max_end   = max(s.end_time   for s in schedule_records)
    min_start = min(s.start_time for s in schedule_records)
    makespan_hours = (max_end - min_start).total_seconds() / 3600
    makespan_days  = makespan_hours / 24

    machine_stats = {}
    for machine_name in machine_availability:
        used_hours = ProductionSchedule.objects.filter(
            machine__name=machine_name
        ).aggregate(total=Sum('duration_hours'))['total'] or 0

        utilization = (used_hours / makespan_hours * 100) if makespan_hours > 0 else 0
        machine_stats[machine_name] = {
            'used_hours':   round(used_hours, 2),
            'utilization':  round(utilization, 2)
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
    """Build final summary metrics table."""
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
    """Calculate shift-wise production outputs."""
    fg_output   = {}
    conn_output = {}

    for i, shift in enumerate(SHIFT_LABELS):
        col_idx = 14 + i
        if col_idx < 50:
            fg   = df.iloc[:, col_idx][finished_filter].sum()
            conn = df.iloc[:, col_idx][connector_filter].sum()
        else:
            fg = conn = 0

        fg_output[shift]   = f"{fg:,.0f}" if fg > 0 else "0"
        conn_output[shift] = f"{conn:,.0f}" if conn > 0 else "0"

    return fg_output, conn_output


def calculate_backlog(df):
    """Calculate backlog per shift."""
    backlog_cols    = range(95, 113)
    backlog_values  = []

    for idx in backlog_cols:
        total = df.iloc[2:264, idx]
        total = total.loc[total > 0].sum()
        backlog_values.append(total if total > 0 else 0)

    result = []
    for val in backlog_values:
        result.extend([val, 0])

    return (result + [0] * len(SHIFT_LABELS))[:len(SHIFT_LABELS)]


def calculate_efficiency(df, num_shifts):
    """
    Calculate overall efficiency per shift.
    Efficiency = (Planned Production Time / Available Time) * 100
    """
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

        efficiency = (planned_hours / available_time) * 100 if available_time else 0
        efficiency_list.append(efficiency)

    result = []
    for val in efficiency_list:
        result.extend([val, 0])

    return (result + [0] * 36)[:36]


def apply_filters(df, filters):
    """Apply optional column filters to the dataframe."""
    for column, value in filters.items():
        if value != "All" and column in df.columns:
            df = df[df[column] == value]
    return df


def clean_numeric_columns(df, columns):
    """Convert specified columns to numeric."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '').str.strip(),
                errors='coerce'
            )


def clean_text_columns(df, columns):
    """Strip whitespace from specified text columns."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()


def clean_shift_columns(df, column_ranges):
    """Clean numeric shift-based columns using column index ranges."""
    for start, end in column_ranges:
        for idx in range(start, end):
            df.iloc[:, idx] = pd.to_numeric(
                df.iloc[:, idx]
                .astype(str)
                .str.replace(r'[^\d\.\-]', '', regex=True),
                errors='coerce'
            )


def process_frontpage_data(frontpage_df):
    """Process frontpage CSV data."""
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

    # df = df.head(5)
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient='list')


def process_routing_data(process_df):
    """Process routing CSV data."""
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