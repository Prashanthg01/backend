# api/tasks.py

from celery import shared_task, current_task
from celery.result import AsyncResult
import pandas as pd
import numpy as np
from django.db.models import Sum
from datetime import datetime

from .models import Product, Machine, ProcessStep, ProductionSchedule
from .utils import (
    process_frontpage_data,
    process_routing_data,
    run_job_shop_scheduler,
    compute_schedule_kpis,
    optimize_product_batches_jointly,
    calculate_optimal_batch_size
)
import traceback


@shared_task(bind=True, name='initialize_data_task')
def initialize_data_task(self, frontpage_csv_path, process_csv_path):
    """
    Background task that:
    - Reads CSV files
    - Processes demand and routing data
    - Clears old database data
    - Creates machines
    - Creates products
    - Creates process steps
    - Tracks progress throughout
    """
    try:
        # -------------------------------------------------------------
        # STEP 1: Indicate task started (10%)
        # -------------------------------------------------------------
        self.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'status': 'Reading CSV files'}
        )

        # -------------------------------------------------------------
        # STEP 2: Read CSV files into pandas DataFrames
        # -------------------------------------------------------------
        frontpage_df = pd.read_csv(frontpage_csv_path)
        process_df   = pd.read_csv(process_csv_path)

        # Remove last 2 unnecessary columns from process file
        process_df = process_df.iloc[:, :-2]

        # -------------------------------------------------------------
        # STEP 3: Process raw CSV data into structured format
        # -------------------------------------------------------------
        self.update_state(
            state='PROGRESS',
            meta={'progress': 30, 'status': 'Processing data'}
        )

        # Extract product demand data
        demand_data = process_frontpage_data(frontpage_df)

        # Extract routing steps and list of machines
        process_routing, machines_list = process_routing_data(process_df)

        # -------------------------------------------------------------
        # STEP 4: Clear existing database data (reset system)
        # -------------------------------------------------------------
        self.update_state(
            state='PROGRESS',
            meta={'progress': 40, 'status': 'Clearing existing data'}
        )

        Product.objects.all().delete()
        Machine.objects.all().delete()
        ProcessStep.objects.all().delete()
        ProductionSchedule.objects.all().delete()

        # -------------------------------------------------------------
        # STEP 5: Create Machine records
        # -------------------------------------------------------------
        self.update_state(
            state='PROGRESS',
            meta={'progress': 50, 'status': 'Creating machines'}
        )

        machines = {}

        # Create one Machine object per unique machine name
        # Default available time = 24 hours per day
        for machine_name in machines_list:
            if machine_name:
                machines[machine_name] = Machine.objects.create(
                    name=machine_name,
                    available_hours_per_day=24
                )

        # -------------------------------------------------------------
        # STEP 6: Create Product records
        # -------------------------------------------------------------
        self.update_state(
            state='PROGRESS',
            meta={'progress': 60, 'status': 'Creating products'}
        )

        total_items = len(demand_data['Item'])

        for i in range(total_items):

            # ---------------------------------------------------------
            # Update progress while creating products
            # Formula:
            #   progress = 60 + (i / total_items) * 30
            #
            # 60% → product creation start
            # +30% → product + process step creation
            # ---------------------------------------------------------
            if i % 10 == 0:
                progress = 60 + int((i / total_items) * 30)
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': progress,
                        'status': f'Creating products ({i}/{total_items})'
                    }
                )

            # ---------------------------------------------------------
            # Read product basic data
            # ---------------------------------------------------------
            item   = demand_data['Item'][i]
            demand = demand_data['Demand_2024'][i]

            # If demand is missing → treat as zero
            if pd.isna(demand) or demand is None:
                demand = 0

            # ---------------------------------------------------------
            # Initial batch size logic (before optimization)
            #
            # Formula:
            #   batch_size = ceil(demand / 12)
            #
            # Meaning:
            #   Split yearly demand into 12 batches (monthly assumption)
            #
            # If demand = 0 → batch size = 1 (default safe value)
            # ---------------------------------------------------------
            batch_size = int(np.ceil(demand / 12)) if demand > 0 else 1

            # ---------------------------------------------------------
            # Create Product record
            # num_batches initially fixed to 12
            # ---------------------------------------------------------
            product = Product.objects.create(
                item        = item,
                sap_tn      = str(demand_data['SAP_TN'][i])   if demand_data['SAP_TN'][i]   is not None else '',
                sap_pl      = str(demand_data['SAP_PL'][i])   if demand_data['SAP_PL'][i]   is not None else None,
                dcc_type    = demand_data['DCC_Type'][i]      if demand_data['DCC_Type'][i] is not None else '',
                description = demand_data['Description'][i]   if demand_data['Description'][i] is not None else '',
                demand_2024 = int(demand),
                batch_size  = batch_size,
                num_batches = 12
            )

            # ---------------------------------------------------------
            # STEP 7: Create Process Steps for this product
            #
            # For each routing step:
            #   - If step belongs to this product
            #   - Link it to the correct machine
            # ---------------------------------------------------------
            for step_data in process_routing:

                if step_data['item'] == item:

                    machine_name = step_data['machine']

                    if machine_name in machines:

                        ProcessStep.objects.create(
                            product            = product,
                            step_number        = step_data['step'],
                            machine            = machines[machine_name],
                            step_name          = step_data['name'],
                            cycle_time_seconds = step_data['time'],
                            workers_required   = step_data['workers']
                        )

        # -------------------------------------------------------------
        # STEP 8: Mark task as complete (100%)
        # -------------------------------------------------------------
        self.update_state(
            state='PROGRESS',
            meta={'progress': 100, 'status': 'Complete'}
        )

        # -------------------------------------------------------------
        # STEP 9: Return summary of created records
        # -------------------------------------------------------------
        return {
            'status': 'success',
            'message': 'Database initialized successfully',
            'products_created': Product.objects.count(),
            'machines_created': Machine.objects.count(),
            'process_steps_created': ProcessStep.objects.count()
        }

    # -------------------------------------------------------------
    # ERROR HANDLING
    # If anything fails, return error details
    # -------------------------------------------------------------
    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)
        return {'status': 'error', 'message': str(e), 'traceback': error_trace}


@shared_task(bind=True, name="batch_optimize_preview_task")
def batch_optimize_preview_task(self, params):
    """
    Background task for individual batch optimization preview.

    Simulates what the optimizer *would* do for each product without
    writing anything to the database. Results are returned for display
    in the UI so the user can review before committing.

    Params (dict keys)
    ------------------
    max_num_batches : int   Upper limit on how many batches a product can have
    min_batch_size  : int   Minimum allowed units per batch
    max_batch_size  : int   Maximum allowed units per batch

    Returns
    -------
    dict with:
        batch_analysis  — per-product breakdown of old vs new batch values
        summary         — aggregate stats (totals, averages, std dev)
        parameters      — the params dict echoed back for traceability
    """
    try:
        # -------------------------------------------------------------
        # STEP 1: Signal task has started (10%)
        # Loading all qualifying products from the database is the
        # first real operation, so we report progress immediately.
        # -------------------------------------------------------------
        self.update_state(state="PROGRESS", meta={"progress": 10, "status": "Loading products"})

        # -------------------------------------------------------------
        # STEP 2: Fetch all products that have real demand
        # Products with demand = 0 have nothing to optimize.
        # -------------------------------------------------------------
        products       = Product.objects.filter(demand_2024__gt=0)
        batch_analysis = []   # Will hold per-product result dicts
        total_demand   = 0    # Running sum of all product demands
        total_batches  = 0    # Running sum of projected batch counts
        batch_sizes    = []   # Collect batch sizes for stats (mean, std, etc.)
        total_products = products.count()

        # -------------------------------------------------------------
        # STEP 3: Iterate over every product and simulate optimization
        # -------------------------------------------------------------
        for idx, product in enumerate(products):

            # ---------------------------------------------------------
            # STEP 3A: Update progress every 10 products
            # Formula:
            #   progress = 10 + (idx / total_products) * 80
            #
            # Why:
            #   10%  → reserved for the initial load step above
            #   +80% → spread across all product iterations
            #   10%  → reserved for the final completion step below
            # ---------------------------------------------------------
            if idx % 10 == 0:
                progress = 10 + int((idx / total_products) * 80)
                self.update_state(state="PROGRESS", meta={
                    "progress": progress,
                    "status":   f"Analyzing products ({idx}/{total_products})",
                })

            # ---------------------------------------------------------
            # STEP 3B: Calculate the optimal batch parameters
            # calculate_optimal_batch_size returns:
            #   batch_size   — units per batch (rounded, within bounds)
            #   num_batches  — how many batches needed to cover demand
            #   ideal_batch  — unconstrained floating-point ideal size
            #
            # Constraints enforced internally:
            #   batch_size  ∈ [min_batch_size, max_batch_size]
            #   num_batches ≤ max_num_batches
            #   batch_size × num_batches ≥ demand
            # ---------------------------------------------------------
            demand = int(product.demand_2024)
            batch_size, num_batches, ideal_batch = calculate_optimal_batch_size(
                demand,
                params["max_num_batches"],
                params["min_batch_size"],
                params["max_batch_size"],
            )

            # ---------------------------------------------------------
            # STEP 3C: Capture existing values for comparison
            # These are what is currently stored in the database,
            # before any optimization is applied.
            # ---------------------------------------------------------
            old_batch_size  = product.batch_size
            old_num_batches = product.num_batches

            # ---------------------------------------------------------
            # STEP 3D: Calculate improvement percentage
            # Formula:
            #   improvement = (old_num_batches - new_num_batches)
            #                 / old_num_batches × 100
            #
            # A positive value means fewer batches → fewer setups → better.
            # If nothing changed, show "0%".
            # Guard against old_num_batches = 0 to avoid ZeroDivisionError.
            # ---------------------------------------------------------
            improvement = (
                f"{((old_num_batches - num_batches) / old_num_batches * 100):.1f}%"
                if old_num_batches and old_num_batches != num_batches else "0%"
            )

            # ---------------------------------------------------------
            # STEP 3E: Append this product's result to the analysis list
            # Description is truncated to 50 chars for display safety.
            # ---------------------------------------------------------
            batch_analysis.append({
                "item":             product.item,
                "description":      (product.description or "")[:50],
                "demand":           demand,
                "old_batch_size":   old_batch_size,
                "old_num_batches":  old_num_batches,
                "new_batch_size":   batch_size,
                "new_num_batches":  num_batches,
                "ideal_batch_size": round(ideal_batch, 2),
                "improvement":      improvement,
            })

            # Accumulate totals for the summary section
            total_demand  += demand
            total_batches += num_batches
            batch_sizes.append(batch_size)

        # -------------------------------------------------------------
        # STEP 4: Mark analysis as complete (100%)
        # Nothing has been saved — this is preview-only.
        # -------------------------------------------------------------
        self.update_state(state="PROGRESS", meta={"progress": 100, "status": "Complete"})

        # -------------------------------------------------------------
        # STEP 5: Return full preview result
        # Summary stats use numpy for mean/min/max/std calculations.
        # Fallback to 0 when the list is empty (no qualifying products).
        # -------------------------------------------------------------
        return {
            "status":         "success",
            "batch_analysis": batch_analysis,
            "summary": {
                "total_products": len(batch_analysis),
                "total_demand":   total_demand,
                "total_batches":  total_batches,
                "avg_batch_size": round(np.mean(batch_sizes), 2) if batch_sizes else 0,
                "min_batch_size": int(np.min(batch_sizes))       if batch_sizes else 0,
                "max_batch_size": int(np.max(batch_sizes))       if batch_sizes else 0,
                "std_batch_size": round(np.std(batch_sizes), 2)  if batch_sizes else 0,
            },
            "parameters": params,
        }

    # -------------------------------------------------------------
    # ERROR HANDLING
    # Any unexpected exception is caught here and returned as a
    # structured error dict so the frontend can display it cleanly.
    # -------------------------------------------------------------
    except Exception as e:
        return {
            "status":    "error",
            "message":   str(e),
            "traceback": traceback.format_exc(),
        }


@shared_task(bind=True, name='generate_schedule_task')
def generate_schedule_task(self, params: dict):
    """
    Background task that runs the full job-shop scheduling pipeline
    and persists the resulting schedule to the database.

    Parameters (all optional — sane defaults used when absent)
    ----------------------------------------------------------
    start_date          : str   ISO-8601 date/datetime  (default: today 00:00)
    local_opt_machines  : int   bottleneck machines to re-optimise (default 5)
    enable_compaction   : bool  run left-shift gap elimination (default True)
    clear_existing      : bool  wipe existing schedule first (default True)
    product_pks         : list[int]  subset of products (default: all with demand>0)
    batch_overrides     : [[pk, batch_size, num_batches], …]  (optional)

    Returns
    -------
    dict: status, message, kpis, total_operations
    """

    # Convenience helper to update Celery task state in one line
    def _progress(pct, msg):
        self.update_state(state='PROGRESS', meta={'progress': pct, 'status': msg})

    try:
        # -------------------------------------------------------------
        # STEP 1: Parse and validate incoming parameters
        # -------------------------------------------------------------

        # Parse the scheduling start date.
        # If not provided, default to today at midnight so the schedule
        # always starts from a clean 00:00 boundary.
        start_str = params.get('start_date')
        if start_str:
            start_dt = datetime.fromisoformat(start_str)
        else:
            start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Number of bottleneck machines to apply local re-optimisation on.
        # Higher values improve quality but increase runtime.
        local_opt = int(params.get('local_opt_machines', 5))

        # Whether to run left-shift compaction after scheduling.
        # Compaction closes idle gaps on machines, reducing total makespan.
        enable_compaction = bool(params.get('enable_compaction', True))

        # Whether to delete the existing schedule before saving the new one.
        # Set to False only when appending a partial re-schedule.
        clear = params.get('clear_existing', True)

        # -------------------------------------------------------------
        # STEP 2: Build the batch override map
        # batch_overrides is a list of [pk, batch_size, num_batches] triples
        # that override whatever is stored in the database for those products.
        # Convert to a dict keyed by product pk for O(1) lookup later.
        # -------------------------------------------------------------
        raw_overrides  = params.get('batch_overrides', [])
        batch_override = {int(r[0]): (int(r[1]), int(r[2])) for r in raw_overrides}

        # -------------------------------------------------------------
        # STEP 3: Load products from the database (5%)
        # Only products with actual demand are scheduled.
        # If a product_pks filter is provided, restrict to that subset.
        # -------------------------------------------------------------
        _progress(5, 'Loading products…')
        product_pks = params.get('product_pks')
        if product_pks:
            products = list(Product.objects.filter(pk__in=product_pks, demand_2024__gt=0))
        else:
            products = list(Product.objects.filter(demand_2024__gt=0))

        # Abort early if there is nothing to schedule
        if not products:
            return {'status': 'error', 'message': 'No products with demand found in the database.'}

        # -------------------------------------------------------------
        # STEP 4: Run the job-shop scheduler
        # This is the heavy computation phase (Phases 1a, 1b, and 2):
        #
        #   Phase 1a — Initial greedy assignment of operations to machines
        #   Phase 1b — Local optimisation on the top N bottleneck machines
        #   Phase 2  — Optional left-shift compaction to close idle gaps
        #
        # Progress updates during scheduling are handled via the
        # progress_callback, which calls _progress() internally.
        #
        # Returns a list of schedule_row dicts, each containing:
        #   product_pk, step_number, batch_id, batch_num, batch_size,
        #   start_dt, end_dt, dur_hours, end_hrs
        # -------------------------------------------------------------
        schedule_rows = run_job_shop_scheduler(
            products           = products,
            start_dt           = start_dt,
            batch_override     = batch_override if batch_override else None,
            local_opt_machines = local_opt,
            enable_compaction  = enable_compaction,
            progress_callback  = _progress,
        )

        # Guard: if the scheduler returned nothing, something is wrong
        # with the process step data (e.g., no routing defined).
        if not schedule_rows:
            return {
                'status':  'error',
                'message': 'Scheduler produced no operations. Check that products have process steps.'
            }

        # -------------------------------------------------------------
        # STEP 5: Persist the schedule to the database (90%)
        # -------------------------------------------------------------
        _progress(90, 'Saving schedule to database…')

        # Optionally wipe the existing schedule before writing the new one
        if clear:
            ProductionSchedule.objects.all().delete()

        # Build lookup maps for products and process steps to avoid
        # repeated per-row database queries inside the loop below.
        # prod_map  : pk → Product instance
        # step_map  : (product_pk, step_number) → ProcessStep instance
        prod_map = {p.pk: p for p in products}
        step_map = {}
        for p in products:
            for s in ProcessStep.objects.filter(product=p).select_related('machine'):
                step_map[(p.pk, s.step_number)] = s

        # -------------------------------------------------------------
        # STEP 6: Convert schedule rows to ORM objects
        # Skip any row where the product or process step can't be
        # resolved (data integrity guard).
        # -------------------------------------------------------------
        to_create = []
        for row in schedule_rows:
            product   = prod_map.get(row['product_pk'])
            proc_step = step_map.get((row['product_pk'], row['step_number']))

            if not product or not proc_step:
                continue

            to_create.append(
                ProductionSchedule(
                    machine        = proc_step.machine,
                    product        = product,
                    process_step   = proc_step,
                    batch_id       = row['batch_id'],
                    batch_num      = row['batch_num'],
                    batch_size     = row['batch_size'],
                    start_time     = row['start_dt'],
                    end_time       = row['end_dt'],
                    duration_hours = round(row['dur_hours'], 4),
                )
            )

        # -------------------------------------------------------------
        # STEP 7: Bulk-insert in chunks of 1000
        # Chunking avoids hitting database parameter limits on large
        # schedules and keeps memory usage predictable.
        # -------------------------------------------------------------
        chunk = 1000
        for i in range(0, len(to_create), chunk):
            ProductionSchedule.objects.bulk_create(to_create[i:i + chunk])

        # -------------------------------------------------------------
        # STEP 8: Compute KPIs from the schedule rows
        # makespan = the latest end time across all operations (hours)
        # kpis     = dict of schedule quality metrics (utilisation, etc.)
        # -------------------------------------------------------------
        makespan = max(row['end_hrs'] for row in schedule_rows)
        kpis     = compute_schedule_kpis(schedule_rows, makespan)

        # -------------------------------------------------------------
        # STEP 9: Mark task complete and return results (100%)
        # -------------------------------------------------------------
        _progress(100, 'Complete')

        return {
            'status':           'success',
            'message':          'Schedule generated and saved successfully.',
            'total_operations': len(to_create),
            'kpis':             kpis,
        }

    # -------------------------------------------------------------
    # ERROR HANDLING
    # Unlike other tasks, we also call update_state(FAILURE) here
    # so Celery marks the task as failed in addition to returning
    # the error dict — useful for monitoring dashboards.
    # -------------------------------------------------------------
    except Exception as exc:
        tb = traceback.format_exc()
        print(tb)
        self.update_state(
            state='FAILURE',
            meta={'progress': 0, 'status': 'Failed', 'error': str(exc)}
        )
        return {'status': 'error', 'message': str(exc), 'traceback': tb}


@shared_task(bind=True, name="batch_optimize_save_task")
def batch_optimize_save_task(self, params):
    """
    Background task that:
    - Reads all products with demand > 0
    - Calculates optimal batch size & number of batches
    - Updates products if values changed
    - Tracks progress
    - Returns summary

    Unlike batch_optimize_preview_task, this task WRITES results to
    the database. It should only be called after the user has reviewed
    the preview and confirmed they want to apply the changes.

    Params (dict keys)
    ------------------
    max_num_batches : int   Upper limit on how many batches a product can have
    min_batch_size  : int   Minimum allowed units per batch
    max_batch_size  : int   Maximum allowed units per batch
    """
    try:
        # -------------------------------------------------------------
        # STEP 1: Mark task as started (10% progress)
        # This tells the frontend that processing has begun.
        # -------------------------------------------------------------
        self.update_state(
            state="PROGRESS",
            meta={"progress": 10, "status": "Loading products"}
        )

        # -------------------------------------------------------------
        # STEP 2: Fetch only products that actually need optimization
        # Condition:
        #     demand_2024 > 0
        # (No need to optimize products with zero demand)
        # -------------------------------------------------------------
        products = Product.objects.filter(demand_2024__gt=0)

        # Count total products for progress calculation
        total_products = products.count()

        # Lists to track results
        updated = []   # Products whose batch values changed
        skipped = []   # Products skipped (invalid result from optimizer)

        # -------------------------------------------------------------
        # STEP 3: Loop through each product and apply optimization
        # -------------------------------------------------------------
        for idx, product in enumerate(products):

            # ---------------------------------------------------------
            # STEP 3A: Update progress every 10 products
            # Formula:
            #     progress = 10 + (idx / total_products) * 80
            #
            # Why?
            # - First 10% reserved for loading (Step 1 above)
            # - Next 80% spread across all product iterations
            # - Final 10% reserved for the completion step below
            # ---------------------------------------------------------
            if idx % 10 == 0:
                progress = 10 + int((idx / total_products) * 80)
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "progress": progress,
                        "status": f"Optimizing products ({idx}/{total_products})",
                    },
                )

            # ---------------------------------------------------------
            # STEP 3B: Read product demand
            # Cast to int to ensure consistent arithmetic downstream.
            # ---------------------------------------------------------
            demand = int(product.demand_2024)

            # ---------------------------------------------------------
            # STEP 3C: Call optimizer to calculate:
            #   - Optimal batch size (B)
            #   - Optimal number of batches (N)
            #   - Unconstrained ideal batch size (ignored here)
            #
            # Core constraint solved inside:
            #       B × N ≥ demand  (total production covers demand)
            #
            # Objective:
            #       Minimize N  (fewer setups = less changeover cost)
            #
            # Bounds enforced:
            #       min_batch_size ≤ B ≤ max_batch_size
            #       N ≤ max_num_batches
            # ---------------------------------------------------------
            batch_size, num_batches, _ = calculate_optimal_batch_size(
                demand,
                params["max_num_batches"],
                params["min_batch_size"],
                params["max_batch_size"],
            )

            # ---------------------------------------------------------
            # STEP 3D: Skip products where optimizer found no solution
            # batch_size == 0 means the demand is impossible to satisfy
            # within the given constraints — log and move on.
            # ---------------------------------------------------------
            if batch_size == 0:
                skipped.append(product.item)
                continue

            # ---------------------------------------------------------
            # STEP 3E: Update product ONLY if values changed
            # Avoid unnecessary database writes for products where
            # the optimizer agrees with what is already stored.
            # ---------------------------------------------------------
            if (
                product.batch_size != batch_size
                or product.num_batches != num_batches
            ):
                # Store old values for the change report returned at the end
                old_bs = product.batch_size
                old_nb = product.num_batches

                # Apply new optimized values
                product.batch_size  = batch_size
                product.num_batches = num_batches

                # Save only the two changed fields for efficiency —
                # avoids triggering any unrelated field validation.
                product.save(update_fields=["batch_size", "num_batches"])

                # Record the change for the final summary
                updated.append({
                    "item":            product.item,
                    "old_batch_size":  old_bs,
                    "old_num_batches": old_nb,
                    "new_batch_size":  batch_size,
                    "new_num_batches": num_batches,
                })

        # -------------------------------------------------------------
        # STEP 4: Mark task as completed (100%)
        # All products have been processed and changes saved.
        # -------------------------------------------------------------
        self.update_state(
            state="PROGRESS",
            meta={"progress": 100, "status": "Complete"}
        )

        # -------------------------------------------------------------
        # STEP 5: Return success summary
        # Includes:
        # - total_updated      — how many products were actually changed
        # - total_skipped      — how many had no valid solution
        # - updated_products   — detailed before/after for each changed product
        # - skipped_items      — item codes with no valid solution
        # - parameters         — the params dict echoed back for traceability
        # -------------------------------------------------------------
        return {
            "status":           "success",
            "message":          "Batch optimization applied successfully",
            "total_updated":    len(updated),
            "total_skipped":    len(skipped),
            "updated_products": updated,
            "skipped_items":    skipped,
            "parameters":       params,
        }

    # -------------------------------------------------------------
    # ERROR HANDLING
    # Any unexpected exception is caught here and returned as a
    # structured error dict so the frontend can display it cleanly.
    # -------------------------------------------------------------
    except Exception as e:
        return {
            "status":    "error",
            "message":   str(e),
            "traceback": traceback.format_exc(),
        }


@shared_task(bind=True, name="joint_optimize_task")
def joint_optimize_task(self, params, save_to_db: bool = True):
    """
    Celery task for joint multi-product batch optimization.

    Unlike the individual optimizer (batch_optimize_save_task) which
    treats each product in isolation, this task runs a single MILP
    (Mixed-Integer Linear Program) that considers ALL products and ALL
    machines simultaneously. This allows it to balance machine loads
    across the entire product mix rather than optimizing each product
    greedily one by one.

    Optionally saves results to DB (save_to_db=True) or returns a
    preview without touching the database (save_to_db=False).

    Params (dict keys)
    ------------------
    max_num_batches : int   Upper limit on batches per product
    min_batch_size  : int   Minimum allowed units per batch
    max_batch_size  : int   Maximum allowed units per batch
    """
    try:
        # -------------------------------------------------------------
        # STEP 1: Load products and their routing data (10%)
        # prefetch_related pulls in ProcessStep + Machine in one query,
        # avoiding queries when the MILP builder inspects routings.
        # -------------------------------------------------------------
        self.update_state(state="PROGRESS", meta={
            "progress": 10,
            "status":   "Loading products and routing data",
        })

        products = list(
            Product.objects.filter(demand_2024__gt=0).prefetch_related(
                "processstep_set__machine"
            )
        )

        # -------------------------------------------------------------
        # STEP 2: Signal that the MILP model is being assembled (20%)
        # Building the model involves constructing decision variables,
        # constraints, and the objective function for all products and
        # machines. This can take a few seconds for large catalogs.
        # -------------------------------------------------------------
        self.update_state(state="PROGRESS", meta={
            "progress": 20,
            "status":   f"Building joint MILP for {len(products)} products",
        })

        # -------------------------------------------------------------
        # STEP 3: Run the optimizer
        # Two paths depending on save_to_db:
        #
        # save_to_db = True  → full run via optimize_product_batches_jointly
        #              Solves the MILP, updates Product records in DB,
        #              and returns a result dict with per-product outcomes.
        #
        # save_to_db = False → preview-only via _joint_preview_no_save
        #              Solves the MILP but does NOT touch the database.
        #              Used when the user wants to review results first.
        #
        # time_limit_seconds limits the CBC solver to avoid hanging
        # indefinitely on very large or infeasible problems.
        # -------------------------------------------------------------
        if save_to_db:
            result = optimize_product_batches_jointly(
                products,
                params["max_num_batches"],
                params["min_batch_size"],
                params["max_batch_size"],
                time_limit_seconds=180,
            )
        else:
            from .utils import _joint_preview_no_save
            result = _joint_preview_no_save(
                products,
                params["max_num_batches"],
                params["min_batch_size"],
                params["max_batch_size"],
            )

        # -------------------------------------------------------------
        # STEP 4: Mark optimization as complete (100%)
        # The heavy computation is done; the rest is just summarising.
        # -------------------------------------------------------------
        self.update_state(state="PROGRESS", meta={"progress": 100, "status": "Complete"})

        # -------------------------------------------------------------
        # STEP 5: Build summary statistics from the result
        # results_list — one entry per product with old/new batch values
        # batch_sizes  — new batch sizes (used for average calculation)
        # improvements — percentage improvement per product (parsed from
        #                the "X.X%" string stored in each result row)
        # -------------------------------------------------------------
        results_list = result.get("results", [])
        batch_sizes  = [r["new_batch_size"] for r in results_list if r.get("demand", 0) > 0]

        # Parse the improvement strings (e.g., "12.5%") back to floats
        # for computing the average. Fallback to 0.0 on parse errors.
        improvements = []
        for r in results_list:
            try:
                improvements.append(float(r["improvement"].replace("%", "")))
            except (ValueError, AttributeError):
                improvements.append(0.0)

        # -------------------------------------------------------------
        # STEP 6: Return full result dict
        # Includes:
        #   batch_analysis      — per-product breakdown
        #   machine_loads       — dict of machine_name → total load hours
        #   makespan_proxy      — MILP objective value (proxy for makespan)
        #   products_updated    — count of DB rows changed (0 if preview)
        #   summary             — aggregate stats for the UI dashboard
        #   parameters          — echo of input params for traceability
        # -------------------------------------------------------------
        return {
            "status":           "success",
            "solver_status":    result["status"],
            "message":          result["message"],
            "batch_analysis":   results_list,
            "machine_loads":    result["machine_loads"],
            "makespan_proxy":   result["makespan_proxy"],
            "products_updated": result.get("products_updated", 0),
            "summary": {
                "total_products":     len(results_list),
                "total_demand":       sum(r["demand"] for r in results_list),
                "total_batches":      sum(r["new_num_batches"] for r in results_list
                                          if r.get("demand", 0) > 0),
                "avg_batch_size":     round(import_mean(batch_sizes), 1) if batch_sizes else 0,
                "avg_improvement":    round(import_mean(improvements), 1) if improvements else 0,
                # 0–100 score indicating how evenly load is spread across machines
                "load_balance_score": _load_balance_score(result["machine_loads"]),
            },
            "parameters": params,
        }

    # -------------------------------------------------------------
    # ERROR HANDLING
    # Structured error dict keeps the API response shape consistent
    # whether the task succeeds or fails.
    # -------------------------------------------------------------
    except Exception as e:
        return {
            "status":    "error",
            "message":   str(e),
            "traceback": traceback.format_exc(),
        }


# ── tiny helpers ──────────────────────────────────────────────────────────────

def import_mean(lst):
    """Return the arithmetic mean of a list, or 0 if the list is empty."""
    return sum(lst) / len(lst) if lst else 0


def _load_balance_score(machine_loads: dict) -> float:
    """
    Compute a 0–100 score representing how evenly machine load is distributed.

    Score interpretation:
        100 — all machines carry exactly equal load (perfectly balanced)
          0 — virtually all load is concentrated on a single machine

    Formula:
        score = (1 - (max_load - min_load) / max_load) × 100

    The small epsilon (1e-9) in the denominator prevents ZeroDivisionError
    when all machines have zero load.
    """
    vals = list(machine_loads.values())
    if not vals or max(vals) == 0:
        return 100.0
    return round(
        (1 - (max(vals) - min(vals)) / (max(vals) + 1e-9)) * 100, 1
    )


@shared_task(bind=True, name="initialize_data_task")
def initialize_data_task(
    self,
    frontpage_csv_path: str | None,
    process_csv_path: str | None,
    synthetic_dataset: dict | None = None,
    clear_existing: bool = True,
):
    """
    Background Celery task that populates master data tables.

    Supports two modes, chosen automatically based on arguments:

    ┌─────────────────────────────────────────────────────────────┐
    │  Mode A — Real CSV (original behaviour, unchanged)          │
    │    frontpage_csv_path  ← path to Frontpage.csv              │
    │    process_csv_path    ← path to Process.csv                │
    │    synthetic_dataset   = None  (default)                    │
    ├─────────────────────────────────────────────────────────────┤
    │  Mode B — Synthetic dataset                                 │
    │    synthetic_dataset   ← dict from generate_synthetic_dataset│
    │    frontpage_csv_path  = None  (ignored)                    │
    │    process_csv_path    = None  (ignored)                    │
    └─────────────────────────────────────────────────────────────┘

    In both modes the task returns the same result dict so the
    frontend can handle both identically.

    Step-by-step (Mode A — Real CSV)
    ---------------------------------
    1.  Update progress → 10 %
    2.  Read CSV files into DataFrames
    3.  Parse demand + routing via process_frontpage_data / process_routing_data
    4.  Clear existing DB data (if clear_existing=True)
    5.  Create Machine records
    6.  Create Product records (batch_size = ceil(demand/12))
    7.  Create ProcessStep records linked to products and machines
    8.  Report 100 % and return summary dict

    Step-by-step (Mode B — Synthetic)
    ----------------------------------
    1.  Update progress → 10 %
    2.  Skip CSV reading — dataset already in memory
    3.  Call insert_synthetic_dataset_into_db() which handles steps 4-7
    4.  Report 100 % and return summary dict
    """

    def _progress(pct: int, msg: str) -> None:
        self.update_state(state="PROGRESS", meta={"progress": pct, "status": msg})

    try:
        # ── STEP 1: Signal task started ───────────────────────────────────────
        _progress(10, "Starting initialization…")

        # =====================================================================
        # MODE B: Synthetic dataset path
        # =====================================================================
        if synthetic_dataset is not None:
            _progress(20, "Loading synthetic dataset into database…")

            from .utils_folder.synthetic_generator import insert_synthetic_dataset_into_db  # noqa

            result = insert_synthetic_dataset_into_db(
                synthetic_dataset,
                clear_existing=clear_existing,
            )

            _progress(100, "Synthetic data initialization complete")

            return {
                "status":                "success",
                "message":               "Synthetic database initialized successfully",
                "dataset_type":          "synthetic",
                "products_created":      result["products_created"],
                "machines_created":      result["machines_created"],
                "process_steps_created": result["process_steps_created"],
                "metadata":              result.get("metadata", {}),
            }

        # =====================================================================
        # MODE A: Real CSV path (original behaviour — unchanged)
        # =====================================================================

        # ── STEP 2: Read CSV files ────────────────────────────────────────────
        _progress(10, "Reading CSV files")

        frontpage_df = pd.read_csv(frontpage_csv_path)
        process_df   = pd.read_csv(process_csv_path)
        process_df   = process_df.iloc[:, :-2]

        # ── STEP 3: Parse CSV data ────────────────────────────────────────────
        _progress(30, "Processing data")

        demand_data                    = process_frontpage_data(frontpage_df)
        process_routing, machines_list = process_routing_data(process_df)

        # ── STEP 4: Clear existing database data ──────────────────────────────
        _progress(40, "Clearing existing data")

        if clear_existing:
            Product.objects.all().delete()
            Machine.objects.all().delete()
            ProcessStep.objects.all().delete()
            ProductionSchedule.objects.all().delete()

        # ── STEP 5: Create Machine records ────────────────────────────────────
        _progress(50, "Creating machines")

        machines: dict = {}
        for machine_name in machines_list:
            if machine_name:
                machines[machine_name] = Machine.objects.create(
                    name=machine_name,
                    available_hours_per_day=24,
                )

        # ── STEP 6: Create Product records ────────────────────────────────────
        _progress(60, "Creating products")

        total_items = len(demand_data["Item"])

        for i in range(total_items):
            if i % 10 == 0:
                progress = 60 + int((i / total_items) * 30)
                _progress(progress, f"Creating products ({i}/{total_items})")

            item   = demand_data["Item"][i]
            demand = demand_data["Demand_2024"][i]

            if pd.isna(demand) or demand is None:
                demand = 0

            batch_size = int(np.ceil(demand / 12)) if demand > 0 else 1

            product = Product.objects.create(
                item        = item,
                sap_tn      = str(demand_data["SAP_TN"][i])    if demand_data["SAP_TN"][i]    is not None else "",
                sap_pl      = str(demand_data["SAP_PL"][i])    if demand_data["SAP_PL"][i]    is not None else None,
                dcc_type    = demand_data["DCC_Type"][i]       if demand_data["DCC_Type"][i]  is not None else "",
                description = demand_data["Description"][i]   if demand_data["Description"][i] is not None else "",
                demand_2024 = int(demand),
                batch_size  = batch_size,
                num_batches = 12,
            )

            # ── STEP 7: Create ProcessStep records ────────────────────────────
            for step_data in process_routing:
                if step_data["item"] == item:
                    machine_name = step_data["machine"]
                    if machine_name in machines:
                        ProcessStep.objects.create(
                            product            = product,
                            step_number        = step_data["step"],
                            machine            = machines[machine_name],
                            step_name          = step_data["name"],
                            cycle_time_seconds = step_data["time"],
                            workers_required   = step_data["workers"],
                        )

        # ── STEP 8: Return summary ────────────────────────────────────────────
        _progress(100, "Complete")

        return {
            "status":                "success",
            "message":               "Database initialized successfully",
            "dataset_type":          "real",
            "products_created":      Product.objects.count(),
            "machines_created":      Machine.objects.count(),
            "process_steps_created": ProcessStep.objects.count(),
        }

    except Exception as exc:
        error_trace = traceback.format_exc()
        print(error_trace)
        return {"status": "error", "message": str(exc), "traceback": error_trace}
