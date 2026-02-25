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
    """Background task for database initialization from CSV files."""
    try:
        self.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Reading CSV files'})

        frontpage_df = pd.read_csv(frontpage_csv_path)
        process_df   = pd.read_csv(process_csv_path)
        process_df   = process_df.iloc[:, :-2]

        self.update_state(state='PROGRESS', meta={'progress': 30, 'status': 'Processing data'})

        demand_data                      = process_frontpage_data(frontpage_df)
        process_routing, machines_list   = process_routing_data(process_df)

        self.update_state(state='PROGRESS', meta={'progress': 40, 'status': 'Clearing existing data'})

        Product.objects.all().delete()
        Machine.objects.all().delete()
        ProcessStep.objects.all().delete()
        ProductionSchedule.objects.all().delete()

        self.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Creating machines'})

        machines = {}
        for machine_name in machines_list:
            if machine_name:
                machines[machine_name] = Machine.objects.create(
                    name=machine_name,
                    available_hours_per_day=24
                )

        self.update_state(state='PROGRESS', meta={'progress': 60, 'status': 'Creating products'})

        total_items = len(demand_data['Item'])
        for i in range(total_items):
            if i % 10 == 0:
                progress = 60 + int((i / total_items) * 30)
                self.update_state(state='PROGRESS', meta={
                    'progress': progress,
                    'status':   f'Creating products ({i}/{total_items})'
                })

            item   = demand_data['Item'][i]
            demand = demand_data['Demand_2024'][i]

            if pd.isna(demand) or demand is None:
                demand = 0

            batch_size = int(np.ceil(demand / 12)) if demand > 0 else 1

            product = Product.objects.create(
                item        = item,
                sap_tn      = str(demand_data['SAP_TN'][i])   if demand_data['SAP_TN'][i]   is not None else '',
                sap_pl      = str(demand_data['SAP_PL'][i])   if demand_data['SAP_PL'][i]   is not None else None,
                dcc_type    = demand_data['DCC_Type'][i]       if demand_data['DCC_Type'][i]  is not None else '',
                description = demand_data['Description'][i]   if demand_data['Description'][i] is not None else '',
                demand_2024 = int(demand),
                batch_size  = batch_size,
                num_batches = 12
            )

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

        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Complete'})

        return {
            'status':               'success',
            'message':              'Database initialized successfully',
            'products_created':     Product.objects.count(),
            'machines_created':     Machine.objects.count(),
            'process_steps_created': ProcessStep.objects.count()
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)
        return {'status': 'error', 'message': str(e), 'traceback': error_trace}


@shared_task(bind=True, name="batch_optimize_preview_task")
def batch_optimize_preview_task(self, params):
    """Background task for individual batch optimization preview."""
    try:
        self.update_state(state="PROGRESS", meta={"progress": 10, "status": "Loading products"})

        products       = Product.objects.filter(demand_2024__gt=0)
        batch_analysis = []
        total_demand   = 0
        total_batches  = 0
        batch_sizes    = []
        total_products = products.count()

        for idx, product in enumerate(products):
            if idx % 10 == 0:
                progress = 10 + int((idx / total_products) * 80)
                self.update_state(state="PROGRESS", meta={
                    "progress": progress,
                    "status":   f"Analyzing products ({idx}/{total_products})",
                })

            demand = int(product.demand_2024)
            batch_size, num_batches, ideal_batch = calculate_optimal_batch_size(
                demand,
                params["max_num_batches"],
                params["min_batch_size"],
                params["max_batch_size"],
            )

            old_batch_size  = product.batch_size
            old_num_batches = product.num_batches

            improvement = (
                f"{((old_num_batches - num_batches) / old_num_batches * 100):.1f}%"
                if old_num_batches and old_num_batches != num_batches else "0%"
            )

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
            total_demand  += demand
            total_batches += num_batches
            batch_sizes.append(batch_size)

        self.update_state(state="PROGRESS", meta={"progress": 100, "status": "Complete"})

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

    except Exception as e:
        return {
            "status":    "error",
            "message":   str(e),
            "traceback": traceback.format_exc(),
        }
    


@shared_task(bind=True, name='generate_schedule_task')
def generate_schedule_task(self, params: dict):
    """
    Parameters (all optional — sane defaults used when absent)
    ----------------------------------------------------------
    start_date          : str   ISO-8601 date/datetime  (default: today 00:00)
    local_opt_machines  : int   bottleneck machines to re-optimise (default 5)
    enable_compaction   : bool  run left-shift gap elimination (default True) ← NEW
    clear_existing      : bool  wipe existing schedule first (default True)
    product_pks         : list[int]  subset of products (default: all with demand>0)
    batch_overrides     : [[pk, batch_size, num_batches], …]  (optional)

    Returns
    -------
    dict: status, message, kpis, total_operations
    """
    def _progress(pct, msg):
        self.update_state(state='PROGRESS', meta={'progress': pct, 'status': msg})

    try:
        # ── Parse params ──────────────────────────────────────────────
        start_str = params.get('start_date')
        if start_str:
            start_dt = datetime.fromisoformat(start_str)
        else:
            start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        local_opt          = int(params.get('local_opt_machines', 5))
        enable_compaction  = bool(params.get('enable_compaction', True))   # ← NEW
        clear              = params.get('clear_existing', True)

        raw_overrides  = params.get('batch_overrides', [])
        batch_override = {int(r[0]): (int(r[1]), int(r[2])) for r in raw_overrides}

        # ── Load products ─────────────────────────────────────────────
        _progress(5, 'Loading products…')
        product_pks = params.get('product_pks')
        if product_pks:
            products = list(Product.objects.filter(pk__in=product_pks, demand_2024__gt=0))
        else:
            products = list(Product.objects.filter(demand_2024__gt=0))

        if not products:
            return {'status': 'error', 'message': 'No products with demand found in the database.'}

        # ── Run scheduler (Phase 1a + 1b + 2) ────────────────────────
        schedule_rows = run_job_shop_scheduler(
            products           = products,
            start_dt           = start_dt,
            batch_override     = batch_override if batch_override else None,
            local_opt_machines = local_opt,
            enable_compaction  = enable_compaction,   # ← NEW: passed through
            progress_callback  = _progress,
        )

        if not schedule_rows:
            return {
                'status':  'error',
                'message': 'Scheduler produced no operations. Check that products have process steps.'
            }

        # ── Persist to DB ─────────────────────────────────────────────
        _progress(90, 'Saving schedule to database…')

        if clear:
            ProductionSchedule.objects.all().delete()

        prod_map = {p.pk: p for p in products}
        step_map = {}
        for p in products:
            for s in ProcessStep.objects.filter(product=p).select_related('machine'):
                step_map[(p.pk, s.step_number)] = s

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

        chunk = 1000
        for i in range(0, len(to_create), chunk):
            ProductionSchedule.objects.bulk_create(to_create[i:i + chunk])

        # ── KPIs ──────────────────────────────────────────────────────
        makespan = max(row['end_hrs'] for row in schedule_rows)
        kpis     = compute_schedule_kpis(schedule_rows, makespan)

        _progress(100, 'Complete')

        return {
            'status':           'success',
            'message':          'Schedule generated and saved successfully.',
            'total_operations': len(to_create),
            'kpis':             kpis,
        }

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
    """Background task to apply optimized batch sizes to all products."""
    try:
        self.update_state(state="PROGRESS", meta={"progress": 10, "status": "Loading products"})

        products       = Product.objects.filter(demand_2024__gt=0)
        total_products = products.count()
        updated        = []
        skipped        = []

        for idx, product in enumerate(products):
            if idx % 10 == 0:
                progress = 10 + int((idx / total_products) * 80)
                self.update_state(state="PROGRESS", meta={
                    "progress": progress,
                    "status":   f"Optimizing products ({idx}/{total_products})",
                })

            demand = int(product.demand_2024)
            batch_size, num_batches, _ = calculate_optimal_batch_size(
                demand,
                params["max_num_batches"],
                params["min_batch_size"],
                params["max_batch_size"],
            )

            if batch_size == 0:
                skipped.append(product.item)
                continue

            if product.batch_size != batch_size or product.num_batches != num_batches:
                old_bs = product.batch_size
                old_nb = product.num_batches
                product.batch_size  = batch_size
                product.num_batches = num_batches
                product.save(update_fields=["batch_size", "num_batches"])
                updated.append({
                    "item":             product.item,
                    "old_batch_size":   old_bs,
                    "old_num_batches":  old_nb,
                    "new_batch_size":   batch_size,
                    "new_num_batches":  num_batches,
                })

        self.update_state(state="PROGRESS", meta={"progress": 100, "status": "Complete"})

        return {
            "status":           "success",
            "message":          "Batch optimization applied successfully",
            "total_updated":    len(updated),
            "total_skipped":    len(skipped),
            "updated_products": updated,
            "skipped_items":    skipped,
            "parameters":       params,
        }

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

    Runs the full MILP that balances machine loads across all products
    simultaneously. Optionally saves results to DB.

    Progress states:
      10% — Loading products & routing data
      20% — Building MILP model
      60% — Solving (CBC solver running)
      90% — Extracting results
     100% — Done
    """
    try:
        self.update_state(state="PROGRESS", meta={
            "progress": 10,
            "status":   "Loading products and routing data",
        })

        products = list(
            Product.objects.filter(demand_2024__gt=0).prefetch_related(
                "processstep_set__machine"
            )
        )

        self.update_state(state="PROGRESS", meta={
            "progress": 20,
            "status":   f"Building joint MILP for {len(products)} products",
        })

        if save_to_db:
            # Full run — saves to DB inside optimize_product_batches_jointly
            result = optimize_product_batches_jointly(
                products,
                params["max_num_batches"],
                params["min_batch_size"],
                params["max_batch_size"],
                time_limit_seconds=180,
            )
        else:
            # Preview-only — use the no-save helper from views
            from .views import _joint_preview_no_save
            result = _joint_preview_no_save(
                products,
                params["max_num_batches"],
                params["min_batch_size"],
                params["max_batch_size"],
            )

        self.update_state(state="PROGRESS", meta={"progress": 100, "status": "Complete"})

        # Build summary for task result
        results_list = result.get("results", [])
        batch_sizes  = [r["new_batch_size"]  for r in results_list if r.get("demand", 0) > 0]
        improvements = []
        for r in results_list:
            try:
                improvements.append(float(r["improvement"].replace("%", "")))
            except (ValueError, AttributeError):
                improvements.append(0.0)

        return {
            "status":           "success",
            "solver_status":    result["status"],
            "message":          result["message"],
            "batch_analysis":   results_list,
            "machine_loads":    result["machine_loads"],
            "makespan_proxy":   result["makespan_proxy"],
            "products_updated": result.get("products_updated", 0),
            "summary": {
                "total_products":    len(results_list),
                "total_demand":      sum(r["demand"] for r in results_list),
                "total_batches":     sum(r["new_num_batches"] for r in results_list
                                        if r.get("demand", 0) > 0),
                "avg_batch_size":    round(import_mean(batch_sizes), 1) if batch_sizes else 0,
                "avg_improvement":   round(import_mean(improvements), 1) if improvements else 0,
                "load_balance_score": _load_balance_score(result["machine_loads"]),
            },
            "parameters": params,
        }

    except Exception as e:
        return {
            "status":    "error",
            "message":   str(e),
            "traceback": traceback.format_exc(),
        }


# ── tiny helpers ──────────────────────────────────────────────────────────────

def import_mean(lst):
    return sum(lst) / len(lst) if lst else 0


def _load_balance_score(machine_loads: dict) -> float:
    """0–100 score: 100 = perfectly balanced, 0 = all load on one machine."""
    vals = list(machine_loads.values())
    if not vals or max(vals) == 0:
        return 100.0
    return round(
        (1 - (max(vals) - min(vals)) / (max(vals) + 1e-9)) * 100, 1
    )


