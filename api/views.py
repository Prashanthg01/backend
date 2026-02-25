# api/views.py

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import numpy as np
from django.db.models import Sum, Count, Q, Max, Min, Avg
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from celery.result import AsyncResult
from collections import defaultdict
import os
import tempfile
import traceback

from .models import Product, Machine, ProcessStep, ProductionSchedule
from .utils import (
    clean_numeric_columns, clean_text_columns, apply_filters,
    clean_shift_columns, calculate_efficiency, calculate_backlog,
    calculate_production_outputs, SHIFT_LABELS, build_summary,
    process_frontpage_data, process_routing_data,
    calculate_optimal_batch_size, pulp_optimize_buffers,
    optimize_product_batches_jointly
)
from .tasks import (
    initialize_data_task,
    generate_schedule_task,
    batch_optimize_preview_task,
    batch_optimize_save_task,
    joint_optimize_task
)

_preview_cache = {}


# ===========================================================================
# CSV PROCESSING
# ===========================================================================

@api_view(['POST'])
def process_csv(request):
    """
    Parse and summarize uploaded production CSV data.

    Step-by-step:
    1. Validate that a CSV file was uploaded in multipart form data.
    2. Parse request filters and the number of shifts to evaluate.
    3. Load CSV rows into a pandas DataFrame.
    4. Normalize numeric/text columns so downstream math/filtering is stable.
    5. Apply user-selected filters (PPS TN, project, machine, tool, area).
    6. Clean shift-range columns used by KPI helpers.
    7. Compute efficiency, backlog, finished-goods output, and connector output.
    8. Return shift-wise KPIs plus a high-level summary payload.
    """
    # Step 1: Validate required upload input.
    csv_file = request.FILES.get('file')
    if not csv_file:
        return Response({'error': 'No file uploaded'}, status=400)

    # Step 2: Read optional runtime parameters.
    num_shifts = int(request.POST.get("num_shifts", 28))
    # Step 3: Load CSV into memory for transformation and aggregation.
    df = pd.read_csv(csv_file)

    # Step 4: Standardize columns before any filtering or calculations.
    clean_numeric_columns(df, ['Planned', 'Realized', 'Backlog', 'Open'])
    clean_text_columns(df, ['Step', 'Area', 'Sub-Project'])
    clean_text_columns(df, ['PPS TN', 'Project', 'Sub-Project', 'Machine', 'Tool No.', 'Area'])

    # Step 5: Apply optional API filters from request body values.
    df = apply_filters(df, {
        'PPS TN':      request.POST.get("pps_tn",       "All"),
        'Project':     request.POST.get("project",      "All"),
        'Sub-Project': request.POST.get("sub_project",  "All"),
        'Machine':     request.POST.get("machine",      "All"),
        'Tool No.':    request.POST.get("tool_no",      "All"),
        'Area':        request.POST.get("area",         "All"),
    })

    # Step 6: Clean shift columns expected by legacy index windows.
    clean_shift_columns(df, [(14, 50), (95, 113)])

    # Step 7: Compute KPI series used in the response.
    efficiency = calculate_efficiency(df, num_shifts)
    backlog    = calculate_backlog(df)

    finished_filter  = (df['Step'] == 'F') & df['Sub-Project'].notna()
    connector_filter = (df['Area'] == 'Assembly') & df['Sub-Project'].notna()

    fg_output, conn_output = calculate_production_outputs(df, finished_filter, connector_filter)

    # Step 8: Build a client-friendly response shape.
    result = {
        "Total Backlog Finished Goods":      dict(zip(SHIFT_LABELS, map(str, backlog))),
        "Production Output Finished Goods":  fg_output,
        "Production Output Connectors":      conn_output,
        "Overall Efficiency": {
            shift: f"{eff:.2f}%" if eff > 0 else "-"
            for shift, eff in zip(SHIFT_LABELS, efficiency)
        },
    }

    return Response({"ShiftWise": result, "Summary": build_summary(df)})


@api_view(['GET'])
def get_filter_options(request):
    """
    Return static/derived filter options used by frontend controls.

    Step-by-step:
    1. Read machines from DB.
    2. Read products with positive demand.
    3. Build a distinct sorted list of non-empty DCC types.
    4. Detect available schedule date range (min start, max end).
    5. Return all filter metadata in one payload.
    """
    try:
        # Step 1: Load all machine names.
        machines = list(Machine.objects.all().values_list('name', flat=True))
        # Step 2: Load active products with core fields used by UI.
        products = list(
            Product.objects.filter(demand_2024__gt=0).values(
                'item', 'description', 'dcc_type', 'demand_2024', 'batch_size', 'num_batches'
            )
        )
        # Step 3: Build distinct list of available DCC types.
        dcc_types = list(
            Product.objects.filter(demand_2024__gt=0)
            .values_list('dcc_type', flat=True)
            .distinct()
            .order_by('dcc_type')
        )
        dcc_types = [t for t in dcc_types if t]

        # Step 4: Compute schedule date boundaries if schedules exist.
        schedules = ProductionSchedule.objects.all()
        if schedules.exists():
            min_date = schedules.aggregate(Min('start_time'))['start_time__min']
            max_date = schedules.aggregate(Max('end_time'))['end_time__max']
        else:
            min_date = max_date = None

        # Step 5: Return filter options.
        return Response({
            'machines':   machines,
            'products':   products,
            'dcc_types':  dcc_types,
            'date_range': {'min': min_date, 'max': max_date}
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# KPIs
# ===========================================================================

@api_view(['GET'])
def get_kpis(request):
    """
    Compute global schedule KPIs and machine-level utilization.

    Step-by-step:
    1. Ensure a schedule exists.
    2. Compute makespan window in hours/days.
    3. Compute per-machine used hours, operation count, and utilization.
    4. Identify the most utilized machine as bottleneck.
    5. Compute throughput metrics using total demand and makespan.
    6. Return full KPI payload.
    """
    try:
        # Step 1: Load all schedule rows and short-circuit when empty.
        schedules = ProductionSchedule.objects.all()
        if not schedules.exists():
            return Response({'message': 'No schedules available'}, status=status.HTTP_200_OK)

        # Step 2: Compute total makespan of the persisted schedule.
        min_start      = schedules.aggregate(Min('start_time'))['start_time__min']
        max_end        = schedules.aggregate(Max('end_time'))['end_time__max']
        makespan_hours = (max_end - min_start).total_seconds() / 3600
        makespan_days  = makespan_hours / 24

        machines      = Machine.objects.all()
        machine_stats = []

        # Step 3: Build machine-level utilization metrics.
        for machine in machines:
            used_hours     = schedules.filter(machine=machine).aggregate(
                total=Sum('duration_hours')
            )['total'] or 0
            num_operations = schedules.filter(machine=machine).count()
            utilization    = (used_hours / makespan_hours * 100) if makespan_hours > 0 else 0

            machine_stats.append({
                'machine':        machine.name,
                'used_hours':     round(used_hours, 2),
                'utilization':    round(utilization, 2),
                'num_operations': num_operations
            })

        machine_stats_sorted = sorted(machine_stats, key=lambda x: x['utilization'], reverse=True)

        # Step 4: Top utilization machine is treated as current bottleneck.
        bottleneck = None
        if machine_stats_sorted:
            bottleneck = {
                'machine':     machine_stats_sorted[0]['machine'],
                'utilization': machine_stats_sorted[0]['utilization'],
                'used_hours':  machine_stats_sorted[0]['used_hours']
            }

        # Step 5: Compute total throughput from demand and makespan.
        total_units         = Product.objects.filter(demand_2024__gt=0).aggregate(
            total=Sum('demand_2024')
        )['total'] or 0
        throughput_per_day  = (total_units / makespan_days)  if makespan_days  > 0 else 0
        throughput_per_hour = (total_units / makespan_hours) if makespan_hours > 0 else 0

        # Step 6: Return KPI response.
        return Response({
            'total_makespan_hours':     round(makespan_hours,      2),
            'total_makespan_days':      round(makespan_days,       2),
            'machine_utilization':      machine_stats_sorted,
            'bottleneck':               bottleneck,
            'total_operations':         schedules.count(),
            'throughput_units_per_day':  round(throughput_per_day,  2),
            'throughput_units_per_hour': round(throughput_per_hour, 2),
            'total_units_scheduled':    total_units
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# TASK STATUS
# ===========================================================================

@api_view(['GET'])
def get_task_status(request, task_id):
    """
    Return current status/progress for a Celery background task.

    Step-by-step:
    1. Resolve the task by id.
    2. Build a common response envelope.
    3. Map Celery states to user-facing progress/status fields.
    4. Return standardized task response.
    """
    try:
        # Step 1: Lookup task state and metadata from Celery backend.
        task_result = AsyncResult(task_id)
        # Step 2: Start response with generic identifiers.
        response    = {'task_id': task_id, 'state': task_result.state}

        # Step 3: Translate state into API-friendly status object.
        if task_result.state == 'PENDING':
            response.update({'progress': 0,   'status': 'Task is waiting to start...'})
        elif task_result.state == 'PROGRESS':
            response.update({
                'progress': task_result.info.get('progress', 0),
                'status':   task_result.info.get('status', 'Processing...')
            })
        elif task_result.state == 'SUCCESS':
            response.update({'progress': 100, 'status': 'Complete', 'result': task_result.result})
        elif task_result.state == 'FAILURE':
            response.update({'progress': 0,   'status': 'Task failed', 'error': str(task_result.info)})
        else:
            response.update({'progress': 0,   'status': task_result.state})

        # Step 4: Return normalized payload.
        return Response(response, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# DATA INITIALISATION
# ===========================================================================

@api_view(['POST'])
def initialize_data(request):
    """
    Initialize master data tables from uploaded Frontpage and Process CSVs.

    Step-by-step:
    1. Validate required files are present in the request.
    2. Choose async or sync execution path.
    3. Async path saves files temporarily and enqueues Celery initialization.
    4. Sync path parses CSVs and converts them into normalized structures.
    5. Clear existing model data to rebuild a clean dataset.
    6. Create machine records.
    7. Create product records and link process steps per product.
    8. Return created-record counters.
    """
    try:
        # Step 1: Validate required file uploads.
        if 'frontpage' not in request.FILES or 'process' not in request.FILES:
            return Response(
                {'error': 'Both Frontpage.csv and Process.csv files are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        frontpage_file = request.FILES['frontpage']
        process_file   = request.FILES['process']
        async_mode     = request.POST.get('async_mode', 'true').lower() == 'true'

        # Step 2-3: Async path for long-running initialization workloads.
        if async_mode:
            temp_dir       = tempfile.mkdtemp()
            frontpage_path = os.path.join(temp_dir, 'frontpage.csv')
            process_path   = os.path.join(temp_dir, 'process.csv')

            with open(frontpage_path, 'wb+') as f:
                for chunk in frontpage_file.chunks():
                    f.write(chunk)
            with open(process_path, 'wb+') as f:
                for chunk in process_file.chunks():
                    f.write(chunk)

            task = initialize_data_task.delay(frontpage_path, process_path)

            return Response({
                'task_id':          task.id,
                'status':           'processing',
                'message':          'Data initialization started in background',
                'check_status_url': f'/api/task-status/{task.id}/'
            }, status=status.HTTP_202_ACCEPTED)

        # ── Sync fallback ─────────────────────────────────────────────
        # Step 4: Sync fallback parses CSV files directly in request lifecycle.
        try:
            frontpage_df = pd.read_csv(frontpage_file)
            process_df   = pd.read_csv(process_file)
            process_df   = process_df.iloc[:, :-2]
        except Exception as e:
            return Response({'error': f'Error reading CSV files: {str(e)}'}, status=400)

        # Step 4 (continued): Normalize source CSVs into domain-specific structures.
        demand_data                    = process_frontpage_data(frontpage_df)
        process_routing, machines_list = process_routing_data(process_df)

        # Step 5: Reset existing rows before rebuilding from source files.
        Product.objects.all().delete()
        Machine.objects.all().delete()
        ProcessStep.objects.all().delete()
        ProductionSchedule.objects.all().delete()

        # Step 6: Create and cache machine objects by machine name.
        machines = {}
        for machine_name in machines_list:
            if machine_name:
                machines[machine_name] = Machine.objects.create(
                    name=machine_name, available_hours_per_day=24
                )

        # Step 7: Create products and attach all matching process steps.
        for i in range(len(demand_data['Item'])):
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

        # Step 8: Return import summary counters.
        return Response({
            'message':               'Database initialized successfully',
            'products_created':      Product.objects.count(),
            'machines_created':      Machine.objects.count(),
            'process_steps_created': ProcessStep.objects.count()
        }, status=status.HTTP_201_CREATED)

    except Exception as e:
        import traceback; traceback.print_exc()
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# BUFFER OPTIMISATION
# ===========================================================================

@api_view(['GET'])
def get_buffer_optimization(request):
    """
    Recommend buffer sizes per machine using schedule statistics.

    Step-by-step:
    1. Validate schedule availability.
    2. Read optimization parameters (safety factor and optional budget).
    3. Compute makespan and throughput baseline.
    4. Compute required buffer per machine from average operation duration.
    5. Optionally optimize allocation under budget constraints.
    6. Return ranked recommendations and parameter context.
    """
    try:
        # Step 1: Ensure schedule data exists.
        schedules = ProductionSchedule.objects.all()
        if not schedules.exists():
            return Response({'message': 'No schedules available'}, status=status.HTTP_200_OK)

        # Step 2: Parse request parameters used by buffer sizing logic.
        safety_factor = float(request.GET.get('safety_factor', 1.5))
        total_budget  = request.GET.get('total_budget')

        # Step 3: Compute makespan and throughput baseline.
        min_start      = schedules.aggregate(Min('start_time'))['start_time__min']
        max_end        = schedules.aggregate(Max('end_time'))['end_time__max']
        makespan_hours = (max_end - min_start).total_seconds() / 3600

        total_units         = Product.objects.filter(demand_2024__gt=0).aggregate(
            total=Sum('demand_2024')
        )['total'] or 0
        throughput_per_hour = (total_units / makespan_hours) if makespan_hours > 0 else 0

        machines    = Machine.objects.all()
        buffer_data = []

        # Step 4: Build one recommendation row per active machine.
        for machine in machines:
            machine_schedules = schedules.filter(machine=machine)
            if not machine_schedules.exists():
                continue

            avg_duration_hours = machine_schedules.aggregate(avg=Avg('duration_hours'))['avg'] or 0
            used_hours         = machine_schedules.aggregate(total=Sum('duration_hours'))['total'] or 0
            utilization        = (used_hours / makespan_hours * 100) if makespan_hours > 0 else 0
            required_buffer    = throughput_per_hour * avg_duration_hours * safety_factor

            buffer_data.append({
                'machine':                  machine.name,
                'required_buffer':          round(required_buffer, 2),
                'buffer_size_units':        round(required_buffer, 2),
                'avg_operation_time_hours': round(avg_duration_hours, 4),
                'throughput_per_hour':      round(throughput_per_hour, 2),
                'safety_factor':           safety_factor,
                'utilization':             round(utilization, 2),
                'total_operations':        machine_schedules.count(),
                'recommendation':          (
                    'HIGH PRIORITY'   if utilization > 80 else
                    'MEDIUM PRIORITY' if utilization > 60 else
                    'LOW PRIORITY'
                ),
            })

        # Step 5: Apply LP-based constrained allocation when budget is provided.
        if total_budget is not None:
            total_budget = float(total_budget)
            buffer_data  = pulp_optimize_buffers(buffer_data, total_budget)

        # Step 6: Sort worst-to-best by required buffer for UI consumption.
        buffer_data.sort(key=lambda x: x['required_buffer'], reverse=True)

        response_payload = {
            'buffer_recommendations': buffer_data,
            'parameters': {
                'throughput_per_hour': round(throughput_per_hour, 2),
                'makespan_hours':      round(makespan_hours, 2),
                'safety_factor':       safety_factor,
                'total_units':         total_units,
            },
            'formula': 'required_buffer = throughput_per_hour × avg_operation_time_hours × safety_factor',
        }

        if total_budget is not None:
            response_payload['pulp_allocation'] = {
                'total_budget':    total_budget,
                'total_required':  round(sum(d['required_buffer']             for d in buffer_data), 2),
                'total_allocated': round(sum(d.get('allocated_buffer', 0) for d in buffer_data), 2),
            }

        return Response(response_payload, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# BOTTLENECK ANALYSIS
# ===========================================================================

@api_view(['GET'])
def get_bottleneck_analysis(request):
    """
    Provide machine-level bottleneck diagnostics with recommendations.

    Step-by-step:
    1. Validate schedule availability.
    2. Compute global makespan.
    3. Compute utilization/idle metrics per machine.
    4. Classify each machine by utilization band.
    5. Rank machines and return summary plus detail rows.
    """
    try:
        # Step 1: Ensure schedule data exists before computing KPIs.
        schedules = ProductionSchedule.objects.all()
        if not schedules.exists():
            return Response({'message': 'No schedules available'}, status=status.HTTP_200_OK)

        # Step 2: Compute the total schedule window.
        min_start      = schedules.aggregate(Min('start_time'))['start_time__min']
        max_end        = schedules.aggregate(Max('end_time'))['end_time__max']
        makespan_hours = (max_end - min_start).total_seconds() / 3600

        machines             = Machine.objects.all()
        bottleneck_analysis  = []

        # Step 3-4: Build per-machine diagnostic metrics and classification.
        for machine in machines:
            machine_schedules = schedules.filter(machine=machine)
            if not machine_schedules.exists():
                continue

            used_hours         = machine_schedules.aggregate(total=Sum('duration_hours'))['total'] or 0
            utilization        = (used_hours / makespan_hours * 100) if makespan_hours > 0 else 0
            num_operations     = machine_schedules.count()
            avg_operation_time = machine_schedules.aggregate(avg=Avg('duration_hours'))['avg'] or 0
            idle_hours         = makespan_hours - used_hours
            idle_percentage    = (idle_hours / makespan_hours * 100) if makespan_hours > 0 else 0

            products_on_machine = machine_schedules.values_list(
                'product__item', flat=True
            ).distinct().count()

            if utilization >= 85:
                status_label   = 'CRITICAL BOTTLENECK'
                recommendation = 'Consider adding capacity, optimizing setups, or redistributing work'
            elif utilization >= 70:
                status_label   = 'POTENTIAL BOTTLENECK'
                recommendation = 'Monitor closely, consider process improvements'
            elif utilization >= 50:
                status_label   = 'WELL UTILIZED'
                recommendation = 'Operating efficiently'
            else:
                status_label   = 'UNDERUTILIZED'
                recommendation = 'Opportunity to consolidate operations or reduce capacity'

            bottleneck_analysis.append({
                'machine':                  machine.name,
                'utilization':              round(utilization, 2),
                'used_hours':               round(used_hours, 2),
                'idle_hours':               round(idle_hours, 2),
                'idle_percentage':          round(idle_percentage, 2),
                'num_operations':           num_operations,
                'avg_operation_time_hours': round(avg_operation_time, 4),
                'products_processed':       products_on_machine,
                'status':                   status_label,
                'recommendation':           recommendation,
            })

        # Step 5: Rank by utilization and compute summary headline metrics.
        bottleneck_analysis.sort(key=lambda x: x['utilization'], reverse=True)

        summary = {
            'total_makespan_hours':   round(makespan_hours, 2),
            'bottleneck_machine':     bottleneck_analysis[0]['machine']      if bottleneck_analysis else None,
            'bottleneck_utilization': bottleneck_analysis[0]['utilization']  if bottleneck_analysis else 0,
            'avg_utilization':        round(np.mean([m['utilization'] for m in bottleneck_analysis]), 2) if bottleneck_analysis else 0,
            'total_machines':         len(bottleneck_analysis),
        }

        return Response({'summary': summary, 'machine_analysis': bottleneck_analysis},
                        status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# BATCH OPTIMISATION PREVIEW
# ===========================================================================

def _build_params(request_obj, is_post=False):
    """
    Extract optimization parameters from request and cast to integers.

    Step-by-step:
    1. Select source collection based on request type.
    2. Read values with safe defaults.
    3. Convert to integers for deterministic calculations.
    """
    # Step 1: Choose request payload source (`data` vs query params).
    getter = request_obj.data if is_post else request_obj.GET
    # Step 2-3: Read values and cast for downstream math.
    return {
        "max_num_batches": int(getter.get("max_num_batches", 25)),
        "min_batch_size":  int(getter.get("min_batch_size",  50)),
        "max_batch_size":  int(getter.get("max_batch_size",  500)),
    }


@api_view(["GET"])
def get_batch_optimization_preview(request):
    """
    Preview optimized batch parameters for all products without DB writes.

    Step-by-step:
    1. Parse optimization parameters and execution mode.
    2. If async is enabled, enqueue background preview task.
    3. In sync mode, iterate all products with positive demand.
    4. Compute optimized batch size/num batches per product.
    5. Compare against current settings and build analysis rows.
    6. Compute aggregate summary metrics and return response.
    """
    try:
        # Step 1: Parse validated params and mode toggle.
        params     = _build_params(request)
        async_mode = request.GET.get("async_mode", "false").lower() == "true"

        # Step 2: Async path returns task metadata immediately.
        if async_mode:
            task = batch_optimize_preview_task.delay(params)
            return Response({
                "task_id":          task.id,
                "status":           "processing",
                "message":          "Batch optimization preview started in background",
                "check_status_url": f"/api/task-status/{task.id}/",
            }, status=status.HTTP_202_ACCEPTED)

        # ── Sync ──────────────────────────────────────────────────────────────
        # Step 3: Sync path computes previews inline for each product.
        products       = Product.objects.filter(demand_2024__gt=0)
        batch_analysis = []
        total_demand   = 0
        total_batches  = 0
        batch_sizes    = []

        # Step 4-5: Evaluate optimized values and prepare per-product output.
        for product in products:
            demand = int(product.demand_2024)

            batch_size, num_batches, ideal_batch = calculate_optimal_batch_size(
                demand,
                params["max_num_batches"],
                params["min_batch_size"],
                params["max_batch_size"],
            )

            old_batch_size  = product.batch_size   # actual DB value (not re-computed)
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
                # NEW: surface the effective bounds so UI can explain them
                "effective_min_batch": max(
                    params["min_batch_size"],
                    -(-demand // params["max_num_batches"])   # ceil
                ),
                "effective_max_batch": min(params["max_batch_size"], demand)
                    if params["max_batch_size"] <= demand else demand,
            })

            total_demand  += demand
            total_batches += num_batches
            batch_sizes.append(batch_size)

        # Step 6: Return full analysis and aggregate summary statistics.
        return Response({
            "batch_analysis": batch_analysis,
            "summary": {
                "total_products": len(batch_analysis),
                "total_demand":   total_demand,
                "total_batches":  total_batches,
                "avg_batch_size": round(np.mean(batch_sizes), 2) if batch_sizes else 0,
                "min_batch_size": int(np.min(batch_sizes))       if batch_sizes else 0,
                "max_batch_size": int(np.max(batch_sizes))       if batch_sizes else 0,
                "std_batch_size": round(np.std(batch_sizes), 2)  if batch_sizes else 0,
                "note": (
                    "Bounds are applied adaptively per product. "
                    "For small demands the max bound is clamped to the demand; "
                    "for large demands the min bound is raised to ceil(demand/max_batches)."
                ),
            },
            "parameters": params,
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ===========================================================================
# GENERATE SCHEDULE
# ===========================================================================

@api_view(['POST'])
def generate_schedule(request):
    """
    Generate production schedule either asynchronously or synchronously.

    Request body
    ------------
    start_date         : "YYYY-MM-DD"   (default today)
    local_opt_machines : int            (default 5)
    enable_compaction  : bool           gap elimination ON/OFF (default true) ← NEW
    clear_existing     : bool           (default true)
    product_pks        : [int]          (default: all)
    batch_overrides    : [[pk,bs,nb],…] (optional)
    async_mode         : bool           (default true)
    Step-by-step:
    1. Validate prerequisite master data (products + process steps).
    2. Parse scheduler parameters from request body.
    3. If async mode, dispatch background task and return task metadata.
    4. If sync mode, compute start datetime and run scheduler inline.
    5. Compute schedule KPIs and return operation count + KPIs.
    """
    try:
        # Step 1: Validate required master data exists before scheduling.
        if not Product.objects.filter(demand_2024__gt=0).exists():
            return Response(
                {'error': 'No products found. Please run Initialize Data first.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        if not ProcessStep.objects.exists():
            return Response(
                {'error': 'No process steps found. Please run Initialize Data first.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Step 2: Build scheduler parameter dict from request data.
        params = {
            'start_date':          request.data.get('start_date'),
            'local_opt_machines':  int(request.data.get('local_opt_machines', 5)),
            'enable_compaction':   bool(request.data.get('enable_compaction', True)),  # ← NEW
            'clear_existing':      request.data.get('clear_existing', True),
            'product_pks':         request.data.get('product_pks'),
            'batch_overrides':     request.data.get('batch_overrides', []),
        }

        async_mode = str(request.data.get('async_mode', 'true')).lower() == 'true'

        # Step 3: Async path delegates heavy scheduling to Celery.
        if async_mode:
            task = generate_schedule_task.delay(params)
            return Response({
                'task_id':          task.id,
                'status':           'processing',
                'message':          'Schedule generation started in background.',
                'check_status_url': f'/api/task-status/{task.id}/',
            }, status=status.HTTP_202_ACCEPTED)

        # Step 4: Sync fallback runs scheduler in-request.
        from .utils import run_job_shop_scheduler, compute_schedule_kpis
        from datetime import datetime

        start_str = params.get('start_date')
        start_dt  = datetime.fromisoformat(start_str) if start_str else datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        products = list(Product.objects.filter(demand_2024__gt=0))
        rows     = run_job_shop_scheduler(
            products,
            start_dt,
            local_opt_machines = params['local_opt_machines'],
            enable_compaction  = params['enable_compaction'],   # ← NEW
        )
        # Step 5: Compute KPI summary from generated operation rows.
        makespan = max(r['end_hrs'] for r in rows) if rows else 0
        kpis     = compute_schedule_kpis(rows, makespan)

        return Response(
            {'status': 'success', 'kpis': kpis, 'total_operations': len(rows)},
            status=status.HTTP_201_CREATED
        )

    except Exception as e:
        import traceback; traceback.print_exc()
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# GET SCHEDULE  (paginated)
# ===========================================================================

@api_view(['GET'])
def get_schedule(request):
    """
    Retrieve persisted schedule rows with filtering and pagination.

    Step-by-step:
    1. Start from base queryset with related objects for response shaping.
    2. Apply backward-compatible single-value filters.
    3. Apply multi-value filters (machines/products lists).
    4. Apply date, batch-size, duration, step, and DCC filters.
    5. Apply pagination and output format conversion.
    6. Return paginated metadata and row payload.
    """
    try:
        # Step 1: Build base queryset with related objects and deterministic order.
        schedules = ProductionSchedule.objects.select_related(
            'machine', 'product', 'process_step'
        ).order_by('start_time', 'machine')

        # Step 2: Backward-compatible single machine/product filters.
        machine_filter = request.GET.get('machine')
        product_filter = request.GET.get('product')
        if machine_filter:
            schedules = schedules.filter(machine__name__icontains=machine_filter)
        if product_filter:
            try:
                schedules = schedules.filter(product__item=int(product_filter))
            except (ValueError, TypeError):
                pass

        # Step 3: Comma-separated multi machine/product filters.
        machines_param = request.GET.get('machines')
        if machines_param:
            names = [m.strip() for m in machines_param.split(',') if m.strip()]
            if names:
                schedules = schedules.filter(machine__name__in=names)
        products_param = request.GET.get('products')
        if products_param:
            try:
                items = [int(x.strip()) for x in products_param.split(',') if x.strip()]
                if items:
                    schedules = schedules.filter(product__item__in=items)
            except (ValueError, TypeError):
                pass

        # Step 4: Date range filters.
        start_param = request.GET.get('start')
        end_param = request.GET.get('end')
        if start_param:
            schedules = schedules.filter(start_time__gte=start_param)
        if end_param:
            schedules = schedules.filter(end_time__lte=end_param)

        # Step 4 (continued): Batch size range filters.
        batch_min = request.GET.get('batch_size_min')
        batch_max = request.GET.get('batch_size_max')
        if batch_min is not None and batch_min != '':
            try:
                schedules = schedules.filter(batch_size__gte=int(batch_min))
            except (ValueError, TypeError):
                pass
        if batch_max is not None and batch_max != '':
            try:
                schedules = schedules.filter(batch_size__lte=int(batch_max))
            except (ValueError, TypeError):
                pass

        # Step 4 (continued): Duration range filters.
        dur_min = request.GET.get('duration_min')
        dur_max = request.GET.get('duration_max')
        if dur_min is not None and dur_min != '':
            try:
                schedules = schedules.filter(duration_hours__gte=float(dur_min))
            except (ValueError, TypeError):
                pass
        if dur_max is not None and dur_max != '':
            try:
                schedules = schedules.filter(duration_hours__lte=float(dur_max))
            except (ValueError, TypeError):
                pass

        # Step 4 (continued): Specific process-step filter.
        step_param = request.GET.get('step')
        if step_param is not None and step_param != '':
            try:
                step_num = int(step_param)
                schedules = schedules.filter(process_step__step_number=step_num)
            except (ValueError, TypeError):
                pass

        # Step 4 (continued): Product-level DCC type filter.
        dcc_param = request.GET.get('dcc_type')
        if dcc_param and dcc_param.strip():
            schedules = schedules.filter(product__dcc_type=dcc_param.strip())

        # Step 5: Pagination controls with safe bounds.
        total_count = schedules.count()
        page_size   = min(int(request.GET.get('page_size', 200)), 1000)
        page        = max(int(request.GET.get('page', 1)), 1)
        offset      = (page - 1) * page_size
        schedules   = schedules[offset: offset + page_size]

        # Step 5 (continued): Choose response schema (`gantt` vs `full`).
        out_format = request.GET.get('format', 'full')

        if out_format == 'gantt':
            rows = [
                {
                    'machine':    s.machine.name,
                    'product':    s.product.item,
                    'batch_id':   s.batch_id,
                    'batch_num':  s.batch_num,
                    'step':       s.process_step.step_number,
                    'step_name':  s.process_step.step_name,
                    'start':      s.start_time.isoformat(),
                    'end':        s.end_time.isoformat(),
                    'duration_h': round(s.duration_hours, 4),
                }
                for s in schedules
            ]
        else:
            rows = [
                {
                    'id':          s.pk,
                    'machine':     s.machine.name,
                    'product':     s.product.item,
                    'description': s.product.description[:50],
                    'batch_id':    s.batch_id,
                    'batch_num':   s.batch_num,
                    'batch_size':  s.batch_size,
                    'step':        s.process_step.step_number,
                    'step_name':   s.process_step.step_name,
                    'start':       s.start_time.isoformat(),
                    'end':         s.end_time.isoformat(),
                    'duration_h':  round(s.duration_hours, 4),
                }
                for s in schedules
            ]

        # Step 6: Return paginated payload and metadata.
        return Response({
            'count':       total_count,
            'page':        page,
            'page_size':   page_size,
            'total_pages': max(1, (total_count + page_size - 1) // page_size),
            'results':     rows,
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# GANTT DATA
# ===========================================================================

@api_view(['GET'])
def get_schedule_gantt(request):
    """
    Return schedule rows in a Gantt-friendly shape.

    Step-by-step:
    1. Start with schedule queryset and optional machine/date filtering.
    2. Early-return empty payload when no schedule rows exist.
    3. Apply bar-limit truncation for frontend performance.
    4. Build deterministic product color keys.
    5. Build gantt bars and machine ordering metadata.
    6. Return bars, date range, and truncation info.
    """
    try:
        # Step 1: Base queryset with related models for rendering fields.
        schedules = ProductionSchedule.objects.select_related(
            'machine', 'product', 'process_step'
        )

        # Step 1 (continued): Optional machine filter.
        machines_param = request.GET.get('machines')
        if machines_param:
            machine_names = [m.strip() for m in machines_param.split(',')]
            schedules = schedules.filter(machine__name__in=machine_names)

        # Step 1 (continued): Optional date-window filters.
        start_param = request.GET.get('start')
        end_param   = request.GET.get('end')
        if start_param:
            schedules = schedules.filter(start_time__gte=start_param)
        if end_param:
            schedules = schedules.filter(end_time__lte=end_param)

        # Step 2: Return canonical empty response when no data is found.
        if not schedules.exists():
            return Response({'gantt_bars': [], 'machines': [], 'date_range': {}, 'total_bars': 0})

        # Step 3: Limit output size and keep stable machine/time ordering.
        total_bars = schedules.count()
        max_bars   = int(request.GET.get('max_bars', 500))
        schedules  = schedules.order_by('machine__name', 'start_time')[:max_bars]

        # Step 4: Build stable product->color-key mapping.
        product_items = list(
            ProductionSchedule.objects.values_list('product__item', flat=True).distinct()
        )
        color_map = {item: idx % 20 for idx, item in enumerate(sorted(product_items))}

        # Step 5: Convert schedule rows to Gantt bar schema.
        gantt_bars = [
            {
                'machine':    s.machine.name,
                'product':    s.product.item,
                'batch_id':   s.batch_id,
                'batch_num':  s.batch_num,
                'step':       s.process_step.step_number,
                'step_name':  s.process_step.step_name,
                'start':      s.start_time.isoformat(),
                'end':        s.end_time.isoformat(),
                'duration_h': round(s.duration_hours, 4),
                'color_key':  color_map.get(s.product.item, 0),
            }
            for s in schedules
        ]

        # Step 5 (continued): Build machine display order by descending load.
        machine_loads = (
            ProductionSchedule.objects
            .values('machine__name')
            .annotate(load=Sum('duration_hours'))
            .order_by('-load')
        )
        machine_order = [m['machine__name'] for m in machine_loads]

        agg = ProductionSchedule.objects.aggregate(
            start=Min('start_time'), end=Max('end_time')
        )

        # Step 6: Return chart-ready payload with metadata.
        return Response({
            'gantt_bars': gantt_bars,
            'machines':   machine_order,
            'date_range': {
                'start': agg['start'].isoformat() if agg['start'] else None,
                'end':   agg['end'].isoformat()   if agg['end']   else None,
            },
            'total_bars': total_bars,
            'truncated':  total_bars > max_bars,
        })

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# GAP ANALYSIS                                                          ← NEW
# ===========================================================================

@api_view(['GET'])
def gap_analysis(request):
    """
    GET /api/gap-analysis/

    Analyse machine idle gaps between consecutive operations.

    Response
    --------
    {
      "total_gaps":       12,
      "total_idle_hours": 8.4,
      "worst_machine":    "Cutting Automation",
      "clean_machines":   3,
      "machine_stats": [
          {"machine": "...", "idle_hours": 2.1, "gap_count": 4}, ...
      ],
      "gap_list": [
          {
            "machine":    "Cutting Automation",
            "gap_start":  "2026-02-20T03:45:00",
            "gap_end":    "2026-02-20T05:32:00",
            "idle_hours": 1.78,
            "cause":      "precedence",
            "before_op":  "P3 Step 31",
            "after_op":   "P3 Step 67"
          }, ...
      ],
      "causes": {"precedence": 7, "interleave": 4, "spt": 1}
    }

    Gap cause heuristic
    -------------------
    "interleave"  – consecutive ops are from different products
                    (batch-interleaving gap)
    "precedence"  – same product, step > 1
                    (upstream step on different machine not yet done)
    "spt"         – same product, first step
                    (SPT reordering artefact)
    Step-by-step:
    1. Load schedule rows ordered by machine and start time.
    2. Group operations by machine.
    3. Scan adjacent operations and detect meaningful idle gaps.
    4. Classify each gap cause using lightweight heuristics.
    5. Aggregate machine statistics and global totals.
    6. Return gap list, machine stats, and cause counters.
    """
    try:
        # Step 1: Fetch schedule rows in machine/time order for gap detection.
        schedules = (
            ProductionSchedule.objects
            .select_related('machine', 'product', 'process_step')
            .order_by('machine__name', 'start_time')
        )

        # Step 1: Abort with explicit message when no schedule is available.
        if not schedules.exists():
            return Response(
                {'detail': 'No schedule found. Generate a schedule first.'},
                status=status.HTTP_404_NOT_FOUND
            )

        # Step 2: Group operations by machine name.
        machine_ops: dict = defaultdict(list)
        for s in schedules:
            machine_ops[s.machine.name].append(s)

        gap_list      = []
        machine_stats = []
        causes        = {'precedence': 0, 'interleave': 0, 'spt': 0}

        # Step 3-4: Walk each machine timeline and classify detected gaps.
        for machine_name, mops in machine_ops.items():
            # Already ordered by start_time from the queryset
            idle_hours = 0.0
            gap_count  = 0

            # Iterate adjacent operations to measure idle intervals.
            for i in range(1, len(mops)):
                prev     = mops[i - 1]
                curr     = mops[i]
                gap_secs = (curr.start_time - prev.end_time).total_seconds()

                # Ignore tiny timing differences and track only genuine idle gaps.
                if gap_secs > 60:       # > 1 minute = genuine idle gap
                    idle_h     = gap_secs / 3600
                    idle_hours += idle_h
                    gap_count  += 1

                    # Step 4: Assign a coarse cause label for root-cause analysis.
                    prev_prod = str(prev.product.item)
                    curr_prod = str(curr.product.item)

                    if curr_prod != prev_prod:
                        cause = 'interleave'
                        causes['interleave'] += 1
                    elif curr.process_step.step_number > 1:
                        cause = 'precedence'
                        causes['precedence'] += 1
                    else:
                        cause = 'spt'
                        causes['spt'] += 1

                    gap_list.append({
                        'machine':    machine_name,
                        'gap_start':  prev.end_time.isoformat(),
                        'gap_end':    curr.start_time.isoformat(),
                        'idle_hours': round(idle_h, 3),
                        'cause':      cause,
                        'before_op':  f"P{prev_prod} Step {prev.process_step.step_number}",
                        'after_op':   f"P{curr_prod} Step {curr.process_step.step_number}",
                    })

            machine_stats.append({
                'machine':    machine_name,
                'idle_hours': round(idle_hours, 3),
                'gap_count':  gap_count,
            })

        # Sort gaps worst-first
        # Step 5: Rank for "worst first" dashboards and aggregate headline metrics.
        gap_list.sort(key=lambda g: g['idle_hours'], reverse=True)
        machine_stats.sort(key=lambda m: m['idle_hours'], reverse=True)

        worst = machine_stats[0]['machine'] if machine_stats else '—'
        clean = sum(1 for m in machine_stats if m['gap_count'] == 0)

        # Step 6: Return complete analysis payload.
        return Response({
            'total_gaps':       len(gap_list),
            'total_idle_hours': round(sum(g['idle_hours'] for g in gap_list), 2),
            'worst_machine':    worst,
            'clean_machines':   clean,
            'machine_stats':    machine_stats,
            'gap_list':         gap_list,
            'causes':           causes,
        }, status=status.HTTP_200_OK)

    except Exception as e:
        import traceback; traceback.print_exc()
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# SCHEDULE COMPARISON
# ===========================================================================

@api_view(['GET'])
def schedule_comparison(request):
    """
    Return baseline KPI snapshot for the currently saved schedule.

    Step-by-step:
    1. Validate schedule existence.
    2. Aggregate top-level makespan/operations/units metrics.
    3. Aggregate machine-level load and operation counts.
    4. Compute utilization percentages and average utilization.
    5. Return normalized comparison payload.
    """
    try:
        # Step 1: Load schedules and stop early when no data exists.
        schedules = ProductionSchedule.objects.select_related('machine', 'product')
        if not schedules.exists():
            return Response({'message': 'No schedule available.'}, status=200)

        # Step 2: Compute global schedule aggregates.
        agg = schedules.aggregate(
            start=Min('start_time'), end=Max('end_time'),
            total_ops=Count('id'), total_units=Sum('batch_size')
        )
        makespan = (agg['end'] - agg['start']).total_seconds() / 3600 if agg['end'] else 0

        # Step 3: Aggregate per-machine load and operation counts.
        machine_stats = (
            schedules.values('machine__name')
            .annotate(load=Sum('duration_hours'), ops=Count('id'))
            .order_by('-load')
        )

        # Step 4: Convert machine aggregate rows to API response objects.
        stats = [
            {
                'machine':     m['machine__name'],
                'used_hours':  round(m['load'], 2),
                'utilization': round(m['load'] / makespan * 100, 2) if makespan else 0,
                'operations':  m['ops'],
            }
            for m in machine_stats
        ]

        # Step 5: Return current-schedule KPI payload.
        return Response({
            'current_schedule': {
                'makespan_hours':   round(makespan, 2),
                'makespan_days':    round(makespan / 24, 2),
                'total_operations': agg['total_ops'],
                'total_units':      agg['total_units'],
                'machine_stats':    stats,
                'avg_utilization':  round(
                    sum(s['utilization'] for s in stats) / len(stats), 2
                ) if stats else 0,
            }
        })

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# OPTIMIZE SCHEDULE PREVIEW & SAVE
# ===========================================================================

@api_view(['GET'])
def optimize_schedule_preview(request):
    """
    Preview scheduler KPIs without persisting schedule rows.

    Step-by-step:
    1. Parse local optimization parameter and fetch active products.
    2. Validate product availability.
    3. Run scheduler in memory.
    4. Compute and return preview KPIs.
    """
    try:
        # Step 1: Parse scheduler tuning parameter and load active products.
        local_opt = int(request.GET.get('local_opt_machines', 5))
        products  = list(Product.objects.filter(demand_2024__gt=0))

        # Step 2: Guard against empty product data.
        if not products:
            return Response({'error': 'No products in DB.'}, status=400)

        from .utils import run_job_shop_scheduler, compute_schedule_kpis
        from datetime import datetime

        # Step 3: Generate in-memory schedule preview rows.
        rows     = run_job_shop_scheduler(products, datetime.now(), local_opt_machines=local_opt)
        # Step 4: Build KPI snapshot from preview rows.
        makespan = max(r['end_hrs'] for r in rows) if rows else 0
        kpis     = compute_schedule_kpis(rows, makespan)

        return Response({'preview_kpis': kpis, 'total_operations': len(rows)})

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def optimize_schedule_save(request):
    """
    Force schedule re-optimization and persistence via `generate_schedule`.

    Step-by-step:
    1. Force schedule replacement (`clear_existing=True`).
    2. Normalize local optimization parameter.
    3. Delegate to `generate_schedule`.
    """
    # Step 1: Always clear old rows when invoking this endpoint.
    request.data['clear_existing']     = True
    # Step 2: Ensure integer type before delegation.
    request.data['local_opt_machines'] = int(request.data.get('local_opt_machines', 5))
    # Step 3: Reuse shared scheduling implementation.
    return generate_schedule(request)


@api_view(["POST"])
def save_batch_optimization(request):
    """
    Apply independently optimized batch settings and persist to DB.

    Step-by-step:
    1. Parse optimization params and async toggle.
    2. Async path enqueues save task.
    3. Sync path loops through active products.
    4. Compute optimal settings for each product.
    5. Persist only changed rows; track skipped items.
    6. Return update summary.
    """
    try:
        # Step 1: Parse parameter bounds and async mode.
        params     = _build_params(request, is_post=True)
        async_mode = request.data.get("async_mode", "false").lower() == "true"

        # Step 2: Async save path for long-running updates.
        if async_mode:
            task = batch_optimize_save_task.delay(params)
            return Response({
                "task_id":          task.id,
                "status":           "processing",
                "message":          "Batch optimization save started in background",
                "check_status_url": f"/api/task-status/{task.id}/",
            }, status=status.HTTP_202_ACCEPTED)

        # Step 3: Sync path applies updates in request lifecycle.
        products = Product.objects.filter(demand_2024__gt=0)
        updated  = []
        skipped  = []

        # Step 4-5: Compute and persist only changed batch settings.
        for product in products:
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

        # Step 6: Return summary payload of updates and skips.
        return Response({
            "message":          "Batch optimization applied successfully",
            "total_updated":    len(updated),
            "total_skipped":    len(skipped),
            "updated_products": updated,
            "skipped_items":    skipped,
            "parameters":       params,
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(["GET"])
def get_joint_batch_preview(request):
    """
    Preview joint multi-product optimization without writing to DB.

    Step-by-step:
    1. Parse optimization bounds and async mode.
    2. Async path enqueues preview task.
    3. Sync path loads active products.
    4. Run joint optimization preview routine.
    5. Return result matrix, machine loads, and summary.
    """
    try:
        from .utils import _joint_preview_no_save, _build_joint_summary
        from .models import Product

        # Step 1: Parse bounds and execution mode.
        params = {
            "max_num_batches": int(request.GET.get("max_num_batches", 25)),
            "min_batch_size":  int(request.GET.get("min_batch_size",  50)),
            "max_batch_size":  int(request.GET.get("max_batch_size",  500)),
        }
        async_mode = request.GET.get("async_mode", "false").lower() == "true"

        # Step 2: Async preview path for long-running calculations.
        if async_mode:
            from .tasks import joint_optimize_task
            task = joint_optimize_task.delay(params, save_to_db=False)
            return Response({
                "task_id":          task.id,
                "status":           "processing",
                "message":          "Joint optimization preview started in background",
                "check_status_url": f"/api/task-status/{task.id}/",
            }, status=status.HTTP_202_ACCEPTED)

        # ── Simple queryset — NO prefetch_related needed ──────────────────────
        # optimize_product_batches_jointly / _joint_preview_no_save both query
        # ProcessStep internally via ProcessStep.objects.filter(product=product)
        # Step 3: Fetch products included in optimization.
        products = Product.objects.filter(demand_2024__gt=0)

        # Step 4: Compute joint optimization result without DB updates.
        result = _joint_preview_no_save(
            products,
            params["max_num_batches"],
            params["min_batch_size"],
            params["max_batch_size"],
        )

        # Step 5: Return preview payload and computed summary metrics.
        return Response({
            "status":         result["status"],
            "batch_analysis": result["results"],
            "machine_loads":  result["machine_loads"],
            "makespan_proxy": result["makespan_proxy"],
            "summary":        _build_joint_summary(result),
            "parameters":     params,
        }, status=status.HTTP_200_OK)

    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# =============================================================================
# NEW  — POST /api/batch-optimization-joint/
# =============================================================================


@api_view(["POST"])
def save_joint_batch_optimization(request):
    """
    Step-by-step:
    1. Parse optimization bounds and async mode.
    2. Async path enqueues save task.
    3. Sync path loads active products.
    4. Run joint optimization and persist product updates.
    5. Return updated state, load metrics, and summary.
    Joint multi-product batch optimization — saves results to DB.

    Optimises all products simultaneously, minimising the makespan proxy
    (max machine load hours) rather than each product independently.
    """
    try:
        # Step 1: Parse bounds and execution mode, then choose async/sync path.
        from .utils import optimize_product_batches_jointly, _build_joint_summary
        from .models import Product

        params = {
            "max_num_batches": int(request.data.get("max_num_batches", 25)),
            "min_batch_size":  int(request.data.get("min_batch_size",  50)),
            "max_batch_size":  int(request.data.get("max_batch_size",  500)),
        }
        async_mode = request.data.get("async_mode", "false").lower() == "true"

        # Step 2: Async save path for heavy optimization runs.
        if async_mode:
            from .tasks import joint_optimize_task
            task = joint_optimize_task.delay(params, save_to_db=True)
            return Response({
                "task_id":          task.id,
                "status":           "processing",
                "message":          "Joint batch optimization started in background",
                "check_status_url": f"/api/task-status/{task.id}/",
            }, status=status.HTTP_202_ACCEPTED)

        # ── Simple queryset — NO prefetch_related needed ──────────────────────
        # Step 3: Load products participating in optimization.
        products = Product.objects.filter(demand_2024__gt=0)

        # Step 4: Optimize jointly and persist product-level updates.
        result = optimize_product_batches_jointly(
            products,
            params["max_num_batches"],
            params["min_batch_size"],
            params["max_batch_size"],
        )

        # Step 5: Return optimization output with summary and updated counts.
        return Response({
            "status":           result["status"],
            "message":          result["message"],
            "batch_analysis":   result["results"],
            "machine_loads":    result["machine_loads"],
            "makespan_proxy":   result["makespan_proxy"],
            "products_updated": result["products_updated"],
            "summary":          _build_joint_summary(result),
            "parameters":       params,
        }, status=status.HTTP_200_OK)

    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

