from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import numpy as np
from django.db.models import Sum, Count, Q, Max, Min, Avg
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from celery.result import AsyncResult
import os
import tempfile

from .models import Product, Machine, ProcessStep, ProductionSchedule
from .serializers import ProductionScheduleSerializer
from .utils import (
    clean_numeric_columns, clean_text_columns, apply_filters,
    clean_shift_columns, calculate_efficiency, calculate_backlog,
    calculate_production_outputs, SHIFT_LABELS, build_summary,
    get_batch_params, optimize_product_batches_jointly,
    generate_production_schedule, calculate_kpis,
    process_frontpage_data, process_routing_data,
    calculate_optimal_batch_size, pulp_optimize_buffers,
)

# Import Celery tasks
from .tasks import (
    generate_schedule_task,
    initialize_data_task,
    batch_optimize_preview_task
)


# ===========================================================================
# CSV PROCESSING
# ===========================================================================

@api_view(['POST'])
def process_csv(request):
    """
    Process a production planning CSV file and return shift-wise metrics.

    Request Parameters
    ------------------
    file          : CSV file (required)
    num_shifts    : int (optional, default=28)
    pps_tn, project, sub_project, machine, tool_no, area : str (optional)

    Response
    --------
    JSON with ShiftWise metrics and a Summary table.
    """
    csv_file = request.FILES.get('file')
    if not csv_file:
        return Response({'error': 'No file uploaded'}, status=400)

    num_shifts = int(request.POST.get("num_shifts", 28))
    df = pd.read_csv(csv_file)

    clean_numeric_columns(df, ['Planned', 'Realized', 'Backlog', 'Open'])
    clean_text_columns(df, ['Step', 'Area', 'Sub-Project'])
    clean_text_columns(df, ['PPS TN', 'Project', 'Sub-Project', 'Machine', 'Tool No.', 'Area'])

    df = apply_filters(df, {
        'PPS TN':      request.POST.get("pps_tn",        "All"),
        'Project':     request.POST.get("project",       "All"),
        'Sub-Project': request.POST.get("sub_project",   "All"),
        'Machine':     request.POST.get("machine",       "All"),
        'Tool No.':    request.POST.get("tool_no",       "All"),
        'Area':        request.POST.get("area",          "All"),
    })

    clean_shift_columns(df, [(14, 50), (95, 113)])

    efficiency = calculate_efficiency(df, num_shifts)
    backlog    = calculate_backlog(df)

    finished_filter  = (df['Step'] == 'F') & df['Sub-Project'].notna()
    connector_filter = (df['Area'] == 'Assembly') & df['Sub-Project'].notna()

    fg_output, conn_output = calculate_production_outputs(df, finished_filter, connector_filter)

    result = {
        "Total Backlog Finished Goods": dict(zip(SHIFT_LABELS, map(str, backlog))),
        "Production Output Finished Goods": fg_output,
        "Production Output Connectors":     conn_output,
        "Overall Efficiency": {
            shift: f"{eff:.2f}%" if eff > 0 else "-"
            for shift, eff in zip(SHIFT_LABELS, efficiency)
        },
    }

    return Response({"ShiftWise": result, "Summary": build_summary(df)})


@api_view(['POST'])
def get_csv_filter_options(request):
    """Return unique filter values extracted from an uploaded CSV."""
    csv_file = request.FILES.get('file')
    if not csv_file:
        return Response({'error': 'No file uploaded'}, status=400)

    df = pd.read_csv(csv_file)

    filter_columns = ['PPS TN', 'Project', 'Sub-Project', 'Machine', 'Tool No.', 'Area']
    filter_options = {}
    for col in filter_columns:
        if col in df.columns:
            unique_values = df[col].astype(str).str.strip().unique()
            unique_values = sorted(v for v in unique_values if v and v != 'nan')
            filter_options[col] = unique_values
        else:
            filter_options[col] = []

    return Response(filter_options)


# ===========================================================================
# SCHEDULING  –  uses PuLP joint batch optimiser + job-shop scheduler
# ===========================================================================

@api_view(['POST'])
def generate_schedule(request):
    """
    Generate an optimised production schedule (supports async mode).

    Request Body (JSON)
    -------------------
    max_num_batches  : int (default 25)
    min_batch_size   : int (default 50)
    max_batch_size   : int (default 500)
    use_pulp_scheduler : bool or null (default null)
    time_limit       : int (default 60)
    async_mode       : bool (default true) ⭐ NEW
        - true: Return immediately with task_id (recommended for production)
        - false: Wait for completion (useful for testing/small datasets)

    Response (async_mode=true)
    --------------------------
    {
        "task_id": "abc-123-def",
        "status": "processing",
        "check_status_url": "/api/task-status/abc-123-def/"
    }

    Response (async_mode=false)
    ---------------------------
    Same as before (full results)
    """
    try:
        max_num_batches, min_batch_size, max_batch_size = get_batch_params(request)
        
        use_pulp_scheduler = request.data.get('use_pulp_scheduler')
        time_limit = int(request.data.get('time_limit', 60))
        async_mode = request.data.get('async_mode', True)  # Default to async
        
        params = {
            'max_num_batches': max_num_batches,
            'min_batch_size': min_batch_size,
            'max_batch_size': max_batch_size,
            'use_pulp_scheduler': use_pulp_scheduler,
            'time_limit': time_limit
        }
        
        # ── ASYNC MODE: Return task ID immediately ──────────────────
        if async_mode:
            task = generate_schedule_task.delay(params)
            return Response({
                'task_id': task.id,
                'status': 'processing',
                'message': 'Schedule generation started in background',
                'check_status_url': f'/api/task-status/{task.id}/'
            }, status=status.HTTP_202_ACCEPTED)
        
        # ── SYNC MODE: Wait for completion ──────────────────────────
        ProductionSchedule.objects.all().delete()

        products = Product.objects.filter(demand_2024__gt=0).order_by('item')
        if not products.exists():
            return Response(
                {'error': 'No products with demand found'},
                status=status.HTTP_400_BAD_REQUEST
            )

        batch_log = optimize_product_batches_jointly(
            products, max_num_batches, min_batch_size, max_batch_size
        )

        schedules, machine_availability = generate_production_schedule(
            products, 
            use_pulp=use_pulp_scheduler,
            time_limit_seconds=time_limit
        )
        kpis = calculate_kpis(schedules, machine_availability)

        num_operations = len(schedules)
        if use_pulp_scheduler is True:
            method_used = "PuLP job-shop (forced)"
        elif use_pulp_scheduler is False:
            method_used = "Improved greedy (forced)"
        else:
            method_used = "PuLP job-shop (auto)" if num_operations <= 100 else "Improved greedy (auto)"

        return Response({
            'message': f'Schedule generated successfully with {len(schedules)} operations',
            'scheduling_method': method_used,
            'kpis': kpis,
            'schedule_count': len(schedules),
            'batch_optimization': {
                'parameters': {
                    'max_num_batches': max_num_batches,
                    'min_batch_size':  min_batch_size,
                    'max_batch_size':  max_batch_size,
                },
                'products_optimized':   len(batch_log),
                'sample_optimizations': batch_log[:5]
            },
            'performance': {
                'total_operations': num_operations,
                'method_used': method_used,
                'time_limit_seconds': time_limit if use_pulp_scheduler else None
            }
        }, status=status.HTTP_201_CREATED)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_schedule(request):
    """Get production schedule with optional filters."""
    try:
        schedules = ProductionSchedule.objects.all()

        machine    = request.GET.get('machine')
        product    = request.GET.get('product')
        start_date = request.GET.get('start_date')
        end_date   = request.GET.get('end_date')

        if machine:
            schedules = schedules.filter(machine__name=machine)
        if product:
            schedules = schedules.filter(product__item=product)
        if start_date:
            schedules = schedules.filter(start_time__gte=start_date)
        if end_date:
            schedules = schedules.filter(end_time__lte=end_date)

        serializer = ProductionScheduleSerializer(schedules, many=True)
        return Response({'schedules': serializer.data, 'count': schedules.count()},
                        status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_filter_options(request):
    """Get available filter options (machines, products, date range) from DB."""
    try:
        machines = Machine.objects.all().values_list('name', flat=True)
        products = Product.objects.filter(demand_2024__gt=0).values('item', 'description')

        schedules = ProductionSchedule.objects.all()
        if schedules.exists():
            min_date = schedules.aggregate(Min('start_time'))['start_time__min']
            max_date = schedules.aggregate(Max('end_time'))['end_time__max']
        else:
            min_date = max_date = None

        return Response({
            'machines': list(machines),
            'products': list(products),
            'date_range': {'min': min_date, 'max': max_date}
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# KPIs  (unchanged logic)
# ===========================================================================

@api_view(['GET'])
def get_kpis(request):
    """Get KPI metrics with bottleneck detection."""
    try:
        schedules = ProductionSchedule.objects.all()
        if not schedules.exists():
            return Response({'message': 'No schedules available'}, status=status.HTTP_200_OK)

        min_start      = schedules.aggregate(Min('start_time'))['start_time__min']
        max_end        = schedules.aggregate(Max('end_time'))['end_time__max']
        makespan_hours = (max_end - min_start).total_seconds() / 3600
        makespan_days  = makespan_hours / 24

        machines       = Machine.objects.all()
        machine_stats  = []

        for machine in machines:
            used_hours = schedules.filter(machine=machine).aggregate(
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

        bottleneck = None
        if machine_stats_sorted:
            bottleneck = {
                'machine':     machine_stats_sorted[0]['machine'],
                'utilization': machine_stats_sorted[0]['utilization'],
                'used_hours':  machine_stats_sorted[0]['used_hours']
            }

        total_units          = Product.objects.filter(demand_2024__gt=0).aggregate(
            total=Sum('demand_2024')
        )['total'] or 0
        throughput_per_day   = (total_units / makespan_days)  if makespan_days  > 0 else 0
        throughput_per_hour  = (total_units / makespan_hours) if makespan_hours > 0 else 0

        return Response({
            'total_makespan_hours':      round(makespan_hours,      2),
            'total_makespan_days':       round(makespan_days,       2),
            'machine_utilization':       machine_stats_sorted,
            'bottleneck':                bottleneck,
            'total_operations':          schedules.count(),
            'throughput_units_per_day':   round(throughput_per_day,  2),
            'throughput_units_per_hour':  round(throughput_per_hour, 2),
            'total_units_scheduled':     total_units
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# DATA INITIALISATION  (unchanged)
# ===========================================================================

@api_view(['GET'])
def get_task_status(request, task_id):
    """
    Check the status of an async task.
    
    URL: /api/task-status/<task_id>/
    
    Response States
    ---------------
    PENDING   : Task is waiting to start
    PROGRESS  : Task is running (includes progress %)
    SUCCESS   : Task completed successfully
    FAILURE   : Task failed with error
    
    Response Format
    ---------------
    {
        "task_id": "abc-123",
        "state": "PROGRESS",
        "progress": 60,
        "status": "Generating production schedule",
        "result": null  // Available when state=SUCCESS
    }
    """
    try:
        task_result = AsyncResult(task_id)
        
        response = {
            'task_id': task_id,
            'state': task_result.state,
        }
        
        if task_result.state == 'PENDING':
            response.update({
                'progress': 0,
                'status': 'Task is waiting to start...'
            })
        elif task_result.state == 'PROGRESS':
            response.update({
                'progress': task_result.info.get('progress', 0),
                'status': task_result.info.get('status', 'Processing...')
            })
        elif task_result.state == 'SUCCESS':
            response.update({
                'progress': 100,
                'status': 'Complete',
                'result': task_result.result
            })
        elif task_result.state == 'FAILURE':
            response.update({
                'progress': 0,
                'status': 'Task failed',
                'error': str(task_result.info)
            })
        else:
            response.update({
                'progress': 0,
                'status': task_result.state
            })
        
        return Response(response, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def initialize_data(request):
    """
    Initialize database with uploaded CSV files (supports async mode).
    
    Request Parameters
    ------------------
    frontpage : CSV file (required)
    process   : CSV file (required)
    async_mode : bool (default true)
    
    Response (async_mode=true)
    --------------------------
    {
        "task_id": "xyz-789",
        "status": "processing",
        "check_status_url": "/api/task-status/xyz-789/"
    }
    """
    try:
        if 'frontpage' not in request.FILES or 'process' not in request.FILES:
            return Response(
                {'error': 'Both Frontpage.csv and Process.csv files are required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        frontpage_file = request.FILES['frontpage']
        process_file = request.FILES['process']
        async_mode = request.POST.get('async_mode', 'true').lower() == 'true'
        
        # ── ASYNC MODE ───────────────────────────────────────────────
        if async_mode:
            # Save files temporarily
            temp_dir = tempfile.mkdtemp()
            frontpage_path = os.path.join(temp_dir, 'frontpage.csv')
            process_path = os.path.join(temp_dir, 'process.csv')
            
            with open(frontpage_path, 'wb+') as f:
                for chunk in frontpage_file.chunks():
                    f.write(chunk)
            
            with open(process_path, 'wb+') as f:
                for chunk in process_file.chunks():
                    f.write(chunk)
            
            task = initialize_data_task.delay(frontpage_path, process_path)
            
            return Response({
                'task_id': task.id,
                'status': 'processing',
                'message': 'Data initialization started in background',
                'check_status_url': f'/api/task-status/{task.id}/'
            }, status=status.HTTP_202_ACCEPTED)
        
        # ── SYNC MODE ────────────────────────────────────────────────
        try:
            frontpage_df = pd.read_csv(frontpage_file)
            process_df = pd.read_csv(process_file)
            process_df = process_df.iloc[:, :-2]
        except Exception as e:
            return Response(
                {'error': f'Error reading CSV files: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )

        demand_data = process_frontpage_data(frontpage_df)
        process_routing, machines_list = process_routing_data(process_df)

        Product.objects.all().delete()
        Machine.objects.all().delete()
        ProcessStep.objects.all().delete()
        ProductionSchedule.objects.all().delete()

        machines = {}
        for machine_name in machines_list:
            if machine_name:
                machines[machine_name] = Machine.objects.create(
                    name=machine_name, available_hours_per_day=24
                )

        for i in range(len(demand_data['Item'])):
            item = demand_data['Item'][i]
            demand = demand_data['Demand_2024'][i]

            if pd.isna(demand) or demand is None:
                demand = 0

            batch_size = int(np.ceil(demand / 12)) if demand > 0 else 1

            product = Product.objects.create(
                item=item,
                sap_tn=str(demand_data['SAP_TN'][i]) if demand_data['SAP_TN'][i] is not None else '',
                sap_pl=str(demand_data['SAP_PL'][i]) if demand_data['SAP_PL'][i] is not None else None,
                dcc_type=demand_data['DCC_Type'][i] if demand_data['DCC_Type'][i] is not None else '',
                description=demand_data['Description'][i] if demand_data['Description'][i] is not None else '',
                demand_2024=int(demand),
                batch_size=batch_size,
                num_batches=12
            )

            for step_data in process_routing:
                if step_data['item'] == item:
                    machine_name = step_data['machine']
                    if machine_name in machines:
                        ProcessStep.objects.create(
                            product=product,
                            step_number=step_data['step'],
                            machine=machines[machine_name],
                            step_name=step_data['name'],
                            cycle_time_seconds=step_data['time'],
                            workers_required=step_data['workers']
                        )

        return Response({
            'message': 'Database initialized successfully',
            'products_created': Product.objects.count(),
            'machines_created': Machine.objects.count(),
            'process_steps_created': ProcessStep.objects.count()
        }, status=status.HTTP_201_CREATED)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# BUFFER OPTIMISATION  –  now uses PuLP allocation model
# ===========================================================================

@api_view(['GET'])
def get_buffer_optimization(request):
    """
    Allocate buffers across machines using a PuLP LP.

    Query Parameters
    ----------------
    safety_factor  : float (default 1.5)  – multiplier on the raw buffer estimate
    total_budget   : float (optional)     – if provided, the PuLP model distributes
                                            this budget optimally.  When omitted the
                                            raw (unconstrained) recommended buffer is
                                            returned for every machine.

    How the PuLP model works
    ------------------------
    * Each machine has a *required* buffer  =  throughput × avg_op_time × safety_factor.
    * If total_budget < Σ required, we cannot fill every machine fully.
    * The LP minimises utilisation-weighted shortfall, so high-utilisation
      machines get buffer first.
    """
    try:
        schedules = ProductionSchedule.objects.all()
        if not schedules.exists():
            return Response({'message': 'No schedules available'}, status=status.HTTP_200_OK)

        safety_factor = float(request.GET.get('safety_factor', 1.5))
        total_budget  = request.GET.get('total_budget')          # optional

        # ── Compute raw required buffers (same formula as before) ──
        min_start      = schedules.aggregate(Min('start_time'))['start_time__min']
        max_end        = schedules.aggregate(Max('end_time'))['end_time__max']
        makespan_hours = (max_end - min_start).total_seconds() / 3600

        total_units        = Product.objects.filter(demand_2024__gt=0).aggregate(
            total=Sum('demand_2024')
        )['total'] or 0
        throughput_per_hour = (total_units / makespan_hours) if makespan_hours > 0 else 0

        machines = Machine.objects.all()
        buffer_data = []                                  # input to PuLP helper

        for machine in machines:
            machine_schedules = schedules.filter(machine=machine)
            if not machine_schedules.exists():
                continue

            avg_duration_hours = machine_schedules.aggregate(
                avg=Avg('duration_hours')
            )['avg'] or 0

            used_hours  = machine_schedules.aggregate(
                total=Sum('duration_hours')
            )['total'] or 0
            utilization = (used_hours / makespan_hours * 100) if makespan_hours > 0 else 0

            required_buffer = throughput_per_hour * avg_duration_hours * safety_factor

            buffer_data.append({
                'machine':                machine.name,
                'required_buffer':        round(required_buffer, 2),
                'buffer_size_units':      round(required_buffer, 2),   # kept for response compat
                'avg_operation_time_hours': round(avg_duration_hours, 4),
                'throughput_per_hour':    round(throughput_per_hour, 2),
                'safety_factor':         safety_factor,
                'utilization':           round(utilization, 2),
                'total_operations':      machine_schedules.count(),
                'recommendation':        (
                    'HIGH PRIORITY'   if utilization > 80 else
                    'MEDIUM PRIORITY' if utilization > 60 else
                    'LOW PRIORITY'
                ),
            })

        # ── Run PuLP allocation if a budget is supplied ───────────
        if total_budget is not None:
            total_budget = float(total_budget)
            buffer_data  = pulp_optimize_buffers(buffer_data, total_budget)

        # Sort by required buffer descending
        buffer_data.sort(key=lambda x: x['required_buffer'], reverse=True)

        response_payload = {
            'buffer_recommendations': buffer_data,
            'parameters': {
                'throughput_per_hour': round(throughput_per_hour, 2),
                'makespan_hours':     round(makespan_hours, 2),
                'safety_factor':      safety_factor,
                'total_units':        total_units,
            },
            'formula': 'required_buffer = throughput_per_hour × avg_operation_time_hours × safety_factor',
        }

        if total_budget is not None:
            response_payload['pulp_allocation'] = {
                'total_budget':    total_budget,
                'total_required':  round(sum(d['required_buffer'] for d in buffer_data), 2),
                'total_allocated': round(sum(d.get('allocated_buffer', 0) for d in buffer_data), 2),
            }

        return Response(response_payload, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# BOTTLENECK ANALYSIS  (unchanged – purely analytical, no optimisation)
# ===========================================================================

@api_view(['GET'])
def get_bottleneck_analysis(request):
    """Detailed bottleneck analysis with recommendations."""
    try:
        schedules = ProductionSchedule.objects.all()
        if not schedules.exists():
            return Response({'message': 'No schedules available'}, status=status.HTTP_200_OK)

        min_start      = schedules.aggregate(Min('start_time'))['start_time__min']
        max_end        = schedules.aggregate(Max('end_time'))['end_time__max']
        makespan_hours = (max_end - min_start).total_seconds() / 3600

        machines           = Machine.objects.all()
        bottleneck_analysis = []

        for machine in machines:
            machine_schedules = schedules.filter(machine=machine)
            if not machine_schedules.exists():
                continue

            used_hours = machine_schedules.aggregate(
                total=Sum('duration_hours')
            )['total'] or 0

            utilization       = (used_hours / makespan_hours * 100) if makespan_hours > 0 else 0
            num_operations    = machine_schedules.count()
            avg_operation_time = machine_schedules.aggregate(
                avg=Avg('duration_hours')
            )['avg'] or 0

            idle_hours      = makespan_hours - used_hours
            idle_percentage = (idle_hours / makespan_hours * 100) if makespan_hours > 0 else 0

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

        bottleneck_analysis.sort(key=lambda x: x['utilization'], reverse=True)

        summary = {
            'total_makespan_hours':  round(makespan_hours, 2),
            'bottleneck_machine':    bottleneck_analysis[0]['machine']      if bottleneck_analysis else None,
            'bottleneck_utilization': bottleneck_analysis[0]['utilization'] if bottleneck_analysis else 0,
            'avg_utilization':       round(np.mean([m['utilization'] for m in bottleneck_analysis]), 2) if bottleneck_analysis else 0,
            'total_machines':        len(bottleneck_analysis),
        }

        return Response({'summary': summary, 'machine_analysis': bottleneck_analysis},
                        status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# BATCH OPTIMISATION PREVIEW  –  inherits PuLP via calculate_optimal_batch_size
# ===========================================================================

@api_view(['GET'])
def get_batch_optimization_preview(request):
    """
    Preview batch-size optimization for every product (supports async).
    
    Query Parameters
    ----------------
    max_num_batches : int (default 25)
    min_batch_size  : int (default 50)
    max_batch_size  : int (default 500)
    async_mode      : bool (default false) - set to true for large datasets
    
    For large datasets (100+ products), consider using async_mode=true
    """
    try:
        max_num_batches = int(request.GET.get('max_num_batches', 25))
        min_batch_size  = int(request.GET.get('min_batch_size',  50))
        max_batch_size  = int(request.GET.get('max_batch_size', 500))
        async_mode      = request.GET.get('async_mode', 'false').lower() == 'true'
        
        params = {
            'max_num_batches': max_num_batches,
            'min_batch_size': min_batch_size,
            'max_batch_size': max_batch_size
        }
        
        # ── ASYNC MODE ───────────────────────────────────────────────
        if async_mode:
            task = batch_optimize_preview_task.delay(params)
            return Response({
                'task_id': task.id,
                'status': 'processing',
                'message': 'Batch optimization preview started in background',
                'check_status_url': f'/api/task-status/{task.id}/'
            }, status=status.HTTP_202_ACCEPTED)
        
        # ── SYNC MODE ────────────────────────────────────────────────
        products = Product.objects.filter(demand_2024__gt=0)

        batch_analysis = []
        total_demand   = 0
        total_batches  = 0
        batch_sizes    = []

        for product in products:
            demand = product.demand_2024

            batch_size, num_batches, ideal_batch = calculate_optimal_batch_size(
                demand, max_num_batches, min_batch_size, max_batch_size
            )

            old_batch_size  = int(np.ceil(demand / 12))
            old_num_batches = 12

            batch_analysis.append({
                'item':             product.item,
                'description':      product.description[:50],
                'demand':           demand,
                'old_batch_size':   old_batch_size,
                'old_num_batches':  old_num_batches,
                'new_batch_size':   batch_size,
                'new_num_batches':  num_batches,
                'ideal_batch_size': round(ideal_batch, 2),
                'improvement':      (
                    f"{((old_num_batches - num_batches) / old_num_batches * 100):.1f}%"
                    if old_num_batches != num_batches else "0%"
                ),
            })

            total_demand  += demand
            total_batches += num_batches
            batch_sizes.append(batch_size)

        return Response({
            'batch_analysis': batch_analysis,
            'summary': {
                'total_products':  len(batch_analysis),
                'total_demand':    total_demand,
                'total_batches':   total_batches,
                'avg_batch_size':  round(np.mean(batch_sizes), 2)  if batch_sizes else 0,
                'min_batch_size':  int(np.min(batch_sizes))        if batch_sizes else 0,
                'max_batch_size':  int(np.max(batch_sizes))        if batch_sizes else 0,
                'std_batch_size':  round(np.std(batch_sizes), 2)   if batch_sizes else 0,
            },
            'parameters': {
                'max_num_batches': max_num_batches,
                'min_batch_size':  min_batch_size,
                'max_batch_size':  max_batch_size,
            }
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)