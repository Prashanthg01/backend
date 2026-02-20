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

from .models import Product, Machine, ProcessStep, ProductionSchedule
from .utils import (
    clean_numeric_columns, clean_text_columns, apply_filters,
    clean_shift_columns, calculate_efficiency, calculate_backlog,
    calculate_production_outputs, SHIFT_LABELS, build_summary,
    process_frontpage_data, process_routing_data,
    calculate_optimal_batch_size, pulp_optimize_buffers,
)
from .tasks import (
    initialize_data_task,
    generate_schedule_task,
    batch_optimize_preview_task,
)

_preview_cache = {}


# ===========================================================================
# CSV PROCESSING
# ===========================================================================

@api_view(['POST'])
def process_csv(request):
    csv_file = request.FILES.get('file')
    if not csv_file:
        return Response({'error': 'No file uploaded'}, status=400)

    num_shifts = int(request.POST.get("num_shifts", 28))
    df = pd.read_csv(csv_file)

    clean_numeric_columns(df, ['Planned', 'Realized', 'Backlog', 'Open'])
    clean_text_columns(df, ['Step', 'Area', 'Sub-Project'])
    clean_text_columns(df, ['PPS TN', 'Project', 'Sub-Project', 'Machine', 'Tool No.', 'Area'])

    df = apply_filters(df, {
        'PPS TN':      request.POST.get("pps_tn",       "All"),
        'Project':     request.POST.get("project",      "All"),
        'Sub-Project': request.POST.get("sub_project",  "All"),
        'Machine':     request.POST.get("machine",      "All"),
        'Tool No.':    request.POST.get("tool_no",      "All"),
        'Area':        request.POST.get("area",         "All"),
    })

    clean_shift_columns(df, [(14, 50), (95, 113)])

    efficiency = calculate_efficiency(df, num_shifts)
    backlog    = calculate_backlog(df)

    finished_filter  = (df['Step'] == 'F') & df['Sub-Project'].notna()
    connector_filter = (df['Area'] == 'Assembly') & df['Sub-Project'].notna()

    fg_output, conn_output = calculate_production_outputs(df, finished_filter, connector_filter)

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
            'machines':   list(machines),
            'products':   list(products),
            'date_range': {'min': min_date, 'max': max_date}
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# KPIs
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

        machines      = Machine.objects.all()
        machine_stats = []

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

        bottleneck = None
        if machine_stats_sorted:
            bottleneck = {
                'machine':     machine_stats_sorted[0]['machine'],
                'utilization': machine_stats_sorted[0]['utilization'],
                'used_hours':  machine_stats_sorted[0]['used_hours']
            }

        total_units         = Product.objects.filter(demand_2024__gt=0).aggregate(
            total=Sum('demand_2024')
        )['total'] or 0
        throughput_per_day  = (total_units / makespan_days)  if makespan_days  > 0 else 0
        throughput_per_hour = (total_units / makespan_hours) if makespan_hours > 0 else 0

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
    """Check the status of an async Celery task."""
    try:
        task_result = AsyncResult(task_id)
        response    = {'task_id': task_id, 'state': task_result.state}

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

        return Response(response, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# DATA INITIALISATION
# ===========================================================================

@api_view(['POST'])
def initialize_data(request):
    """Initialize database with uploaded CSV files (supports async mode)."""
    try:
        if 'frontpage' not in request.FILES or 'process' not in request.FILES:
            return Response(
                {'error': 'Both Frontpage.csv and Process.csv files are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        frontpage_file = request.FILES['frontpage']
        process_file   = request.FILES['process']
        async_mode     = request.POST.get('async_mode', 'true').lower() == 'true'

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
        try:
            frontpage_df = pd.read_csv(frontpage_file)
            process_df   = pd.read_csv(process_file)
            process_df   = process_df.iloc[:, :-2]
        except Exception as e:
            return Response({'error': f'Error reading CSV files: {str(e)}'}, status=400)

        demand_data                    = process_frontpage_data(frontpage_df)
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
    """Allocate buffers across machines using a PuLP LP."""
    try:
        schedules = ProductionSchedule.objects.all()
        if not schedules.exists():
            return Response({'message': 'No schedules available'}, status=status.HTTP_200_OK)

        safety_factor = float(request.GET.get('safety_factor', 1.5))
        total_budget  = request.GET.get('total_budget')

        min_start      = schedules.aggregate(Min('start_time'))['start_time__min']
        max_end        = schedules.aggregate(Max('end_time'))['end_time__max']
        makespan_hours = (max_end - min_start).total_seconds() / 3600

        total_units         = Product.objects.filter(demand_2024__gt=0).aggregate(
            total=Sum('demand_2024')
        )['total'] or 0
        throughput_per_hour = (total_units / makespan_hours) if makespan_hours > 0 else 0

        machines    = Machine.objects.all()
        buffer_data = []

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

        if total_budget is not None:
            total_budget = float(total_budget)
            buffer_data  = pulp_optimize_buffers(buffer_data, total_budget)

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
    """Detailed bottleneck analysis with recommendations."""
    try:
        schedules = ProductionSchedule.objects.all()
        if not schedules.exists():
            return Response({'message': 'No schedules available'}, status=status.HTTP_200_OK)

        min_start      = schedules.aggregate(Min('start_time'))['start_time__min']
        max_end        = schedules.aggregate(Max('end_time'))['end_time__max']
        makespan_hours = (max_end - min_start).total_seconds() / 3600

        machines             = Machine.objects.all()
        bottleneck_analysis  = []

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

@api_view(['GET'])
def get_batch_optimization_preview(request):
    """Preview batch-size optimization for every product (supports async)."""
    try:
        max_num_batches = int(request.GET.get('max_num_batches', 25))
        min_batch_size  = int(request.GET.get('min_batch_size',  50))
        max_batch_size  = int(request.GET.get('max_batch_size', 500))
        async_mode      = request.GET.get('async_mode', 'false').lower() == 'true'

        params = {
            'max_num_batches': max_num_batches,
            'min_batch_size':  min_batch_size,
            'max_batch_size':  max_batch_size,
        }

        if async_mode:
            task = batch_optimize_preview_task.delay(params)
            return Response({
                'task_id':          task.id,
                'status':           'processing',
                'message':          'Batch optimization preview started in background',
                'check_status_url': f'/api/task-status/{task.id}/'
            }, status=status.HTTP_202_ACCEPTED)

        products       = Product.objects.filter(demand_2024__gt=0)
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
                'total_products': len(batch_analysis),
                'total_demand':   total_demand,
                'total_batches':  total_batches,
                'avg_batch_size': round(np.mean(batch_sizes), 2) if batch_sizes else 0,
                'min_batch_size': int(np.min(batch_sizes))       if batch_sizes else 0,
                'max_batch_size': int(np.max(batch_sizes))       if batch_sizes else 0,
                'std_batch_size': round(np.std(batch_sizes), 2)  if batch_sizes else 0,
            },
            'parameters': params
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===========================================================================
# GENERATE SCHEDULE
# ===========================================================================

@api_view(['POST'])
def generate_schedule(request):
    """
    Kick off async job-shop schedule generation.

    Request body
    ------------
    start_date         : "YYYY-MM-DD"   (default today)
    local_opt_machines : int            (default 5)
    enable_compaction  : bool           gap elimination ON/OFF (default true) ← NEW
    clear_existing     : bool           (default true)
    product_pks        : [int]          (default: all)
    batch_overrides    : [[pk,bs,nb],…] (optional)
    async_mode         : bool           (default true)
    """
    try:
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

        params = {
            'start_date':          request.data.get('start_date'),
            'local_opt_machines':  int(request.data.get('local_opt_machines', 5)),
            'enable_compaction':   bool(request.data.get('enable_compaction', True)),  # ← NEW
            'clear_existing':      request.data.get('clear_existing', True),
            'product_pks':         request.data.get('product_pks'),
            'batch_overrides':     request.data.get('batch_overrides', []),
        }

        async_mode = str(request.data.get('async_mode', 'true')).lower() == 'true'

        if async_mode:
            task = generate_schedule_task.delay(params)
            return Response({
                'task_id':          task.id,
                'status':           'processing',
                'message':          'Schedule generation started in background.',
                'check_status_url': f'/api/task-status/{task.id}/',
            }, status=status.HTTP_202_ACCEPTED)

        # Sync fallback
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
    """Retrieve persisted schedule records with optional filters."""
    try:
        schedules = ProductionSchedule.objects.select_related(
            'machine', 'product', 'process_step'
        ).order_by('start_time', 'machine')

        machine_filter = request.GET.get('machine')
        product_filter = request.GET.get('product')
        if machine_filter:
            schedules = schedules.filter(machine__name__icontains=machine_filter)
        if product_filter:
            schedules = schedules.filter(product__item=product_filter)

        total_count = schedules.count()
        page_size   = min(int(request.GET.get('page_size', 200)), 1000)
        page        = max(int(request.GET.get('page', 1)), 1)
        offset      = (page - 1) * page_size
        schedules   = schedules[offset: offset + page_size]

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
    """Return Gantt-ready data aggregated by machine."""
    try:
        schedules = ProductionSchedule.objects.select_related(
            'machine', 'product', 'process_step'
        )

        machines_param = request.GET.get('machines')
        if machines_param:
            machine_names = [m.strip() for m in machines_param.split(',')]
            schedules = schedules.filter(machine__name__in=machine_names)

        start_param = request.GET.get('start')
        end_param   = request.GET.get('end')
        if start_param:
            schedules = schedules.filter(start_time__gte=start_param)
        if end_param:
            schedules = schedules.filter(end_time__lte=end_param)

        if not schedules.exists():
            return Response({'gantt_bars': [], 'machines': [], 'date_range': {}, 'total_bars': 0})

        total_bars = schedules.count()
        max_bars   = int(request.GET.get('max_bars', 500))
        schedules  = schedules.order_by('machine__name', 'start_time')[:max_bars]

        product_items = list(
            ProductionSchedule.objects.values_list('product__item', flat=True).distinct()
        )
        color_map = {item: idx % 20 for idx, item in enumerate(sorted(product_items))}

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

    Analyse the current saved schedule for idle gaps between consecutive
    operations on each machine.

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
    """
    try:
        schedules = (
            ProductionSchedule.objects
            .select_related('machine', 'product', 'process_step')
            .order_by('machine__name', 'start_time')
        )

        if not schedules.exists():
            return Response(
                {'detail': 'No schedule found. Generate a schedule first.'},
                status=status.HTTP_404_NOT_FOUND
            )

        # Group by machine
        machine_ops: dict = defaultdict(list)
        for s in schedules:
            machine_ops[s.machine.name].append(s)

        gap_list      = []
        machine_stats = []
        causes        = {'precedence': 0, 'interleave': 0, 'spt': 0}

        for machine_name, mops in machine_ops.items():
            # Already ordered by start_time from the queryset
            idle_hours = 0.0
            gap_count  = 0

            for i in range(1, len(mops)):
                prev     = mops[i - 1]
                curr     = mops[i]
                gap_secs = (curr.start_time - prev.end_time).total_seconds()

                if gap_secs > 60:       # > 1 minute = genuine idle gap
                    idle_h     = gap_secs / 3600
                    idle_hours += idle_h
                    gap_count  += 1

                    # Classify cause
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
        gap_list.sort(key=lambda g: g['idle_hours'], reverse=True)
        machine_stats.sort(key=lambda m: m['idle_hours'], reverse=True)

        worst = machine_stats[0]['machine'] if machine_stats else '—'
        clean = sum(1 for m in machine_stats if m['gap_count'] == 0)

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
    """Compare KPIs of the current saved schedule."""
    try:
        schedules = ProductionSchedule.objects.select_related('machine', 'product')
        if not schedules.exists():
            return Response({'message': 'No schedule available.'}, status=200)

        agg = schedules.aggregate(
            start=Min('start_time'), end=Max('end_time'),
            total_ops=Count('id'), total_units=Sum('batch_size')
        )
        makespan = (agg['end'] - agg['start']).total_seconds() / 3600 if agg['end'] else 0

        machine_stats = (
            schedules.values('machine__name')
            .annotate(load=Sum('duration_hours'), ops=Count('id'))
            .order_by('-load')
        )

        stats = [
            {
                'machine':     m['machine__name'],
                'used_hours':  round(m['load'], 2),
                'utilization': round(m['load'] / makespan * 100, 2) if makespan else 0,
                'operations':  m['ops'],
            }
            for m in machine_stats
        ]

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
    """Preview re-running the scheduler without saving to DB."""
    try:
        local_opt = int(request.GET.get('local_opt_machines', 5))
        products  = list(Product.objects.filter(demand_2024__gt=0))

        if not products:
            return Response({'error': 'No products in DB.'}, status=400)

        from .utils import run_job_shop_scheduler, compute_schedule_kpis
        from datetime import datetime

        rows     = run_job_shop_scheduler(products, datetime.now(), local_opt_machines=local_opt)
        makespan = max(r['end_hrs'] for r in rows) if rows else 0
        kpis     = compute_schedule_kpis(rows, makespan)

        return Response({'preview_kpis': kpis, 'total_operations': len(rows)})

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def optimize_schedule_save(request):
    """Same as generate_schedule but always re-optimises and saves."""
    request.data['clear_existing']     = True
    request.data['local_opt_machines'] = int(request.data.get('local_opt_machines', 5))
    return generate_schedule(request)