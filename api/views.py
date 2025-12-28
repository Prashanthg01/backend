from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd
import numpy as np
from django.db.models import Sum, Count, Q, Max, Min, Avg
from .utils import clean_numeric_columns, clean_text_columns, apply_filters, clean_shift_columns, calculate_efficiency, calculate_backlog, calculate_production_outputs, SHIFT_LABELS, build_summary

@api_view(['POST'])
def process_csv(request):
    """
    Process a production planning CSV file and return shift-wise metrics.

    This API accepts a CSV upload along with optional filter parameters
    (PPS TN, Project, Sub-Project, Machine, Tool No., Area). It performs
    data cleaning, filtering, and computes:

    - Shift-wise production output (Finished Goods & Connectors)
    - Shift-wise backlog (Finished Goods)
    - Shift-wise overall efficiency (%)
    - A summary table of planned vs realized output and backlog status

    Request Parameters:
    -------------------
    file : CSV file (required)
    num_shifts : int (optional, default=28)
    pps_tn, project, sub_project, machine, tool_no, area : str (optional)

    Response:
    --------
    JSON object with:
      - ShiftWise metrics
      - Summary table
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
        'PPS TN': request.POST.get("pps_tn", "All"),
        'Project': request.POST.get("project", "All"),
        'Sub-Project': request.POST.get("sub_project", "All"),
        'Machine': request.POST.get("machine", "All"),
        'Tool No.': request.POST.get("tool_no", "All"),
        'Area': request.POST.get("area", "All"),
    })

    clean_shift_columns(df, [(14, 50), (95, 113)])

    efficiency = calculate_efficiency(df, num_shifts)
    backlog = calculate_backlog(df)

    finished_filter = (df['Step'] == 'F') & df['Sub-Project'].notna()
    connector_filter = (df['Area'] == 'Assembly') & df['Sub-Project'].notna()

    fg_output, conn_output = calculate_production_outputs(
        df, finished_filter, connector_filter
    )

    result = {
        "Total Backlog Finished Goods": dict(zip(SHIFT_LABELS, map(str, backlog))),
        "Production Output Finished Goods": fg_output,
        "Production Output Connectors": conn_output,
        "Overall Efficiency": {
            shift: f"{eff:.2f}%" if eff > 0 else "-"
            for shift, eff in zip(SHIFT_LABELS, efficiency)
        },
    }

    return Response({
        "ShiftWise": result,
        "Summary": build_summary(df)
    })



@api_view(['POST'])
def get_filter_options(request):
    """Endpoint to get unique values for each filter column"""
    csv_file = request.FILES.get('file')
    if not csv_file:
        return Response({'error': 'No file uploaded'}, status=400)
    
    df = pd.read_csv(csv_file)
    
    filter_options = {}
    filter_columns = ['PPS TN', 'Project', 'Sub-Project', 'Machine', 'Tool No.', 'Area']
    
    for col in filter_columns:
        if col in df.columns:
            # Clean and get unique values
            unique_values = df[col].astype(str).str.strip().unique()
            # Remove 'nan' and empty strings
            unique_values = [val for val in unique_values if val and val != 'nan']
            unique_values = sorted(unique_values)
            filter_options[col] = unique_values
        else:
            filter_options[col] = []
    
    return Response(filter_options)

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.db.models import Sum, Count, Q, Max, Min
from datetime import datetime, timedelta
import numpy as np
from .models import Product, Machine, ProcessStep, ProductionSchedule
from .serializers import (
    ProductSerializer, MachineSerializer, 
    ProcessStepSerializer, ProductionScheduleSerializer
)


@api_view(['POST'])
def generate_schedule(request):
    """
    Generate production schedule with optimized batch sizes
    """
    try:
        # Get batch optimization parameters
        max_num_batches = int(request.data.get('max_num_batches', 25))
        min_batch_size = int(request.data.get('min_batch_size', 50))
        max_batch_size = int(request.data.get('max_batch_size', 500))
        
        # Clear existing schedules
        ProductionSchedule.objects.all().delete()
        
        # Get all products with demand > 0
        products = Product.objects.filter(demand_2024__gt=0).order_by('item')
        
        if not products.exists():
            return Response({
                'error': 'No products with demand found'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Update products with optimized batch sizes
        batch_optimization_log = []
        
        for product in products:
            batch_size, num_batches, ideal_batch = calculate_optimal_batch_size(
                product.demand_2024,
                max_num_batches,
                min_batch_size,
                max_batch_size
            )
            
            # Update product
            product.batch_size = batch_size
            product.num_batches = num_batches
            product.save()
            
            batch_optimization_log.append({
                'item': product.item,
                'demand': product.demand_2024,
                'batch_size': batch_size,
                'num_batches': num_batches,
                'ideal_batch_size': round(ideal_batch, 2)
            })
        
        # Initialize tracking variables
        machine_availability = {}
        batch_completion = {}
        schedule_records = []
        
        # Start time
        start_date = datetime.now()
        
        # Generate schedule for each product
        for product in products:
            # Get process steps for this product
            process_steps = ProcessStep.objects.filter(
                product=product,
                cycle_time_seconds__gt=0
            ).order_by('step_number')
            
            if not process_steps.exists():
                continue
            
            # Process each batch
            for batch_num in range(1, product.num_batches + 1):
                batch_id = f"Item{product.item}_B{batch_num}"
                
                # Process each step
                for step in process_steps:
                    machine = step.machine
                    
                    # Calculate total processing time
                    total_time_sec = step.cycle_time_seconds * product.batch_size
                    total_time_hours = total_time_sec / 3600
                    
                    # Determine start time
                    machine_key = machine.name
                    machine_available_time = machine_availability.get(machine_key, start_date)
                    
                    prev_step_key = f"{batch_id}_Step{step.step_number - 1}"
                    prev_step_completion = batch_completion.get(prev_step_key, start_date)
                    
                    operation_start = max(machine_available_time, prev_step_completion)
                    operation_end = operation_start + timedelta(hours=total_time_hours)
                    
                    # Update trackers
                    machine_availability[machine_key] = operation_end
                    batch_completion[f"{batch_id}_Step{step.step_number}"] = operation_end
                    
                    # Create schedule record
                    schedule = ProductionSchedule.objects.create(
                        machine=machine,
                        product=product,
                        process_step=step,
                        batch_id=batch_id,
                        batch_num=batch_num,
                        batch_size=product.batch_size,
                        start_time=operation_start,
                        end_time=operation_end,
                        duration_hours=round(total_time_hours, 4)
                    )
                    schedule_records.append(schedule)
        
        # Calculate KPIs
        total_schedules = len(schedule_records)
        
        if total_schedules > 0:
            max_end_time = max(s.end_time for s in schedule_records)
            min_start_time = min(s.start_time for s in schedule_records)
            makespan_hours = (max_end_time - min_start_time).total_seconds() / 3600
            makespan_days = makespan_hours / 24
            
            # Machine utilization
            machine_stats = {}
            for machine_name, end_time in machine_availability.items():
                used_hours = ProductionSchedule.objects.filter(
                    machine__name=machine_name
                ).aggregate(total=Sum('duration_hours'))['total'] or 0
                
                utilization = (used_hours / makespan_hours * 100) if makespan_hours > 0 else 0
                machine_stats[machine_name] = {
                    'used_hours': round(used_hours, 2),
                    'utilization': round(utilization, 2)
                }
            
            # Calculate throughput
            total_units = Product.objects.filter(demand_2024__gt=0).aggregate(
                total=Sum('demand_2024')
            )['total'] or 0
            throughput_per_day = (total_units / makespan_days) if makespan_days > 0 else 0
            
            kpis = {
                'total_makespan_hours': round(makespan_hours, 2),
                'total_makespan_days': round(makespan_days, 2),
                'machine_utilization': machine_stats,
                'total_operations': total_schedules,
                'throughput_units_per_day': round(throughput_per_day, 2),
                'total_units_scheduled': total_units
            }
        else:
            kpis = {}
        
        return Response({
            'message': f'Schedule generated successfully with {total_schedules} operations',
            'kpis': kpis,
            'schedule_count': total_schedules,
            'batch_optimization': {
                'parameters': {
                    'max_num_batches': max_num_batches,
                    'min_batch_size': min_batch_size,
                    'max_batch_size': max_batch_size
                },
                'products_optimized': len(batch_optimization_log),
                'sample_optimizations': batch_optimization_log[:5]  # First 5 as sample
            }
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    

@api_view(['GET'])
def get_schedule(request):
    """
    Get production schedule with optional filters
    """
    try:
        schedules = ProductionSchedule.objects.all()
        
        # Apply filters
        machine = request.GET.get('machine')
        product = request.GET.get('product')
        start_date = request.GET.get('start_date')
        end_date = request.GET.get('end_date')
        
        if machine:
            schedules = schedules.filter(machine__name=machine)
        
        if product:
            schedules = schedules.filter(product__item=product)
        
        if start_date:
            schedules = schedules.filter(start_time__gte=start_date)
        
        if end_date:
            schedules = schedules.filter(end_time__lte=end_date)
        
        serializer = ProductionScheduleSerializer(schedules, many=True)
        
        return Response({
            'schedules': serializer.data,
            'count': schedules.count()
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_filter_options(request):
    """
    Get available filter options for frontend
    """
    try:
        machines = Machine.objects.all().values_list('name', flat=True)
        products = Product.objects.filter(demand_2024__gt=0).values('item', 'description')
        
        # Get date range from schedules
        schedules = ProductionSchedule.objects.all()
        if schedules.exists():
            min_date = schedules.aggregate(Min('start_time'))['start_time__min']
            max_date = schedules.aggregate(Max('end_time'))['end_time__max']
        else:
            min_date = None
            max_date = None
        
        return Response({
            'machines': list(machines),
            'products': list(products),
            'date_range': {
                'min': min_date,
                'max': max_date
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_kpis(request):
    """
    Get KPI metrics with bottleneck detection
    """
    try:
        schedules = ProductionSchedule.objects.all()
        
        if not schedules.exists():
            return Response({
                'message': 'No schedules available'
            }, status=status.HTTP_200_OK)
        
        # Calculate makespan
        min_start = schedules.aggregate(Min('start_time'))['start_time__min']
        max_end = schedules.aggregate(Max('end_time'))['end_time__max']
        makespan_hours = (max_end - min_start).total_seconds() / 3600
        makespan_days = makespan_hours / 24
        
        # Machine utilization and bottleneck detection
        machines = Machine.objects.all()
        machine_stats = []
        
        for machine in machines:
            used_hours = schedules.filter(machine=machine).aggregate(
                total=Sum('duration_hours')
            )['total'] or 0
            
            num_operations = schedules.filter(machine=machine).count()
            utilization = (used_hours / makespan_hours * 100) if makespan_hours > 0 else 0
            
            machine_stats.append({
                'machine': machine.name,
                'used_hours': round(used_hours, 2),
                'utilization': round(utilization, 2),
                'num_operations': num_operations
            })
        
        # Sort by utilization to identify bottleneck
        machine_stats_sorted = sorted(machine_stats, key=lambda x: x['utilization'], reverse=True)
        
        # Identify bottleneck (highest utilization)
        bottleneck = None
        if machine_stats_sorted:
            bottleneck = {
                'machine': machine_stats_sorted[0]['machine'],
                'utilization': machine_stats_sorted[0]['utilization'],
                'used_hours': machine_stats_sorted[0]['used_hours']
            }
        
        # Throughput
        total_units = Product.objects.filter(demand_2024__gt=0).aggregate(
            total=Sum('demand_2024')
        )['total'] or 0
        throughput_per_day = (total_units / makespan_days) if makespan_days > 0 else 0
        throughput_per_hour = (total_units / makespan_hours) if makespan_hours > 0 else 0
        
        # Number of setups
        num_setups = schedules.count()
        
        return Response({
            'total_makespan_hours': round(makespan_hours, 2),
            'total_makespan_days': round(makespan_days, 2),
            'machine_utilization': machine_stats_sorted,
            'bottleneck': bottleneck,
            'total_operations': num_setups,
            'throughput_units_per_day': round(throughput_per_day, 2),
            'throughput_units_per_hour': round(throughput_per_hour, 2),
            'total_units_scheduled': total_units
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def initialize_data(request):
    """
    Initialize database with sample data
    """
    try:
        # Clear existing data
        Product.objects.all().delete()
        Machine.objects.all().delete()
        ProcessStep.objects.all().delete()
        ProductionSchedule.objects.all().delete()
        
        # Create machines
        machines_data = ['SKM Seal and outer Housing Assembly DCC', '', 'ARBURG 375ST Machine 1-5', 'Kappa 350 / Kappa 330', 'TSK T1500', 'ARBURG 375ST Machine 6-10', 'Alpha 550 / Alpha 433', 'PUR-Tube Assembly Station', 'Sigma 688 / Alpha 488', 'Cutting Automation', 'Connector Assembly Station', 'Wire Rolling & Taping Station', 'SKM DCPC Crimp (Crimp and Ass)', 'ARBURG 375ST Machine 11,12,13', 'Wire Cut & Separating Station']
        
        machines = {}
        for machine_name in machines_data:
            machine = Machine.objects.create(
                name=machine_name,
                available_hours_per_day=24
            )
            machines[machine_name] = machine

        demand_data =  {'Item': [1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  12,
  13,
  14,
  15,
  16,
  17,
  18,
  19,
  20,
  21,
  22,
  23],
 'SAP_TN': [249076,
  249077,
  249313,
  249314,
  249315,
  249316,
  249317,
  249078,
  249079,
  249080,
  249081,
  249082,
  249083,
  249084,
  249085,
  249086,
  249087,
  249088,
  249089,
  249090,
  249091,
  249092,
  249093],
 'SAP_PL': [249041,
  249042,
  238895,
  238896,
  238897,
  238899,
  238900,
  249043,
  249044,
  249045,
  249046,
  249047,
  249048,
  249049,
  249050,
  249051,
  249052,
  249053,
  249054,
  249055,
  249056,
  249057,
  249058],
 'DCC_Type': ['60° & 30°',
  '60° & 30°',
  '60° & 90°B',
  '60° & 90°B',
  '60° & 2*90°B',
  '60° & 90°B',
  '60° & 2*90°B',
  '30°',
  '90°B',
  '90°',
  '60°',
  '60°',
  '60°',
  '180°',
  '180°',
  '30°',
  '90°B',
  '60°',
  '90°',
  '60°',
  '60°',
  '180°',
  '180°'],
 'Description': ['4 Wire Jacket 2xDCC Modul 9Y4252B',
  '4 Wire Jacket 2xDCC Modul 9Y4256B',
  '4 Wire Jacket 2xDCC Modul 9Y4251',
  '6 Wire Jacket 2xDCC Modul 9Y4251A',
  '6 Wire Jacket 3xDCC Modul 9Y4252',
  '6 Wire Jacket 2xDCC Modul 9Y4255',
  '6 Wire Jacket 3xDCC Modul 9Y4256',
  'Twisted Wires 1xDCC Modul 9Y4279 AA,AB,AC',
  'Twisted Wires 1xDCC Modul 9Y4279 AA,AB,AC',
  'Twisted Wires 1xDCC Modul 9Y4279 AD,AE,AF',
  'Twisted Wires 1xDCC Modul 9Y4279 AD,AE,AF',
  'Twisted Wires 1xDCC Modul 9Y4279 AA,AD',
  'Twisted Wires 1xDCC Modul 9Y4279 AB,AC,AE,AF',
  'Twisted Wires 1xDCC Modul 9Y4279 AA,AD',
  'Twisted Wires 1xDCC Modul 9Y4279 AB,AC,AE,AF',
  'Twisted Wires 1xDCC Modul 9Y4286 M,N',
  'Twisted Wires 1xDCC Modul 9Y4286 M,N',
  'Twisted Wires 1xDCC Modul 9Y4286 P,Q',
  'Twisted Wires 1xDCC Modul 9Y4286 P,Q',
  'Twisted Wires 1xDCC Modul 9Y4286 N,Q',
  'Twisted Wires 1xDCC Modul 9Y4286 M,P',
  'Twisted Wires 1xDCC Modul 9Y4286 M,P',
  'Twisted Wires 1xDCC Modul 9Y4286 N,Q'],
 'Demand_2024': [121,
  121,
  1141,
  201,
  1221,
  1342,
  1221,
  1221,
  1221,
  120,
  120,
  1127,
  214,
  1127,
  214,
  1221,
  1221,
  120,
  120,
  201,
  1140,
  1140,
  201]}

        process_routing = [{'item': 1,
  'machine': 'Kappa 350 / Kappa 330',
  'name': 'Cutting Stripping Jacket Cable 4-Wire',
  'step': 4,
  'time': 6.76,
  'workers': 0.5},
 {'item': 1,
  'machine': 'Wire Cut & Separating Station',
  'name': 'Separating & Cutting Wires to Length 1 of 2 Pairs',
  'step': 10,
  'time': 8.12,
  'workers': 0.5},
 {'item': 1,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm Jacket Cable 111mm-200mm',
  'step': 18,
  'time': 20.88,
  'workers': 0.5},
 {'item': 1,
  'machine': 'SKM DCPC Crimp (Crimp and Ass)',
  'name': 'Crimping & Assembly DCC Connector',
  'step': 30,
  'time': 20.0,
  'workers': 0.5},
 {'item': 1,
  'machine': 'ARBURG 375ST Machine 11,12,13',
  'name': 'Overmolding 60° Right 9J1 973 752 A Cod. C Blue Cod. Up With CPA',
  'step': 57,
  'time': 18.72,
  'workers': 0.5},
 {'item': 1,
  'machine': 'ARBURG 375ST Machine 6-10',
  'name': 'Overmolding 30° Left 9Y4 973 752 Cod. A Black Cod. Up With CPA',
  'step': 59,
  'time': 20.0,
  'workers': 0.5},
 {'item': 1,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Jacket Cable',
  'step': 61,
  'time': 19.58,
  'workers': 0.5},
 {'item': 1,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 114.06,
  'workers': 0.5},
 {'item': 1,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 1.9,
  'workers': 0.5},
 {'item': 2,
  'machine': 'Kappa 350 / Kappa 330',
  'name': 'Cutting Stripping Jacket Cable 4-Wire',
  'step': 4,
  'time': 6.76,
  'workers': 0.5},
 {'item': 2,
  'machine': 'Wire Cut & Separating Station',
  'name': 'Separating & Cutting Wires to Length 1 of 2 Pairs',
  'step': 10,
  'time': 8.12,
  'workers': 0.5},
 {'item': 2,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm Jacket Cable 111mm-200mm',
  'step': 18,
  'time': 20.88,
  'workers': 0.5},
 {'item': 2,
  'machine': 'SKM DCPC Crimp (Crimp and Ass)',
  'name': 'Crimping & Assembly DCC Connector',
  'step': 30,
  'time': 20.0,
  'workers': 0.5},
 {'item': 2,
  'machine': 'ARBURG 375ST Machine 11,12,13',
  'name': 'Overmolding 60° Left 9J1 973 752 Cod. C Blue Cod. Up With CPA',
  'step': 54,
  'time': 18.72,
  'workers': 0.5},
 {'item': 2,
  'machine': 'ARBURG 375ST Machine 6-10',
  'name': 'Overmolding 30° Right 9Y4 973 752 A Cod. A Black Cod. Up With CPA',
  'step': 60,
  'time': 20.0,
  'workers': 0.5},
 {'item': 2,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Jacket Cable',
  'step': 61,
  'time': 19.58,
  'workers': 0.5},
 {'item': 2,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 114.06,
  'workers': 0.5},
 {'item': 2,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 1.9,
  'workers': 0.5},
 {'item': 3,
  'machine': 'Sigma 688 / Alpha 488',
  'name': 'Cutting Stripping Crimping Twisting Single Wires',
  'step': 1,
  'time': 7.3,
  'workers': 0.5},
 {'item': 3,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 60mm-180mm',
  'step': 20,
  'time': 5.93,
  'workers': 0.5},
 {'item': 3,
  'machine': 'Connector Assembly Station',
  'name': 'Assembly DCC Connector Manually Single Wires',
  'step': 31,
  'time': 8.09,
  'workers': 0.5},
 {'item': 3,
  'machine': 'ARBURG 375ST Machine 6-10',
  'name': 'Overmolding 30° Right 9Y4 973 752 A Cod. A Black Cod. Up With CPA',
  'step': 60,
  'time': 20.0,
  'workers': 0.5},
 {'item': 3,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Single Wires',
  'step': 62,
  'time': 7.88,
  'workers': 0.5},
 {'item': 3,
  'machine': 'Cutting Automation',
  'name': 'Cutting Automation',
  'step': 67,
  'time': 8.0,
  'workers': 0.5},
 {'item': 3,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 57.2,
  'workers': 0.5},
 {'item': 3,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 0.95,
  'workers': 0.5},
 {'item': 4,
  'machine': 'Sigma 688 / Alpha 488',
  'name': 'Cutting Stripping Crimping Twisting Single Wires',
  'step': 1,
  'time': 7.29,
  'workers': 0.5},
 {'item': 4,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 60mm-180mm',
  'step': 20,
  'time': 5.93,
  'workers': 0.5},
 {'item': 4,
  'machine': 'Connector Assembly Station',
  'name': 'Assembly DCC Connector Manually Single Wires',
  'step': 31,
  'time': 8.09,
  'workers': 0.5},
 {'item': 4,
  'machine': 'ARBURG 375ST Machine 6-10',
  'name': 'Overmolding 90° Bottom 85E 973 752 G Cod. C Blue Cod. Up With CPA',
  'step': 44,
  'time': 16.29,
  'workers': 0.5},
 {'item': 4,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Single Wires',
  'step': 62,
  'time': 7.88,
  'workers': 0.5},
 {'item': 4,
  'machine': 'Cutting Automation',
  'name': 'Cutting Automation',
  'step': 67,
  'time': 8.0,
  'workers': 0.5},
 {'item': 4,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 53.48,
  'workers': 0.5},
 {'item': 4,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 0.89,
  'workers': 0.5},
 {'item': 5,
  'machine': 'Sigma 688 / Alpha 488',
  'name': 'Cutting Stripping Crimping Twisting Single Wires',
  'step': 1,
  'time': 7.19,
  'workers': 0.5},
 {'item': 5,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 60mm-180mm',
  'step': 20,
  'time': 5.93,
  'workers': 0.5},
 {'item': 5,
  'machine': 'Connector Assembly Station',
  'name': 'Assembly DCC Connector Manually Single Wires',
  'step': 31,
  'time': 8.09,
  'workers': 0.5},
 {'item': 5,
  'machine': 'ARBURG 375ST Machine 11,12,13',
  'name': 'Overmolding 90° Right 4P0 973 752 B Cod. A Black Cod. Up With CPA',
  'step': 45,
  'time': 20.0,
  'workers': 0.5},
 {'item': 5,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Single Wires',
  'step': 62,
  'time': 7.88,
  'workers': 0.5},
 {'item': 5,
  'machine': 'Cutting Automation',
  'name': 'Cutting Automation',
  'step': 67,
  'time': 8.0,
  'workers': 0.5},
 {'item': 5,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 57.09,
  'workers': 0.5},
 {'item': 5,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 0.95,
  'workers': 0.5},
 {'item': 6,
  'machine': 'Sigma 688 / Alpha 488',
  'name': 'Cutting Stripping Crimping Twisting Single Wires',
  'step': 1,
  'time': 7.26,
  'workers': 0.5},
 {'item': 6,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm Tube + Grommet 201mm-300mm',
  'step': 24,
  'time': 12.41,
  'workers': 0.5},
 {'item': 6,
  'machine': 'Connector Assembly Station',
  'name': 'Assembly DCC Connector Manually Single Wires',
  'step': 31,
  'time': 8.09,
  'workers': 0.5},
 {'item': 6,
  'machine': 'ARBURG 375ST Machine 11,12,13',
  'name': 'Overmolding 60° Left 9J1 973 752 Cod. C Blue Cod. Up With CPA',
  'step': 54,
  'time': 18.72,
  'workers': 0.5},
 {'item': 6,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Single Wires',
  'step': 62,
  'time': 7.88,
  'workers': 0.5},
 {'item': 6,
  'machine': 'Cutting Automation',
  'name': 'Cutting Automation',
  'step': 67,
  'time': 8.0,
  'workers': 0.5},
 {'item': 6,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 62.36,
  'workers': 0.5},
 {'item': 6,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 1.04,
  'workers': 0.5},
 {'item': 7,
  'machine': 'Sigma 688 / Alpha 488',
  'name': 'Cutting Stripping Crimping Twisting Single Wires',
  'step': 1,
  'time': 7.13,
  'workers': 0.5},
 {'item': 7,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 181mm-280mm',
  'step': 21,
  'time': 8.76,
  'workers': 0.5},
 {'item': 7,
  'machine': 'Connector Assembly Station',
  'name': 'Assembly DCC Connector Manually Single Wires',
  'step': 31,
  'time': 8.09,
  'workers': 0.5},
 {'item': 7,
  'machine': 'ARBURG 375ST Machine 11,12,13',
  'name': 'Overmolding 60° Right 95C 973 752 B Cod. A Black Cod. Down With CPA',
  'step': 58,
  'time': 18.72,
  'workers': 0.5},
 {'item': 7,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Single Wires',
  'step': 62,
  'time': 7.88,
  'workers': 0.5},
 {'item': 7,
  'machine': 'Cutting Automation',
  'name': 'Cutting Automation',
  'step': 67,
  'time': 8.0,
  'workers': 0.5},
 {'item': 7,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 58.58,
  'workers': 0.5},
 {'item': 7,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 0.98,
  'workers': 0.5},
 {'item': 8,
  'machine': 'Sigma 688 / Alpha 488',
  'name': 'Cutting Stripping Crimping Twisting Single Wires',
  'step': 1,
  'time': 7.22,
  'workers': 0.5},
 {'item': 8,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm Tube + Grommet 201mm-300mm',
  'step': 24,
  'time': 12.41,
  'workers': 0.5},
 {'item': 8,
  'machine': 'Connector Assembly Station',
  'name': 'Assembly DCC Connector Manually Single Wires',
  'step': 31,
  'time': 8.09,
  'workers': 0.5},
 {'item': 8,
  'machine': 'ARBURG 375ST Machine 11,12,13',
  'name': 'Overmolding 60° Left 95C 93 752 C Cod. A Black Cod. Down With CPA',
  'step': 55,
  'time': 18.72,
  'workers': 0.5},
 {'item': 8,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Single Wires',
  'step': 62,
  'time': 7.88,
  'workers': 0.5},
 {'item': 8,
  'machine': 'Cutting Automation',
  'name': 'Cutting Automation',
  'step': 67,
  'time': 8.0,
  'workers': 0.5},
 {'item': 8,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 62.32,
  'workers': 0.5},
 {'item': 8,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 1.04,
  'workers': 0.5},
 {'item': 9,
  'machine': 'Sigma 688 / Alpha 488',
  'name': 'Cutting Stripping Crimping Twisting Single Wires',
  'step': 1,
  'time': 7.15,
  'workers': 0.5},
 {'item': 9,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 60mm-180mm',
  'step': 20,
  'time': 5.93,
  'workers': 0.5},
 {'item': 9,
  'machine': 'Connector Assembly Station',
  'name': 'Assembly DCC Connector Manually Single Wires',
  'step': 31,
  'time': 8.09,
  'workers': 0.5},
 {'item': 9,
  'machine': 'ARBURG 375ST Machine 1-5',
  'name': 'Overmolding 180° Straight 95C 973 752 D Cod. B White Cod. Up With '
          'CPA',
  'step': 37,
  'time': 17.91,
  'workers': 0.5},
 {'item': 9,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Single Wires',
  'step': 62,
  'time': 7.88,
  'workers': 0.5},
 {'item': 9,
  'machine': 'Cutting Automation',
  'name': 'Cutting Automation',
  'step': 67,
  'time': 8.0,
  'workers': 0.5},
 {'item': 9,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 54.96,
  'workers': 0.5},
 {'item': 9,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 0.92,
  'workers': 0.5},
 {'item': 10,
  'machine': 'Sigma 688 / Alpha 488',
  'name': 'Cutting Stripping Crimping Twisting Single Wires',
  'step': 1,
  'time': 7.18,
  'workers': 0.5},
 {'item': 10,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 281mm-380mm',
  'step': 22,
  'time': 11.0,
  'workers': 0.5},
 {'item': 10,
  'machine': 'Connector Assembly Station',
  'name': 'Assembly DCC Connector Manually Single Wires',
  'step': 31,
  'time': 8.09,
  'workers': 0.5},
 {'item': 10,
  'machine': 'ARBURG 375ST Machine 1-5',
  'name': 'Overmolding 180° Straight 95C 973 752 D Cod. B White Cod. Up With '
          'CPA',
  'step': 37,
  'time': 17.91,
  'workers': 0.5},
 {'item': 10,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Single Wires',
  'step': 62,
  'time': 7.88,
  'workers': 0.5},
 {'item': 10,
  'machine': 'Cutting Automation',
  'name': 'Cutting Automation',
  'step': 67,
  'time': 8.0,
  'workers': 0.5},
 {'item': 10,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 60.06,
  'workers': 0.5},
 {'item': 10,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 1.0,
  'workers': 0.5},
 {'item': 11,
  'machine': 'Sigma 688 / Alpha 488',
  'name': 'Cutting Stripping Crimping Twisting Single Wires',
  'step': 1,
  'time': 7.3,
  'workers': 0.5},
 {'item': 11,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 60mm-180mm',
  'step': 20,
  'time': 5.93,
  'workers': 0.5},
 {'item': 11,
  'machine': 'Connector Assembly Station',
  'name': 'Assembly DCC Connector Manually Single Wires',
  'step': 31,
  'time': 8.09,
  'workers': 0.5},
 {'item': 11,
  'machine': 'ARBURG 375ST Machine 6-10',
  'name': 'Overmolding 30° Left 9Y4 973 752 Cod. A Black Cod. Up With CPA',
  'step': 59,
  'time': 20.0,
  'workers': 0.5},
 {'item': 11,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Single Wires',
  'step': 62,
  'time': 7.88,
  'workers': 0.5},
 {'item': 11,
  'machine': 'Cutting Automation',
  'name': 'Cutting Automation',
  'step': 67,
  'time': 8.0,
  'workers': 0.5},
 {'item': 11,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 57.2,
  'workers': 0.5},
 {'item': 11,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 0.95,
  'workers': 0.5},
 {'item': 12,
  'machine': 'Sigma 688 / Alpha 488',
  'name': 'Cutting Stripping Crimping Twisting Single Wires',
  'step': 1,
  'time': 7.29,
  'workers': 0.5},
 {'item': 12,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 60mm-180mm',
  'step': 20,
  'time': 5.93,
  'workers': 0.5},
 {'item': 12,
  'machine': 'Connector Assembly Station',
  'name': 'Assembly DCC Connector Manually Single Wires',
  'step': 31,
  'time': 8.09,
  'workers': 0.5},
 {'item': 12,
  'machine': 'ARBURG 375ST Machine 6-10',
  'name': 'Overmolding 90° Bottom 85E 973 752 G Cod. C Blue Cod. Up With CPA',
  'step': 44,
  'time': 16.29,
  'workers': 0.5},
 {'item': 12,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Single Wires',
  'step': 62,
  'time': 7.88,
  'workers': 0.5},
 {'item': 12,
  'machine': 'Cutting Automation',
  'name': 'Cutting Automation',
  'step': 67,
  'time': 8.0,
  'workers': 0.5},
 {'item': 12,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 53.48,
  'workers': 0.5},
 {'item': 12,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 0.89,
  'workers': 0.5},
 {'item': 13,
  'machine': 'Sigma 688 / Alpha 488',
  'name': 'Cutting Stripping Crimping Twisting Single Wires',
  'step': 1,
  'time': 7.26,
  'workers': 0.5},
 {'item': 13,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm Tube + Grommet 201mm-300mm',
  'step': 24,
  'time': 12.41,
  'workers': 0.5},
 {'item': 13,
  'machine': 'Connector Assembly Station',
  'name': 'Assembly DCC Connector Manually Single Wires',
  'step': 31,
  'time': 8.09,
  'workers': 0.5},
 {'item': 13,
  'machine': 'ARBURG 375ST Machine 11,12,13',
  'name': 'Overmolding 60° Right 9J1 973 752 A Cod. C Blue Cod. Up With CPA',
  'step': 57,
  'time': 18.72,
  'workers': 0.5},
 {'item': 13,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Single Wires',
  'step': 62,
  'time': 7.88,
  'workers': 0.5},
 {'item': 13,
  'machine': 'Cutting Automation',
  'name': 'Cutting Automation',
  'step': 67,
  'time': 8.0,
  'workers': 0.5},
 {'item': 13,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 62.36,
  'workers': 0.5},
 {'item': 13,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 1.04,
  'workers': 0.5},
 {'item': 14,
  'machine': 'Sigma 688 / Alpha 488',
  'name': 'Cutting Stripping Crimping Twisting Single Wires',
  'step': 1,
  'time': 7.19,
  'workers': 0.5},
 {'item': 14,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 60mm-180mm',
  'step': 20,
  'time': 5.93,
  'workers': 0.5},
 {'item': 14,
  'machine': 'Connector Assembly Station',
  'name': 'Assembly DCC Connector Manually Single Wires',
  'step': 31,
  'time': 8.09,
  'workers': 0.5},
 {'item': 14,
  'machine': 'ARBURG 375ST Machine 11,12,13',
  'name': 'Overmolding 90° Left 4P0 973 752 A Cod. A Black Cod. Up With CPA',
  'step': 49,
  'time': 20.0,
  'workers': 0.5},
 {'item': 14,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Single Wires',
  'step': 62,
  'time': 7.88,
  'workers': 0.5},
 {'item': 14,
  'machine': 'Cutting Automation',
  'name': 'Cutting Automation',
  'step': 67,
  'time': 8.0,
  'workers': 0.5},
 {'item': 14,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 57.09,
  'workers': 0.5},
 {'item': 14,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 0.95,
  'workers': 0.5},
 {'item': 15,
  'machine': 'Sigma 688 / Alpha 488',
  'name': 'Cutting Stripping Crimping Twisting Single Wires',
  'step': 1,
  'time': 7.22,
  'workers': 0.5},
 {'item': 15,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm Tube + Grommet 201mm-300mm',
  'step': 24,
  'time': 12.41,
  'workers': 0.5},
 {'item': 15,
  'machine': 'Connector Assembly Station',
  'name': 'Assembly DCC Connector Manually Single Wires',
  'step': 31,
  'time': 8.09,
  'workers': 0.5},
 {'item': 15,
  'machine': 'ARBURG 375ST Machine 11,12,13',
  'name': 'Overmolding 60° Right 95C 973 752 B Cod. A Black Cod. Down With CPA',
  'step': 58,
  'time': 18.72,
  'workers': 0.5},
 {'item': 15,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Single Wires',
  'step': 62,
  'time': 7.88,
  'workers': 0.5},
 {'item': 15,
  'machine': 'Cutting Automation',
  'name': 'Cutting Automation',
  'step': 67,
  'time': 8.0,
  'workers': 0.5},
 {'item': 15,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 62.32,
  'workers': 0.5},
 {'item': 15,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 1.04,
  'workers': 0.5},
 {'item': 16,
  'machine': 'Sigma 688 / Alpha 488',
  'name': 'Cutting Stripping Crimping Twisting Single Wires',
  'step': 1,
  'time': 7.13,
  'workers': 0.5},
 {'item': 16,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 181mm-280mm',
  'step': 21,
  'time': 8.76,
  'workers': 0.5},
 {'item': 16,
  'machine': 'Connector Assembly Station',
  'name': 'Assembly DCC Connector Manually Single Wires',
  'step': 31,
  'time': 8.09,
  'workers': 0.5},
 {'item': 16,
  'machine': 'ARBURG 375ST Machine 11,12,13',
  'name': 'Overmolding 60° Left 95C 93 752 C Cod. A Black Cod. Down With CPA',
  'step': 55,
  'time': 18.72,
  'workers': 0.5},
 {'item': 16,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Single Wires',
  'step': 62,
  'time': 7.88,
  'workers': 0.5},
 {'item': 16,
  'machine': 'Cutting Automation',
  'name': 'Cutting Automation',
  'step': 67,
  'time': 8.0,
  'workers': 0.5},
 {'item': 16,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 58.58,
  'workers': 0.5},
 {'item': 16,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 0.98,
  'workers': 0.5},
 {'item': 17,
  'machine': 'Sigma 688 / Alpha 488',
  'name': 'Cutting Stripping Crimping Twisting Single Wires',
  'step': 1,
  'time': 7.15,
  'workers': 0.5},
 {'item': 17,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 60mm-180mm',
  'step': 20,
  'time': 5.93,
  'workers': 0.5},
 {'item': 17,
  'machine': 'Connector Assembly Station',
  'name': 'Assembly DCC Connector Manually Single Wires',
  'step': 31,
  'time': 8.09,
  'workers': 0.5},
 {'item': 17,
  'machine': 'ARBURG 375ST Machine 1-5',
  'name': 'Overmolding 180° Straight 95C 973 752 D Cod. B White Cod. Up With '
          'CPA',
  'step': 37,
  'time': 17.91,
  'workers': 0.5},
 {'item': 17,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Single Wires',
  'step': 62,
  'time': 7.88,
  'workers': 0.5},
 {'item': 17,
  'machine': 'Cutting Automation',
  'name': 'Cutting Automation',
  'step': 67,
  'time': 8.0,
  'workers': 0.5},
 {'item': 17,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 54.96,
  'workers': 0.5},
 {'item': 17,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 0.92,
  'workers': 0.5},
 {'item': 18,
  'machine': 'Sigma 688 / Alpha 488',
  'name': 'Cutting Stripping Crimping Twisting Single Wires',
  'step': 1,
  'time': 7.18,
  'workers': 0.5},
 {'item': 18,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 281mm-380mm',
  'step': 22,
  'time': 11.0,
  'workers': 0.5},
 {'item': 18,
  'machine': 'Connector Assembly Station',
  'name': 'Assembly DCC Connector Manually Single Wires',
  'step': 31,
  'time': 8.09,
  'workers': 0.5},
 {'item': 18,
  'machine': 'ARBURG 375ST Machine 1-5',
  'name': 'Overmolding 180° Straight 95C 973 752 D Cod. B White Cod. Up With '
          'CPA',
  'step': 37,
  'time': 17.91,
  'workers': 0.5},
 {'item': 18,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Single Wires',
  'step': 62,
  'time': 7.88,
  'workers': 0.5},
 {'item': 18,
  'machine': 'Cutting Automation',
  'name': 'Cutting Automation',
  'step': 67,
  'time': 8.0,
  'workers': 0.5},
 {'item': 18,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 60.06,
  'workers': 0.5},
 {'item': 18,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 1.0,
  'workers': 0.5},
 {'item': 19,
  'machine': 'Kappa 350 / Kappa 330',
  'name': 'Cutting Stripping Jacket Cable 4-Wire',
  'step': 4,
  'time': 6.95,
  'workers': 0.5},
 {'item': 19,
  'machine': 'Wire Cut & Separating Station',
  'name': 'Separating & Cutting Wires to Length 1 of 2 Pairs',
  'step': 10,
  'time': 8.12,
  'workers': 0.5},
 {'item': 19,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 60mm-180mm',
  'step': 20,
  'time': 5.93,
  'workers': 0.5},
 {'item': 19,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm Tube + Grommet 201mm-300mm',
  'step': 24,
  'time': 12.41,
  'workers': 0.5},
 {'item': 19,
  'machine': 'SKM DCPC Crimp (Crimp and Ass)',
  'name': 'Crimping & Assembly DCC Connector',
  'step': 30,
  'time': 20.0,
  'workers': 0.5},
 {'item': 19,
  'machine': 'ARBURG 375ST Machine 6-10',
  'name': 'Overmolding 90° Bottom 85E 973 752 F Cod. B White Cod. Down With '
          'CPA',
  'step': 43,
  'time': 16.29,
  'workers': 0.5},
 {'item': 19,
  'machine': 'ARBURG 375ST Machine 11,12,13',
  'name': 'Overmolding 60° Right 85E 973 752 Cod. B Black Cod. Up With CPA',
  'step': 56,
  'time': 18.72,
  'workers': 0.5},
 {'item': 19,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Jacket Cable',
  'step': 61,
  'time': 19.58,
  'workers': 0.5},
 {'item': 19,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 108.0,
  'workers': 0.5},
 {'item': 19,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 1.8,
  'workers': 0.5},
 {'item': 20,
  'machine': 'Kappa 350 / Kappa 330',
  'name': 'Cutting Stripping Jacket Cable 5-Wire',
  'step': 5,
  'time': 6.95,
  'workers': 0.5},
 {'item': 20,
  'machine': 'Wire Cut & Separating Station',
  'name': 'Separating & Cutting Wires to Length 2 of 3 Pairs',
  'step': 11,
  'time': 12.0,
  'workers': 0.5},
 {'item': 20,
  'machine': 'Wire Cut & Separating Station',
  'name': 'Remove Filler or Fließ from Jacket Cable',
  'step': 14,
  'time': 5.44,
  'workers': 0.5},
 {'item': 20,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 60mm-180mm',
  'step': 20,
  'time': 5.93,
  'workers': 0.5},
 {'item': 20,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm Tube + Grommet 201mm-300mm',
  'step': 24,
  'time': 12.41,
  'workers': 0.5},
 {'item': 20,
  'machine': 'SKM DCPC Crimp (Crimp and Ass)',
  'name': 'Crimping & Assembly DCC Connector',
  'step': 30,
  'time': 20.0,
  'workers': 0.5},
 {'item': 20,
  'machine': 'ARBURG 375ST Machine 6-10',
  'name': 'Overmolding 90° Bottom 85E 973 752 F Cod. B White Cod. Down With '
          'CPA',
  'step': 43,
  'time': 16.29,
  'workers': 0.5},
 {'item': 20,
  'machine': 'ARBURG 375ST Machine 11,12,13',
  'name': 'Overmolding 60° Right 85E 973 752 Cod. B Black Cod. Up With CPA',
  'step': 56,
  'time': 18.72,
  'workers': 0.5},
 {'item': 20,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Jacket Cable',
  'step': 61,
  'time': 19.58,
  'workers': 0.5},
 {'item': 20,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 117.32,
  'workers': 0.5},
 {'item': 20,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 1.96,
  'workers': 0.5},
 {'item': 21,
  'machine': 'Kappa 350 / Kappa 330',
  'name': 'Cutting Stripping Jacket Cable 5-Wire',
  'step': 5,
  'time': 6.92,
  'workers': 0.5},
 {'item': 21,
  'machine': 'Wire Cut & Separating Station',
  'name': 'Separating & Cutting Wires to Length 2 of 3 Pairs',
  'step': 11,
  'time': 12.0,
  'workers': 0.5},
 {'item': 21,
  'machine': 'Wire Cut & Separating Station',
  'name': 'Remove Filler or Fließ from Jacket Cable',
  'step': 14,
  'time': 5.44,
  'workers': 0.5},
 {'item': 21,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 60mm-180mm',
  'step': 20,
  'time': 5.93,
  'workers': 0.5},
 {'item': 21,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 181mm-280mm',
  'step': 21,
  'time': 17.52,
  'workers': 0.5},
 {'item': 21,
  'machine': 'SKM DCPC Crimp (Crimp and Ass)',
  'name': 'Crimping & Assembly DCC Connector',
  'step': 30,
  'time': 30.0,
  'workers': 0.5},
 {'item': 21,
  'machine': 'ARBURG 375ST Machine 6-10',
  'name': 'Overmolding 90° Bottom 3Q0 973 752 Cod. A Black Cod. Up No CPA',
  'step': 39,
  'time': 16.29,
  'workers': 0.5},
 {'item': 21,
  'machine': 'ARBURG 375ST Machine 6-10',
  'name': 'Overmolding 90° Bottom 85E 973 752 G Cod. C Blue Cod. Up With CPA',
  'step': 44,
  'time': 16.29,
  'workers': 0.5},
 {'item': 21,
  'machine': 'ARBURG 375ST Machine 11,12,13',
  'name': 'Overmolding 60° Left 85E 973 752 A Cod. A Black Cod. Up With CPA',
  'step': 53,
  'time': 18.72,
  'workers': 0.5},
 {'item': 21,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Jacket Cable',
  'step': 61,
  'time': 29.37,
  'workers': 0.5},
 {'item': 21,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 158.48,
  'workers': 0.5},
 {'item': 21,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 2.64,
  'workers': 0.5},
 {'item': 22,
  'machine': 'Kappa 350 / Kappa 330',
  'name': 'Cutting Stripping Jacket Cable 5-Wire',
  'step': 5,
  'time': 6.95,
  'workers': 0.5},
 {'item': 22,
  'machine': 'Wire Cut & Separating Station',
  'name': 'Separating & Cutting Wires to Length 2 of 3 Pairs',
  'step': 11,
  'time': 12.0,
  'workers': 0.5},
 {'item': 22,
  'machine': 'Wire Cut & Separating Station',
  'name': 'Remove Filler or Fließ from Jacket Cable',
  'step': 14,
  'time': 5.44,
  'workers': 0.5},
 {'item': 22,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 60mm-180mm',
  'step': 20,
  'time': 5.93,
  'workers': 0.5},
 {'item': 22,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm Tube + Grommet 201mm-300mm',
  'step': 24,
  'time': 12.41,
  'workers': 0.5},
 {'item': 22,
  'machine': 'SKM DCPC Crimp (Crimp and Ass)',
  'name': 'Crimping & Assembly DCC Connector',
  'step': 30,
  'time': 20.0,
  'workers': 0.5},
 {'item': 22,
  'machine': 'ARBURG 375ST Machine 6-10',
  'name': 'Overmolding 90° Bottom 85E 973 752 E Cod. B White Cod. Up With CPA',
  'step': 42,
  'time': 16.29,
  'workers': 0.5},
 {'item': 22,
  'machine': 'ARBURG 375ST Machine 11,12,13',
  'name': 'Overmolding 60° Left 85E 973 752 A Cod. A Black Cod. Up With CPA',
  'step': 53,
  'time': 18.72,
  'workers': 0.5},
 {'item': 22,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Jacket Cable',
  'step': 61,
  'time': 19.58,
  'workers': 0.5},
 {'item': 22,
  'machine': '',
  'name': 'TOTAL(sec)',
  'step': 201,
  'time': 117.32,
  'workers': 0.5},
 {'item': 22,
  'machine': '',
  'name': 'SAP TIMES',
  'step': 202,
  'time': 1.96,
  'workers': 0.5},
 {'item': 23,
  'machine': 'Kappa 350 / Kappa 330',
  'name': 'Cutting Stripping Jacket Cable 5-Wire',
  'step': 5,
  'time': 6.92,
  'workers': 0.5},
 {'item': 23,
  'machine': 'Wire Cut & Separating Station',
  'name': 'Separating & Cutting Wires to Length 2 of 3 Pairs',
  'step': 11,
  'time': 12.0,
  'workers': 0.5},
 {'item': 23,
  'machine': 'Wire Cut & Separating Station',
  'name': 'Remove Filler or Fließ from Jacket Cable',
  'step': 14,
  'time': 5.44,
  'workers': 0.5},
 {'item': 23,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 60mm-180mm',
  'step': 20,
  'time': 5.93,
  'workers': 0.5},
 {'item': 23,
  'machine': 'PUR-Tube Assembly Station',
  'name': 'Assembly PUR-Tube 3,5x1,35mm PUR-Tube 181mm-280mm',
  'step': 21,
  'time': 17.52,
  'workers': 0.5},
 {'item': 23,
  'machine': 'SKM DCPC Crimp (Crimp and Ass)',
  'name': 'Crimping & Assembly DCC Connector',
  'step': 30,
  'time': 30.0,
  'workers': 0.5},
 {'item': 23,
  'machine': 'ARBURG 375ST Machine 6-10',
  'name': 'Overmolding 90° Bottom 3Q0 973 752 Cod. A Black Cod. Up No CPA',
  'step': 39,
  'time': 16.29,
  'workers': 0.5},
 {'item': 23,
  'machine': 'ARBURG 375ST Machine 6-10',
  'name': 'Overmolding 90° Bottom 85E 973 752 G Cod. C Blue Cod. Up With CPA',
  'step': 44,
  'time': 16.29,
  'workers': 0.5},
 {'item': 23,
  'machine': 'ARBURG 375ST Machine 11,12,13',
  'name': 'Overmolding 60° Right 85E 973 752 Cod. B Black Cod. Up With CPA',
  'step': 56,
  'time': 18.72,
  'workers': 0.5},
 {'item': 23,
  'machine': 'SKM Seal and outer Housing Assembly DCC',
  'name': 'Assembly Seal & Outer Housing Round Table Jacket Cable',
  'step': 61,
  'time': 29.37,
  'workers': 0.5}]


        
        # Create products and process steps
        for i in range(len(demand_data['Item'])):
            item = demand_data['Item'][i]
            batch_size = int(np.ceil(demand_data['Demand_2024'][i] / 12))
            
            product = Product.objects.create(
                item=item,
                sap_tn=str(demand_data['SAP_TN'][i]),
                sap_pl=str(demand_data['SAP_PL'][i]) if demand_data['SAP_PL'][i] else None,
                dcc_type=demand_data['DCC_Type'][i],
                description=demand_data['Description'][i],
                demand_2024=demand_data['Demand_2024'][i],
                batch_size=batch_size,
                num_batches=12
            )
            
            # Add process steps for this product
            for step_data in process_routing:
                if step_data['item'] == item:
                    ProcessStep.objects.create(
                        product=product,
                        step_number=step_data['step'],
                        machine=machines[step_data['machine']],
                        step_name=step_data['name'],
                        cycle_time_seconds=step_data['time'],
                        workers_required=step_data['workers']
                    )
        
        product_count = Product.objects.count()
        machine_count = Machine.objects.count()
        step_count = ProcessStep.objects.count()
        
        return Response({
            'message': 'Database initialized successfully',
            'products_created': product_count,
            'machines_created': machine_count,
            'process_steps_created': step_count
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        print(str(e))
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_buffer_optimization(request):
    """
    Calculate optimal buffer sizes for each machine
    """
    try:
        schedules = ProductionSchedule.objects.all()
        
        if not schedules.exists():
            return Response({
                'message': 'No schedules available'
            }, status=status.HTTP_200_OK)
        
        # Get safety factor from request (default 1.5)
        safety_factor = float(request.GET.get('safety_factor', 1.5))
        
        # Calculate makespan and throughput
        min_start = schedules.aggregate(Min('start_time'))['start_time__min']
        max_end = schedules.aggregate(Max('end_time'))['end_time__max']
        makespan_hours = (max_end - min_start).total_seconds() / 3600
        
        total_units = Product.objects.filter(demand_2024__gt=0).aggregate(
            total=Sum('demand_2024')
        )['total'] or 0
        
        throughput_per_hour = (total_units / makespan_hours) if makespan_hours > 0 else 0
        
        # Calculate buffer for each machine
        machines = Machine.objects.all()
        buffer_recommendations = []
        
        for machine in machines:
            # Get average operation time (expected delay) for this machine
            machine_schedules = schedules.filter(machine=machine)
            
            if machine_schedules.exists():
                avg_duration_hours = machine_schedules.aggregate(
                    avg=Avg('duration_hours')
                )['avg'] or 0
                
                # Calculate buffer size
                # buffer = throughput_per_hour × expected_delay_hours × safety_factor
                buffer_units = throughput_per_hour * avg_duration_hours * safety_factor
                
                # Get total operations and utilization
                total_operations = machine_schedules.count()
                used_hours = machine_schedules.aggregate(
                    total=Sum('duration_hours')
                )['total'] or 0
                utilization = (used_hours / makespan_hours * 100) if makespan_hours > 0 else 0
                
                buffer_recommendations.append({
                    'machine': machine.name,
                    'buffer_size_units': round(buffer_units, 2),
                    'avg_operation_time_hours': round(avg_duration_hours, 4),
                    'throughput_per_hour': round(throughput_per_hour, 2),
                    'safety_factor': safety_factor,
                    'utilization': round(utilization, 2),
                    'total_operations': total_operations,
                    'recommendation': 'HIGH PRIORITY' if utilization > 80 else 'MEDIUM PRIORITY' if utilization > 60 else 'LOW PRIORITY'
                })
        
        # Sort by buffer size (descending) - machines needing larger buffers first
        buffer_recommendations_sorted = sorted(
            buffer_recommendations, 
            key=lambda x: x['buffer_size_units'], 
            reverse=True
        )
        
        return Response({
            'buffer_recommendations': buffer_recommendations_sorted,
            'parameters': {
                'throughput_per_hour': round(throughput_per_hour, 2),
                'makespan_hours': round(makespan_hours, 2),
                'safety_factor': safety_factor,
                'total_units': total_units
            },
            'formula': 'buffer_units = throughput_per_hour × avg_operation_time_hours × safety_factor'
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_bottleneck_analysis(request):
    """
    Detailed bottleneck analysis with recommendations
    """
    try:
        schedules = ProductionSchedule.objects.all()
        
        if not schedules.exists():
            return Response({
                'message': 'No schedules available'
            }, status=status.HTTP_200_OK)
        
        # Calculate makespan
        min_start = schedules.aggregate(Min('start_time'))['start_time__min']
        max_end = schedules.aggregate(Max('end_time'))['end_time__max']
        makespan_hours = (max_end - min_start).total_seconds() / 3600
        
        # Analyze each machine
        machines = Machine.objects.all()
        bottleneck_analysis = []
        
        for machine in machines:
            machine_schedules = schedules.filter(machine=machine)
            
            if machine_schedules.exists():
                used_hours = machine_schedules.aggregate(
                    total=Sum('duration_hours')
                )['total'] or 0
                
                utilization = (used_hours / makespan_hours * 100) if makespan_hours > 0 else 0
                num_operations = machine_schedules.count()
                
                avg_operation_time = machine_schedules.aggregate(
                    avg=Avg('duration_hours')
                )['avg'] or 0
                
                # Idle time
                idle_hours = makespan_hours - used_hours
                idle_percentage = (idle_hours / makespan_hours * 100) if makespan_hours > 0 else 0
                
                # Get products processed on this machine
                products_on_machine = machine_schedules.values_list(
                    'product__item', flat=True
                ).distinct().count()
                
                # Determine bottleneck status
                if utilization >= 85:
                    status_label = 'CRITICAL BOTTLENECK'
                    recommendation = 'Consider adding capacity, optimizing setups, or redistributing work'
                elif utilization >= 70:
                    status_label = 'POTENTIAL BOTTLENECK'
                    recommendation = 'Monitor closely, consider process improvements'
                elif utilization >= 50:
                    status_label = 'WELL UTILIZED'
                    recommendation = 'Operating efficiently'
                else:
                    status_label = 'UNDERUTILIZED'
                    recommendation = 'Opportunity to consolidate operations or reduce capacity'
                
                bottleneck_analysis.append({
                    'machine': machine.name,
                    'utilization': round(utilization, 2),
                    'used_hours': round(used_hours, 2),
                    'idle_hours': round(idle_hours, 2),
                    'idle_percentage': round(idle_percentage, 2),
                    'num_operations': num_operations,
                    'avg_operation_time_hours': round(avg_operation_time, 4),
                    'products_processed': products_on_machine,
                    'status': status_label,
                    'recommendation': recommendation
                })
        
        # Sort by utilization (highest first)
        bottleneck_analysis_sorted = sorted(
            bottleneck_analysis, 
            key=lambda x: x['utilization'], 
            reverse=True
        )
        
        # Overall summary
        summary = {
            'total_makespan_hours': round(makespan_hours, 2),
            'bottleneck_machine': bottleneck_analysis_sorted[0]['machine'] if bottleneck_analysis_sorted else None,
            'bottleneck_utilization': bottleneck_analysis_sorted[0]['utilization'] if bottleneck_analysis_sorted else 0,
            'avg_utilization': round(np.mean([m['utilization'] for m in bottleneck_analysis]), 2) if bottleneck_analysis else 0,
            'total_machines': len(bottleneck_analysis)
        }
        
        return Response({
            'summary': summary,
            'machine_analysis': bottleneck_analysis_sorted
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
def calculate_optimal_batch_size(demand, max_num_batches=25, min_batch_size=50, max_batch_size=500):
    """
    Calculate optimal batch size based on demand
    
    Args:
        demand: Total demand for the product
        max_num_batches: Maximum number of batches allowed
        min_batch_size: Minimum size per batch
        max_batch_size: Maximum size per batch
    
    Returns:
        tuple: (batch_size, num_batches, avg_batch_size_used)
    """
    if demand <= 0:
        return 0, 0, 0
    
    # Calculate ideal batch size
    ideal_batch_size = demand / max_num_batches
    
    # Adjust based on constraints
    if ideal_batch_size < min_batch_size:
        # If ideal is too small, use min_batch_size
        batch_size = min_batch_size
        num_batches = int(np.ceil(demand / batch_size))
    elif ideal_batch_size > max_batch_size:
        # If ideal is too large, use max_batch_size
        batch_size = max_batch_size
        num_batches = int(np.ceil(demand / batch_size))
    else:
        # Use a balanced approach
        # Try to get batch size close to ideal while keeping reasonable number of batches
        num_batches = max(1, int(np.ceil(demand / ideal_batch_size)))
        batch_size = int(np.ceil(demand / num_batches))
    
    # Ensure we don't exceed max_num_batches
    if num_batches > max_num_batches:
        num_batches = max_num_batches
        batch_size = int(np.ceil(demand / num_batches))
    
    return batch_size, num_batches, ideal_batch_size


@api_view(['GET'])
def get_batch_optimization_preview(request):
    """
    Preview batch size optimization for all products
    """
    try:
        max_num_batches = int(request.GET.get('max_num_batches', 25))
        min_batch_size = int(request.GET.get('min_batch_size', 50))
        max_batch_size = int(request.GET.get('max_batch_size', 500))
        
        products = Product.objects.filter(demand_2024__gt=0)
        
        batch_analysis = []
        total_demand = 0
        total_batches = 0
        batch_sizes = []
        
        for product in products:
            demand = product.demand_2024
            
            # Calculate optimal batch size
            batch_size, num_batches, ideal_batch = calculate_optimal_batch_size(
                demand, max_num_batches, min_batch_size, max_batch_size
            )
            
            # Old method (fixed 12 batches)
            old_batch_size = int(np.ceil(demand / 12))
            old_num_batches = 12
            
            batch_analysis.append({
                'item': product.item,
                'description': product.description[:50],
                'demand': demand,
                'old_batch_size': old_batch_size,
                'old_num_batches': old_num_batches,
                'new_batch_size': batch_size,
                'new_num_batches': num_batches,
                'ideal_batch_size': round(ideal_batch, 2),
                'improvement': f"{((old_num_batches - num_batches) / old_num_batches * 100):.1f}%" if old_num_batches != num_batches else "0%"
            })
            
            total_demand += demand
            total_batches += num_batches
            batch_sizes.append(batch_size)
        
        # Calculate statistics
        avg_batch_size = np.mean(batch_sizes) if batch_sizes else 0
        min_batch = np.min(batch_sizes) if batch_sizes else 0
        max_batch = np.max(batch_sizes) if batch_sizes else 0
        std_batch = np.std(batch_sizes) if batch_sizes else 0
        
        return Response({
            'batch_analysis': batch_analysis,
            'summary': {
                'total_products': len(batch_analysis),
                'total_demand': total_demand,
                'total_batches': total_batches,
                'avg_batch_size': round(avg_batch_size, 2),
                'min_batch_size': min_batch,
                'max_batch_size': max_batch,
                'std_batch_size': round(std_batch, 2)
            },
            'parameters': {
                'max_num_batches': max_num_batches,
                'min_batch_size': min_batch_size,
                'max_batch_size': max_batch_size
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)