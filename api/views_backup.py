from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd
import numpy as np


@api_view(['POST'])
def process_csv(request):
    csv_file = request.FILES.get('file')
    if not csv_file:
        return Response({'error': 'No file uploaded'}, status=400)

    num_shifts = int(request.POST.get("num_shifts", 28))
    
    # Get filter parameters
    filter_pps_tn = request.POST.get("pps_tn", "All")
    filter_project = request.POST.get("project", "All")
    filter_sub_project = request.POST.get("sub_project", "All")
    filter_machine = request.POST.get("machine", "All")
    filter_tool_no = request.POST.get("tool_no", "All")
    filter_area = request.POST.get("area", "All")
    
    # Read CSV
    df = pd.read_csv(csv_file)

    # Clean numeric columns
    numeric_cols = ['Planned', 'Realized', 'Backlog', 'Open']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce')

    # Clean text columns
    df['Step'] = df['Step'].astype(str).str.strip()
    df['Area'] = df['Area'].astype(str).str.strip()
    df['Sub-Project'] = df['Sub-Project'].astype(str).str.strip()
    
    # Clean filter columns if they exist
    filter_columns = ['PPS TN', 'Project', 'Sub-Project', 'Machine', 'Tool No.', 'Area']
    for col in filter_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Apply filters
    if filter_pps_tn != "All" and 'PPS TN' in df.columns:
        df = df[df['PPS TN'] == filter_pps_tn]
    
    if filter_project != "All" and 'Project' in df.columns:
        df = df[df['Project'] == filter_project]
    
    if filter_sub_project != "All" and 'Sub-Project' in df.columns:
        df = df[df['Sub-Project'] == filter_sub_project]
    
    if filter_machine != "All" and 'Machine' in df.columns:
        df = df[df['Machine'] == filter_machine]
    
    if filter_tool_no != "All" and 'Tool No.' in df.columns:
        df = df[df['Tool No.'] == filter_tool_no]
    
    if filter_area != "All" and 'Area' in df.columns:
        df = df[df['Area'] == filter_area]

    # Filters
    H = df['Step']
    G = df['Area']
    D = df['Sub-Project']

    # Define 36 shift labels
    shift_labels = [
        'Shift 1', 'Shift 1 B', 'Shift 2', 'Shift 2 A', 'Shift 3', 'Shift 3 C',
        'Shift 4', 'Shift 4 B', 'Shift 5', 'Shift 5 A', 'Shift 6', 'Shift 6 C',
        'Shift 7', 'Shift 7 B', 'Shift 8', 'Shift 8 A', 'Shift 9', 'Shift 9 C',
        'Shift 10', 'Shift 10 B', 'Shift 11', 'Shift 11 A', 'Shift 12', 'Shift 12 C',
        'Shift 13', 'Shift 13 B', 'Shift 14', 'Shift 14 A', 'Shift 15', 'Shift 15 C',
        'Shift 16', 'Shift 16 B', 'Shift 17', 'Shift 17 A', 'Shift 18', 'Shift 18 C'
    ]

    # Clean numeric shift columns
    for idx in range(14, 50):
        df.iloc[:, idx] = pd.to_numeric(
            df.iloc[:, idx].astype(str).str.replace(r'[^\d\.\-]', '', regex=True),
            errors='coerce'
        )
    for idx in range(95, 113):
        df.iloc[:, idx] = pd.to_numeric(
            df.iloc[:, idx].astype(str).str.replace(r'[^\d\.\-]', '', regex=True),
            errors='coerce'
        )

    # Calculate Overall Efficiency
    efficiency_list = []
    
    if 'STD' in df.columns:
        std_col = df.loc[2:1003, 'STD']
        std_col = pd.to_numeric(
            std_col.astype(str).str.replace(r'[^\d\.\-]', '', regex=True),
            errors='coerce'
        )
        
        quantity_col_indices = list(range(15, 52, 2))
        shift_time_hours = 7.67
        available_time = shift_time_hours * num_shifts
        
        for col_idx in quantity_col_indices:
            quantity = df.iloc[2:1003, col_idx]
            quantity = pd.to_numeric(
                quantity.astype(str).str.replace(r'[^\d\.\-]', '', regex=True),
                errors='coerce'
            )
            
            valid_mask = quantity.notna() & std_col.notna()
            valid_quantity = quantity[valid_mask]
            valid_std = std_col[valid_mask]
            
            total_planned_time_minutes = (valid_quantity * valid_std).sum()
            total_planned_time_hours = total_planned_time_minutes / 60
            
            if available_time > 0:
                efficiency = (total_planned_time_hours / available_time) * 100
            else:
                efficiency = 0
            
            efficiency_list.append(efficiency)
    
    # Insert 0 between each efficiency value
    efficiency_with_zeros = []
    for eff in efficiency_list:
        efficiency_with_zeros.append(eff)
        efficiency_with_zeros.append(0)
    
    while len(efficiency_with_zeros) < 36:
        efficiency_with_zeros.append(0)
    efficiency_with_zeros = efficiency_with_zeros[:36]

    # Initialize result dictionary
    result = {
        'Total Backlog Finished Goods': {},
        'Production Output Finished Goods': {},
        'Production Output Connectors': {},
        'Overall Efficiency': {}
    }

    # Process backlog
    backlog_cols = list(range(95, 113))
    backlog_values = []
    for col_idx in backlog_cols:
        excel_range = df.iloc[2:264, col_idx]
        total_backlog = excel_range.loc[excel_range > 0].sum()
        backlog_values.append(total_backlog if total_backlog > 0 else 0)

    backlog_with_zeros = []
    for val in backlog_values:
        backlog_with_zeros.append(val)
        backlog_with_zeros.append(0)

    while len(backlog_with_zeros) < len(shift_labels):
        backlog_with_zeros.append(0)

    for shift_label, val in zip(shift_labels, backlog_with_zeros):
        result['Total Backlog Finished Goods'][shift_label] = f"{val:,.0f}" if val > 0 else "0"

    # Filters for production outputs
    finished_goods_filter = (H == 'F') & (D.notna()) & (D.astype(str).str.strip() != '')
    connectors_filter = (G == 'Assembly') & (D.notna()) & (D.astype(str).str.strip() != '')

    # Process production output and efficiency
    for i, shift_label in enumerate(shift_labels):
        col_idx = 14 + i
        if col_idx < 50:
            fg_value = df.iloc[:, col_idx][finished_goods_filter].sum()
            conn_value = df.iloc[:, col_idx][connectors_filter].sum()
            result['Production Output Finished Goods'][shift_label] = f"{fg_value:,.0f}" if fg_value > 0 else "0"
            result['Production Output Connectors'][shift_label] = f"{conn_value:,.0f}" if conn_value > 0 else "0"
        else:
            result['Production Output Finished Goods'][shift_label] = "0"
            result['Production Output Connectors'][shift_label] = "0"
        
        result['Overall Efficiency'][shift_label] = f"{efficiency_with_zeros[i]:.2f}%" if efficiency_with_zeros[i] > 0 else "-"

    # Summary Output Table
    total_planned_fg = df[df['Step'] == 'F']['Planned'].sum()
    total_realized_fg = df[df['Step'] == 'F']['Realized'].sum()
    total_planned_conn = df[df['Area'] == 'Assembly']['Planned'].sum()
    total_realized_conn = df[df['Area'] == 'Assembly']['Realized'].sum()

    current_backlog_fg = df[(df['Backlog'] > 0) & (df['Step'] == 'F')]['Backlog'].sum()
    current_open_fg = df[df['Step'] == 'F']['Open'].sum()
    current_backlog_conn = df[(df['Backlog'] > 0) & (df['Area'] == 'Assembly')]['Backlog'].sum()
    current_open_conn = df[df['Area'] == 'Assembly']['Open'].sum()

    summary_table = {
        "Metric": ["Target Headcount", "Actual Headcount", "Current Backlog", "Currently Open"],
        "Finished Goods": [
            f"{total_planned_fg:,.0f}", f"{total_realized_fg:,.0f}",
            f"{current_backlog_fg:,.0f}", f"{current_open_fg:,.0f}"
        ],
        "Connectors": [
            f"{total_planned_conn:,.0f}", f"{total_realized_conn:,.0f}",
            f"{current_backlog_conn:,.0f}", f"{current_open_conn:,.0f}"
        ]
    }
    
    # Final response
    response_data = {
        "ShiftWise": result,
        "Summary": summary_table
    }

    return Response(response_data)


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
    Generate production schedule based on products and process steps
    """
    try:
        # Clear existing schedules
        ProductionSchedule.objects.all().delete()
        
        # Get all products with demand > 0
        products = Product.objects.filter(demand_2024__gt=0).order_by('item')
        
        if not products.exists():
            return Response({
                'error': 'No products with demand found'
            }, status=status.HTTP_400_BAD_REQUEST)
        
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
            'schedule_count': total_schedules
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
    Get KPI metrics
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
        
        # Machine utilization
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
        
        # Throughput
        total_units = Product.objects.filter(demand_2024__gt=0).aggregate(
            total=Sum('demand_2024')
        )['total'] or 0
        throughput_per_day = (total_units / makespan_days) if makespan_days > 0 else 0
        
        # Number of setups (count of operations)
        num_setups = schedules.count()
        
        return Response({
            'total_makespan_hours': round(makespan_hours, 2),
            'total_makespan_days': round(makespan_days, 2),
            'machine_utilization': machine_stats,
            'total_operations': num_setups,
            'throughput_units_per_day': round(throughput_per_day, 2),
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

        demand_data =    {'Item': [1,
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
  20],
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
  249090],
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
  249055],
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
  '60°'],
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
  201]}

        process_routing = [{'item': 2,
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