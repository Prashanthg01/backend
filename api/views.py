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
        machines_data = [
            'Sigma 688 / Alpha 488',
            'Alpha 550 / Alpha 433',
            'Kappa 350 / Kappa 330'
        ]
        
        machines = {}
        for machine_name in machines_data:
            machine = Machine.objects.create(
                name=machine_name,
                available_hours_per_day=24
            )
            machines[machine_name] = machine
        
        # Product data
        # Product data
        demand_data = {
            'Item': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 39, 40, 46, 47, 48, 49, 50, 51, 55, 56, 58, 59],
            'SAP_TN': [249313, 249314, 249316, 249316, 249078, 249079, 249080, 249081, 249082, 249083, 249084, 249085, 249086, 249087, 249088, 249089, 249090, 249091, 249092, 249093, 255164, 255166, 243866, 243867, 253644, 253651, 249102, 249103, 249108, 249109, 249110, 249111, 249115, 249116, 249118, 249119],
            'SAP_PL': [None, None, None, None, 249043, 249044, 249045, 249046, 249047, 249048, 249049, 249050, 249051, 249052, 249053, 249054, 249055, 249056, 249057, 249058, None, None, None, None, None, None, 234300, 234594, 234316, 234317, 234607, 234608, 234310, 234311, 234602, 234603],
            'DCC_Type': ['60° & 90°B', '60° & 90°B', '60° & 90°B', '60° & 90°B', '30°', '90°B', '90°', '60°', '60°', '60°', '180°', '180°', '30°', '90°B', '60°', '90°', '60°', '60°', '180°', '180°', '180°', '180°', '90°', '90°', '60°', '60°', '180°', '180°', '60°', '180°', '60°', '180°', '60°', '60°', '60°', '60°'],
            'Description': [
                '4 Wire Jacket 2xDCC Modul 9Y4251',
                '6 Wire Jacket 2xDCC Modul 9Y4251A',
                '6 Wire Jacket 3xDCC Modul 9Y4252',
                '6 Wire Jacket 3xDCC Modul 9Y4255',
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
                'Twisted Wires 1xDCC Modul 9Y4286 N,Q',
                'Twisted Wires 1xDCC Modul 512 AC, 513 AC',
                'Twisted Wires 1xDCC Modul 512, 513',
                'Twisted Wires 1xDCC Modul 510 AB,BB',
                'Twisted Wires 1xDCC Modul 511 AB,BB',
                'Twisted Wires 1xDCC Modul T3-512 AA',
                'Twisted Wires 1xDCC Modul T3-513 AA',
                'Twisted Wires 1xDCC Modul 9J0, 9J1 HL',
                'Twisted Wires 1xDCC Modul 9J0, 9J1 HR',
                'Twisted Wires 1xDCC Modul 9J1286 _,C',
                'Twisted Wires 1xDCC Modul 9J1286 _,C',
                'Twisted Wires 1xDCC Modul 9J1286 B,E',
                'Twisted Wires 1xDCC Modul 9J1286 B,E',
                'Twisted Wires 1xDCC Modul 9J1206 A',
                'Twisted Wires 1xDCC Modul 9J1206 A',
                'Twisted Wires 1xDCC Modul 9J1206 D',
                'Twisted Wires 1xDCC Modul 9J1206 D'
            ],
            'Demand_2024': [1141, 201, 1221, 1342, 1221, 1221, 120, 120, 1127, 214, 1127, 214, 1221, 1221, 120, 120, 201, 1140, 1140, 201, None, None, None, None, None, None, 14700, 14800, 12100, 12100, 12300, 12300, 2600, 2600, 2500, 2500]
        }

        # Process routing
        process_routing = [
            # Item 1 - 4 Wire Jacket 2xDCC Modul 9Y4251
            {'item': 1, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 7.30, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 2 - 6 Wire Jacket 2xDCC Modul 9Y4251A
            {'item': 2, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 7.29, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 3 - 6 Wire Jacket 3xDCC Modul 9Y4252
            {'item': 3, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 7.19, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 4 - 6 Wire Jacket 3xDCC Modul 9Y4255
            {'item': 4, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 7.26, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 5 - Twisted Wires 1xDCC Modul 9Y4279 AA,AB,AC (30°)
            {'item': 5, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 7.18, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 6 - Twisted Wires 1xDCC Modul 9Y4279 AD,AE,AF (60°)
            {'item': 6, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 7.30, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 7 - Twisted Wires 1xDCC Modul 9Y4279 AA,AD (60°)
            {'item': 7, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 7.29, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 8 - Twisted Wires 1xDCC Modul 9Y4279 AB,AC,AE,AF (60°)
            {'item': 8, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 7.26, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 9 - Twisted Wires 1xDCC Modul 9Y4279 AA,AD (180°)
            {'item': 9, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 7.19, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 10 - Twisted Wires 1xDCC Modul 9Y4279 AB,AC,AE,AF (180°)
            {'item': 10, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 7.22, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 11 - Twisted Wires 1xDCC Modul 9Y4286 M,N (60°)
            {'item': 11, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 7.13, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 12 - Twisted Wires 1xDCC Modul 9Y4286 M,N (90°B)
            {'item': 12, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 7.15, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 13 - Twisted Wires 1xDCC Modul 9Y4286 P,Q (90°)
            {'item': 13, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 7.18, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 14 - Twisted Wires 1xDCC Modul 9Y4286 P,Q (90°)
            {'item': 14, 'step': 4, 'machine': 'Kappa 350 / Kappa 330', 'time': 6.95, 'name': 'Cutting Stripping Jacket Cable 4-Wire', 'workers': 0.5},
            {'item': 14, 'step': 8, 'machine': 'Kappa 350 / Kappa 330', 'time': 8.12, 'name': 'Cutting Stripping Jacket Cable 8-Wire', 'workers': 0.5},
            
            # Item 15 - Twisted Wires 1xDCC Modul 9Y4286 N,Q (60°)
            {'item': 15, 'step': 5, 'machine': 'Kappa 350 / Kappa 330', 'time': 6.95, 'name': 'Cutting Stripping Jacket Cable 5-Wire', 'workers': 0.5},
            {'item': 15, 'step': 8, 'machine': 'Kappa 350 / Kappa 330', 'time': 12.00, 'name': 'Cutting Stripping Jacket Cable 8-Wire', 'workers': 0.5},
            
            # Item 16 - Twisted Wires 1xDCC Modul 9Y4286 M,P (60°)
            {'item': 16, 'step': 5, 'machine': 'Kappa 350 / Kappa 330', 'time': 6.92, 'name': 'Cutting Stripping Jacket Cable 5-Wire', 'workers': 0.5},
            {'item': 16, 'step': 8, 'machine': 'Kappa 350 / Kappa 330', 'time': 12.00, 'name': 'Cutting Stripping Jacket Cable 8-Wire', 'workers': 0.5},
            
            # Item 17 - Twisted Wires 1xDCC Modul 9Y4286 M,P (180°)
            {'item': 17, 'step': 5, 'machine': 'Kappa 350 / Kappa 330', 'time': 6.95, 'name': 'Cutting Stripping Jacket Cable 5-Wire', 'workers': 0.5},
            {'item': 17, 'step': 8, 'machine': 'Kappa 350 / Kappa 330', 'time': 12.00, 'name': 'Cutting Stripping Jacket Cable 8-Wire', 'workers': 0.5},
            
            # Item 18 - Twisted Wires 1xDCC Modul 9Y4286 N,Q (180°)
            {'item': 18, 'step': 5, 'machine': 'Kappa 350 / Kappa 330', 'time': 6.92, 'name': 'Cutting Stripping Jacket Cable 5-Wire', 'workers': 0.5},
            {'item': 18, 'step': 8, 'machine': 'Kappa 350 / Kappa 330', 'time': 12.00, 'name': 'Cutting Stripping Jacket Cable 8-Wire', 'workers': 0.5},
            
            # Item 19 - Twisted Wires 1xDCC Modul 512 AC, 513 AC (180°)
            {'item': 19, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 6.69, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 20 - Twisted Wires 1xDCC Modul 512, 513 (180°)
            {'item': 20, 'step': 2, 'machine': 'Alpha 550 / Alpha 433', 'time': 3.22, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 21 - Twisted Wires 1xDCC Modul 510 AB,BB (90°)
            {'item': 21, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 6.31, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 22 - Twisted Wires 1xDCC Modul 511 AB,BB (90°)
            {'item': 22, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 6.19, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 23 - Twisted Wires 1xDCC Modul 511 AB,BB (90°)
            {'item': 23, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 6.19, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 24 - 4 Wire Jacket 2xDCC Modul 511 BB (180° & 90°B)
            {'item': 24, 'step': 4, 'machine': 'Kappa 350 / Kappa 330', 'time': 6.59, 'name': 'Cutting Stripping Jacket Cable 4-Wire', 'workers': 0.5},
            
            # Item 25 - 7 Wire Jacket 2xDCC Modul 511 A9 (180° & 90°B)
            {'item': 25, 'step': 7, 'machine': 'Kappa 350 / Kappa 330', 'time': 6.82, 'name': 'Cutting Stripping Jacket Cable 7-Wire', 'workers': 0.5},
            {'item': 25, 'step': 8, 'machine': 'Kappa 350 / Kappa 330', 'time': 8.12, 'name': 'Cutting Stripping Jacket Cable 8-Wire', 'workers': 0.5},
            
            # Item 26 - 4 Wire Jacket 1xDCC Modul 510 BB (180°)
            {'item': 26, 'step': 4, 'machine': 'Kappa 350 / Kappa 330', 'time': 6.54, 'name': 'Cutting Stripping Jacket Cable 4-Wire', 'workers': 0.5},
            
            # Item 27 - 4 Wire Jacket 1xDCC Modul 510 AB (180°)
            {'item': 27, 'step': 5, 'machine': 'Kappa 350 / Kappa 330', 'time': 6.82, 'name': 'Cutting Stripping Jacket Cable 5-Wire', 'workers': 0.5},
            {'item': 27, 'step': 8, 'machine': 'Kappa 350 / Kappa 330', 'time': 8.12, 'name': 'Cutting Stripping Jacket Cable 8-Wire', 'workers': 0.5},
            
            # Item 28 - 4 Wire Jacket 2xDCC Modul T3-512 AA (180°)
            {'item': 28, 'step': 5, 'machine': 'Kappa 350 / Kappa 330', 'time': 7.43, 'name': 'Cutting Stripping Jacket Cable 5-Wire', 'workers': 0.5},
            
            # Item 29 - 4 Wire Jacket 2xDCC Modul T3-513 AA (60°)
            {'item': 29, 'step': 7, 'machine': 'Kappa 350 / Kappa 330', 'time': 7.50, 'name': 'Cutting Stripping Jacket Cable 7-Wire', 'workers': 0.5},
            {'item': 29, 'step': 8, 'machine': 'Kappa 350 / Kappa 330', 'time': 12.00, 'name': 'Cutting Stripping Jacket Cable 8-Wire', 'workers': 0.5},
            
            # Item 30 - 4 Wire Jacket 2xDCC Modul T3-513 AA (180°)
            {'item': 30, 'step': 4, 'machine': 'Kappa 350 / Kappa 330', 'time': 6.34, 'name': 'Cutting Stripping Jacket Cable 4-Wire', 'workers': 0.5},
            {'item': 30, 'step': 8, 'machine': 'Kappa 350 / Kappa 330', 'time': 8.12, 'name': 'Cutting Stripping Jacket Cable 8-Wire', 'workers': 0.5},
            
            # Item 31 - 4 Wire Jacket 2xDCC Modul T3-511 AA (60° & 90°)
            {'item': 31, 'step': 4, 'machine': 'Kappa 350 / Kappa 330', 'time': 6.45, 'name': 'Cutting Stripping Jacket Cable 4-Wire', 'workers': 0.5},
            {'item': 31, 'step': 8, 'machine': 'Kappa 350 / Kappa 330', 'time': 8.12, 'name': 'Cutting Stripping Jacket Cable 8-Wire', 'workers': 0.5},
            
            # Item 32 - 4 Wire Jacket 2xDCC Modul T3-x11 AA (180°)
            {'item': 32, 'step': 4, 'machine': 'Kappa 350 / Kappa 330', 'time': 6.45, 'name': 'Cutting Stripping Jacket Cable 4-Wire', 'workers': 0.5},
            {'item': 32, 'step': 8, 'machine': 'Kappa 350 / Kappa 330', 'time': 8.12, 'name': 'Cutting Stripping Jacket Cable 8-Wire', 'workers': 0.5},
            
            # Item 33 - 4 Wire Jacket 2xDCC Modul T3-x11 AA (60° & 90°)
            {'item': 33, 'step': 4, 'machine': 'Kappa 350 / Kappa 330', 'time': 6.30, 'name': 'Cutting Stripping Jacket Cable 4-Wire', 'workers': 0.5},
            {'item': 33, 'step': 8, 'machine': 'Kappa 350 / Kappa 330', 'time': 8.12, 'name': 'Cutting Stripping Jacket Cable 8-Wire', 'workers': 0.5},
            
            # Item 34 - Twisted Wires 1xDCC Modul T3-x13 AA (60° & 90°)
            {'item': 34, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 6.61, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 35 - Twisted Wires 1xDCC Modul T1-513 AA (60°)
            {'item': 35, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 6.61, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 36 - Single Wires 1xDCC Modul T3-x12 AA (180°)
            {'item': 36, 'step': 2, 'machine': 'Alpha 550 / Alpha 433', 'time': 3.30, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5},
            
            # Item 37 - 4 Wire Jacket 1xDCC Modul 963004 _A (60°)
            {'item': 37, 'step': 4, 'machine': 'Kappa 350 / Kappa 330', 'time': 6.54, 'name': 'Cutting Stripping Jacket Cable 4-Wire', 'workers': 0.5},
            {'item': 37, 'step': 8, 'machine': 'Kappa 350 / Kappa 330', 'time': 8.12, 'name': 'Cutting Stripping Jacket Cable 8-Wire', 'workers': 0.5},
            
            # Item 38 - 4 Wire Jacket 1xDCC Modul 963004 _B (60°)
            {'item': 38, 'step': 7, 'machine': 'Kappa 350 / Kappa 330', 'time': 6.53, 'name': 'Cutting Stripping Jacket Cable 7-Wire', 'workers': 0.5},
            {'item': 38, 'step': 9, 'machine': 'Kappa 350 / Kappa 330', 'time': 16.00, 'name': 'Cutting Stripping Jacket Cable 9-Wire', 'workers': 0.5},
            
            # Item 39 - 4 Wire Jacket 1xDCC Modul 963002 (60°)
            {'item': 39, 'step': 4, 'machine': 'Kappa 350 / Kappa 330', 'time': 6.68, 'name': 'Cutting Stripping Jacket Cable 4-Wire', 'workers': 0.5},
            {'item': 39, 'step': 8, 'machine': 'Kappa 350 / Kappa 330', 'time': 8.12, 'name': 'Cutting Stripping Jacket Cable 8-Wire', 'workers': 0.5},
            
            # Item 40 - 4 Wire Jacket 1xDCC Modul 963001 (60°)
            {'item': 40, 'step': 4, 'machine': 'Kappa 350 / Kappa 330', 'time': 6.33, 'name': 'Cutting Stripping Jacket Cable 4-Wire', 'workers': 0.5},
            {'item': 40, 'step': 8, 'machine': 'Kappa 350 / Kappa 330', 'time': 8.12, 'name': 'Cutting Stripping Jacket Cable 8-Wire', 'workers': 0.5},
            
            # Item 41 - Twisted Wires 1xDCC Modul 9J0, 9J1 HL (180°)
            {'item': 41, 'step': 1, 'machine': 'Sigma 688 / Alpha 488', 'time': 6.90, 'name': 'Cutting Stripping Crimping Twisting Single Wires', 'workers': 0.5}
        ]
        
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
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)