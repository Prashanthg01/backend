from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import numpy as np
from django.db.models import Sum, Count, Q, Max, Min, Avg
from .models import Product, Machine, ProcessStep, ProductionSchedule
from .serializers import ProductionScheduleSerializer
from .utils import clean_numeric_columns, clean_text_columns, apply_filters, clean_shift_columns, calculate_efficiency, calculate_backlog, calculate_production_outputs, SHIFT_LABELS, build_summary
from .utils import get_batch_params, optimize_product_batches, generate_production_schedule, calculate_kpis, process_frontpage_data, process_routing_data

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


@api_view(['POST'])
def generate_schedule(request):
    """
    API endpoint to generate an optimized production schedule.

    Workflow:
        1. Read batch optimization parameters
        2. Optimize batch sizes per product
        3. Generate machine-level production schedule
        4. Compute scheduling KPIs
    """
    try:
        max_num_batches, min_batch_size, max_batch_size = get_batch_params(request)

        ProductionSchedule.objects.all().delete()

        products = Product.objects.filter(demand_2024__gt=0).order_by('item')
        if not products.exists():
            return Response(
                {'error': 'No products with demand found'},
                status=status.HTTP_400_BAD_REQUEST
            )

        batch_log = optimize_product_batches(
            products, max_num_batches, min_batch_size, max_batch_size
        )

        schedules, machine_availability = generate_production_schedule(products)
        kpis = calculate_kpis(schedules, machine_availability)

        return Response({
            'message': f'Schedule generated successfully with {len(schedules)} operations',
            'kpis': kpis,
            'schedule_count': len(schedules),
            'batch_optimization': {
                'parameters': {
                    'max_num_batches': max_num_batches,
                    'min_batch_size': min_batch_size,
                    'max_batch_size': max_batch_size
                },
                'products_optimized': len(batch_log),
                'sample_optimizations': batch_log[:5]
            }
        }, status=status.HTTP_201_CREATED)

    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

   
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
    Initialize database with uploaded CSV files
    """
    try:
        # Check if files are present
        if 'frontpage' not in request.FILES or 'process' not in request.FILES:
            return Response({
                'error': 'Both Frontpage.csv and Process.csv files are required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        frontpage_file = request.FILES['frontpage']
        process_file = request.FILES['process']
        
        # Read CSV files
        try:
            frontpage_df = pd.read_csv(frontpage_file)
            process_df = pd.read_csv(process_file)
            process_df = process_df.iloc[:, :-2]  # Remove last 2 columns
        except Exception as e:
            return Response({
                'error': f'Error reading CSV files: {str(e)}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Process frontpage data
        demand_data = process_frontpage_data(frontpage_df)
        
        # Process routing data
        process_routing, machines_list = process_routing_data(process_df)
        
        # Clear existing data
        Product.objects.all().delete()
        Machine.objects.all().delete()
        ProcessStep.objects.all().delete()
        ProductionSchedule.objects.all().delete()
        
        # Create machines
        machines = {}
        for machine_name in machines_list:
            if machine_name:  # Skip empty machine names
                machine = Machine.objects.create(
                    name=machine_name,
                    available_hours_per_day=24
                )
                machines[machine_name] = machine
        
        # Create products and process steps
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
            
            # Add process steps for this product
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
        import traceback
        print(traceback.format_exc())
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