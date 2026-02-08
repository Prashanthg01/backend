"""
Celery background tasks for production scheduling.

Usage:
    # From Django view
    task = generate_schedule_task.delay(params)
    return Response({'task_id': task.id, 'status': 'processing'})
    
    # Check status later
    result = AsyncResult(task_id)
    if result.ready():
        data = result.get()
"""

from celery import shared_task, current_task
from celery.result import AsyncResult
import pandas as pd
import numpy as np
from django.db.models import Sum
from datetime import datetime

from .models import Product, Machine, ProcessStep, ProductionSchedule
from .utils import (
    optimize_product_batches_jointly,
    generate_production_schedule,
    calculate_kpis,
    process_frontpage_data,
    process_routing_data,
)


@shared_task(bind=True, name='generate_schedule_task')
def generate_schedule_task(self, params):
    """
    Background task for schedule generation.
    
    Parameters
    ----------
    params : dict
        {
            'max_num_batches': int,
            'min_batch_size': int,
            'max_batch_size': int,
            'use_pulp_scheduler': bool or None,
            'time_limit': int
        }
    
    Returns
    -------
    dict : {
        'status': 'success' | 'error',
        'message': str,
        'kpis': dict,
        'schedule_count': int,
        'batch_optimization': dict,
        'performance': dict
    }
    """
    try:
        # Update progress: 10%
        self.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Clearing old schedules'})
        
        ProductionSchedule.objects.all().delete()
        
        products = Product.objects.filter(demand_2024__gt=0).order_by('item')
        if not products.exists():
            return {
                'status': 'error',
                'message': 'No products with demand found'
            }
        
        # Update progress: 30%
        self.update_state(state='PROGRESS', meta={'progress': 30, 'status': 'Optimizing batch sizes'})
        
        batch_log = optimize_product_batches_jointly(
            products,
            params['max_num_batches'],
            params['min_batch_size'],
            params['max_batch_size']
        )
        
        # Update progress: 60%
        self.update_state(state='PROGRESS', meta={'progress': 60, 'status': 'Generating production schedule'})
        
        schedules, machine_availability = generate_production_schedule(
            products,
            use_pulp=params.get('use_pulp_scheduler'),
            time_limit_seconds=params.get('time_limit', 60)
        )
        
        # Update progress: 90%
        self.update_state(state='PROGRESS', meta={'progress': 90, 'status': 'Calculating KPIs'})
        
        kpis = calculate_kpis(schedules, machine_availability)
        
        num_operations = len(schedules)
        
        # Determine method used
        if params.get('use_pulp_scheduler') is True:
            method_used = "PuLP job-shop (forced)"
        elif params.get('use_pulp_scheduler') is False:
            method_used = "Improved greedy (forced)"
        else:
            method_used = "PuLP job-shop (auto)" if num_operations <= 100 else "Improved greedy (auto)"
        
        # Update progress: 100%
        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Complete'})
        
        return {
            'status': 'success',
            'message': f'Schedule generated successfully with {len(schedules)} operations',
            'scheduling_method': method_used,
            'kpis': kpis,
            'schedule_count': len(schedules),
            'batch_optimization': {
                'parameters': {
                    'max_num_batches': params['max_num_batches'],
                    'min_batch_size': params['min_batch_size'],
                    'max_batch_size': params['max_batch_size'],
                },
                'products_optimized': len(batch_log),
                'sample_optimizations': batch_log[:5]
            },
            'performance': {
                'total_operations': num_operations,
                'method_used': method_used,
                'time_limit_seconds': params.get('time_limit')
            }
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(error_trace)
        
        return {
            'status': 'error',
            'message': str(e),
            'traceback': error_trace
        }


@shared_task(bind=True, name='initialize_data_task')
def initialize_data_task(self, frontpage_csv_path, process_csv_path):
    """
    Background task for database initialization from CSV files.
    
    Parameters
    ----------
    frontpage_csv_path : str
        Temporary file path for Frontpage.csv
    process_csv_path : str
        Temporary file path for Process.csv
    
    Returns
    -------
    dict : {
        'status': 'success' | 'error',
        'message': str,
        'products_created': int,
        'machines_created': int,
        'process_steps_created': int
    }
    """
    try:
        # Update progress: 10%
        self.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Reading CSV files'})
        
        frontpage_df = pd.read_csv(frontpage_csv_path)
        process_df = pd.read_csv(process_csv_path)
        process_df = process_df.iloc[:, :-2]
        
        # Update progress: 30%
        self.update_state(state='PROGRESS', meta={'progress': 30, 'status': 'Processing data'})
        
        demand_data = process_frontpage_data(frontpage_df)
        process_routing, machines_list = process_routing_data(process_df)
        
        # Update progress: 40%
        self.update_state(state='PROGRESS', meta={'progress': 40, 'status': 'Clearing existing data'})
        
        Product.objects.all().delete()
        Machine.objects.all().delete()
        ProcessStep.objects.all().delete()
        ProductionSchedule.objects.all().delete()
        
        # Update progress: 50%
        self.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Creating machines'})
        
        machines = {}
        for machine_name in machines_list:
            if machine_name:
                machines[machine_name] = Machine.objects.create(
                    name=machine_name,
                    available_hours_per_day=24
                )
        
        # Update progress: 60%
        self.update_state(state='PROGRESS', meta={'progress': 60, 'status': 'Creating products'})
        
        total_items = len(demand_data['Item'])
        for i in range(total_items):
            # Update progress incrementally
            if i % 10 == 0:
                progress = 60 + int((i / total_items) * 30)
                self.update_state(state='PROGRESS', meta={
                    'progress': progress,
                    'status': f'Creating products ({i}/{total_items})'
                })
            
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
        
        # Update progress: 100%
        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Complete'})
        
        return {
            'status': 'success',
            'message': 'Database initialized successfully',
            'products_created': Product.objects.count(),
            'machines_created': Machine.objects.count(),
            'process_steps_created': ProcessStep.objects.count()
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(error_trace)
        
        return {
            'status': 'error',
            'message': str(e),
            'traceback': error_trace
        }


@shared_task(bind=True, name='batch_optimize_preview_task')
def batch_optimize_preview_task(self, params):
    """
    Background task for batch optimization preview.
    
    Parameters
    ----------
    params : dict
        {
            'max_num_batches': int,
            'min_batch_size': int,
            'max_batch_size': int
        }
    
    Returns
    -------
    dict : Batch analysis results
    """
    try:
        from .utils import calculate_optimal_batch_size
        
        self.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Loading products'})
        
        products = Product.objects.filter(demand_2024__gt=0)
        
        batch_analysis = []
        total_demand = 0
        total_batches = 0
        batch_sizes = []
        
        total_products = products.count()
        
        for idx, product in enumerate(products):
            if idx % 10 == 0:
                progress = 10 + int((idx / total_products) * 80)
                self.update_state(state='PROGRESS', meta={
                    'progress': progress,
                    'status': f'Analyzing products ({idx}/{total_products})'
                })
            
            demand = product.demand_2024
            
            batch_size, num_batches, ideal_batch = calculate_optimal_batch_size(
                demand,
                params['max_num_batches'],
                params['min_batch_size'],
                params['max_batch_size']
            )
            
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
                'improvement': (
                    f"{((old_num_batches - num_batches) / old_num_batches * 100):.1f}%"
                    if old_num_batches != num_batches else "0%"
                ),
            })
            
            total_demand += demand
            total_batches += num_batches
            batch_sizes.append(batch_size)
        
        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Complete'})
        
        return {
            'status': 'success',
            'batch_analysis': batch_analysis,
            'summary': {
                'total_products': len(batch_analysis),
                'total_demand': total_demand,
                'total_batches': total_batches,
                'avg_batch_size': round(np.mean(batch_sizes), 2) if batch_sizes else 0,
                'min_batch_size': int(np.min(batch_sizes)) if batch_sizes else 0,
                'max_batch_size': int(np.max(batch_sizes)) if batch_sizes else 0,
                'std_batch_size': round(np.std(batch_sizes), 2) if batch_sizes else 0,
            },
            'parameters': params
        }
        
    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }