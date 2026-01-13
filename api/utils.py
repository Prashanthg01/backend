import pandas as pd
import numpy as np

SHIFT_LABELS = [
    'Shift 1', 'Shift 1 B', 'Shift 2', 'Shift 2 A', 'Shift 3', 'Shift 3 C',
    'Shift 4', 'Shift 4 B', 'Shift 5', 'Shift 5 A', 'Shift 6', 'Shift 6 C',
    'Shift 7', 'Shift 7 B', 'Shift 8', 'Shift 8 A', 'Shift 9', 'Shift 9 C',
    'Shift 10', 'Shift 10 B', 'Shift 11', 'Shift 11 A', 'Shift 12', 'Shift 12 C',
    'Shift 13', 'Shift 13 B', 'Shift 14', 'Shift 14 A', 'Shift 15', 'Shift 15 C',
    'Shift 16', 'Shift 16 B', 'Shift 17', 'Shift 17 A', 'Shift 18', 'Shift 18 C'
]

def build_summary(df):
    """Build final summary metrics table."""
    return {
        "Metric": ["Target Headcount", "Actual Headcount", "Current Backlog", "Currently Open"],
        "Finished Goods": [
            f"{df[df['Step']=='F']['Planned'].sum():,.0f}",
            f"{df[df['Step']=='F']['Realized'].sum():,.0f}",
            f"{df[(df['Step']=='F') & (df['Backlog']>0)]['Backlog'].sum():,.0f}",
            f"{df[df['Step']=='F']['Open'].sum():,.0f}",
        ],
        "Connectors": [
            f"{df[df['Area']=='Assembly']['Planned'].sum():,.0f}",
            f"{df[df['Area']=='Assembly']['Realized'].sum():,.0f}",
            f"{df[(df['Area']=='Assembly') & (df['Backlog']>0)]['Backlog'].sum():,.0f}",
            f"{df[df['Area']=='Assembly']['Open'].sum():,.0f}",
        ],
    }


def calculate_production_outputs(df, finished_filter, connector_filter):
    """Calculate shift-wise production outputs."""
    fg_output = {}
    conn_output = {}

    for i, shift in enumerate(SHIFT_LABELS):
        col_idx = 14 + i
        if col_idx < 50:
            fg = df.iloc[:, col_idx][finished_filter].sum()
            conn = df.iloc[:, col_idx][connector_filter].sum()
        else:
            fg = conn = 0

        fg_output[shift] = f"{fg:,.0f}" if fg > 0 else "0"
        conn_output[shift] = f"{conn:,.0f}" if conn > 0 else "0"

    return fg_output, conn_output

def calculate_backlog(df):
    """Calculate backlog per shift."""
    backlog_cols = range(95, 113)
    backlog_values = []

    for idx in backlog_cols:
        total = df.iloc[2:264, idx]
        total = total.loc[total > 0].sum()
        backlog_values.append(total if total > 0 else 0)

    result = []
    for val in backlog_values:
        result.extend([val, 0])

    return (result + [0] * len(SHIFT_LABELS))[:len(SHIFT_LABELS)]

def calculate_efficiency(df, num_shifts):
    """
    Calculate overall efficiency per shift.

    Efficiency = (Planned Production Time / Available Time) * 100
    """
    efficiency_list = []

    if 'STD' not in df.columns:
        return [0] * 36

    std_col = pd.to_numeric(
        df.loc[2:1003, 'STD'].astype(str)
        .str.replace(r'[^\d\.\-]', '', regex=True),
        errors='coerce'
    )

    shift_time_hours = 7.67
    available_time = shift_time_hours * num_shifts
    quantity_columns = list(range(15, 52, 2))

    for col_idx in quantity_columns:
        quantity = pd.to_numeric(
            df.iloc[2:1003, col_idx].astype(str)
            .str.replace(r'[^\d\.\-]', '', regex=True),
            errors='coerce'
        )

        valid = quantity.notna() & std_col.notna()
        planned_minutes = (quantity[valid] * std_col[valid]).sum()
        planned_hours = planned_minutes / 60

        efficiency = (planned_hours / available_time) * 100 if available_time else 0
        efficiency_list.append(efficiency)

    # Insert zero after each value to match 36 shifts
    result = []
    for val in efficiency_list:
        result.extend([val, 0])

    return (result + [0] * 36)[:36]

def apply_filters(df, filters):
    """Apply optional column filters to the dataframe."""
    for column, value in filters.items():
        if value != "All" and column in df.columns:
            df = df[df[column] == value]
    return df

def clean_numeric_columns(df, columns):
    """Convert specified columns to numeric by removing commas and invalid characters."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '').str.strip(),
                errors='coerce'
            )


def clean_text_columns(df, columns):
    """Strip whitespace from specified text columns."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()


def clean_shift_columns(df, column_ranges):
    """Clean numeric shift-based columns using column index ranges."""
    for start, end in column_ranges:
        for idx in range(start, end):
            df.iloc[:, idx] = pd.to_numeric(
                df.iloc[:, idx]
                .astype(str)
                .str.replace(r'[^\d\.\-]', '', regex=True),
                errors='coerce'
            )


############################ End of process_csv ########################


def process_frontpage_data(frontpage_df):
    """
    Process frontpage CSV data
    """
    # Rename columns
    df = frontpage_df.rename(columns={
        'SAP TN': 'SAP_TN',
        'SAP PL': 'SAP_PL',
        'DCC Type': 'DCC_Type'
    })
    
    # Select required columns
    df = df[['Item', 'SAP_TN', 'SAP_PL', 'DCC_Type', 'Description', '2024']]
    
    # Rename year column
    df = df.rename(columns={'2024': 'Demand_2024'})
    
    # Columns to convert to integer
    int_cols = ['Item', 'SAP_TN', 'SAP_PL', 'Demand_2024']
    
    # Clean commas and convert to integer
    for col in int_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '', regex=False)
            .replace('None', pd.NA)
            .replace('nan', pd.NA)
            .pipe(pd.to_numeric, errors='coerce')
            .astype('Int64')
        )
    
    # Take first 5 rows (or all if fewer than 5)
    df = df.head(5)
    
    # Replace NaN with None
    df = df.where(pd.notna(df), None)
    
    # Convert to dictionary
    demand_data = df.to_dict(orient='list')
    
    return demand_data


def process_routing_data(process_df):
    """
    Process routing CSV data
    """
    process_routing = []
    
    # Extract machines from row 3 (index 2)
    machines = (
        process_df.iloc[2, 4:]
        .fillna('')
        .astype(str)
        .str.strip()
        .tolist()
    )
    
    # Extract process steps from row 4 (index 3)
    process_steps = (
        process_df.iloc[3, 4:]
        .fillna('')
        .astype(str)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .tolist()
    )
    
    # Get data rows (from row 5 onwards)
    data_df = process_df.iloc[4:].copy()
    
    for _, row in data_df.iterrows():
        # Skip rows without ITEM number
        if not str(row.iloc[0]).replace('.0', '').isdigit():
            continue
        
        try:
            item = int(float(row.iloc[0]))
        except (ValueError, TypeError):
            continue
        
        for idx in range(len(process_steps)):
            raw_val = row.iloc[idx + 4]
            
            # Convert to number safely
            time_val = pd.to_numeric(raw_val, errors='coerce')
            
            if pd.notna(time_val) and time_val > 0:
                process_routing.append({
                    'item': item,
                    'step': idx + 1,
                    'machine': machines[idx],
                    'time': round(float(time_val), 2),
                    'name': process_steps[idx],
                    'workers': 0.5
                })
    
    # Get unique machines
    machines_list = list(set(m for m in machines if m))
    
    return process_routing, machines_list

def get_batch_params(request):
    """
    Extract batch optimization parameters from the API request.

    Returns:
        tuple: (max_num_batches, min_batch_size, max_batch_size)
    """
    return (
        int(request.data.get('max_num_batches', 25)),
        int(request.data.get('min_batch_size', 50)),
        int(request.data.get('max_batch_size', 500)),
    )

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

def optimize_product_batches(products, max_num_batches, min_batch_size, max_batch_size):
    """
    Calculate and persist optimal batch size and batch count for each product.

    Updates:
        - product.batch_size
        - product.num_batches

    Returns:
        list[dict]: Log of batch optimization details per product.
    """
    batch_optimization_log = []

    for product in products:
        batch_size, num_batches, ideal_batch = calculate_optimal_batch_size(
            product.demand_2024,
            max_num_batches,
            min_batch_size,
            max_batch_size
        )

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

    return batch_optimization_log

def generate_production_schedule(products):
    """
    Generate a production schedule for all products and their batches.

    Scheduling rules:
        - A machine can process only one operation at a time
        - A batch step can start only after the previous step completes

    Returns:
        tuple:
            - list[ProductionSchedule]: Created schedule records
            - dict: Machine availability timestamps
    """
    machine_availability = {}
    batch_completion = {}
    schedule_records = []
    start_date = datetime.now()

    for product in products:
        process_steps = ProcessStep.objects.filter(
            product=product,
            cycle_time_seconds__gt=0
        ).order_by('step_number')

        for batch_num in range(1, product.num_batches + 1):
            batch_id = f"Item{product.item}_B{batch_num}"

            for step in process_steps:
                machine = step.machine
                total_hours = (step.cycle_time_seconds * product.batch_size) / 3600

                machine_key = machine.name
                machine_ready = machine_availability.get(machine_key, start_date)

                prev_key = f"{batch_id}_Step{step.step_number - 1}"
                prev_done = batch_completion.get(prev_key, start_date)

                start_time = max(machine_ready, prev_done)
                end_time = start_time + timedelta(hours=total_hours)

                machine_availability[machine_key] = end_time
                batch_completion[f"{batch_id}_Step{step.step_number}"] = end_time

                schedule_records.append(
                    ProductionSchedule.objects.create(
                        machine=machine,
                        product=product,
                        process_step=step,
                        batch_id=batch_id,
                        batch_num=batch_num,
                        batch_size=product.batch_size,
                        start_time=start_time,
                        end_time=end_time,
                        duration_hours=round(total_hours, 4)
                    )
                )

    return schedule_records, machine_availability

def calculate_kpis(schedule_records, machine_availability):
    """
    Calculate high-level scheduling KPIs.

    KPIs include:
        - Makespan (hours & days)
        - Machine utilization
        - Throughput (units/day)
        - Total scheduled operations

    Returns:
        dict: KPI metrics
    """
    if not schedule_records:
        return {}

    max_end = max(s.end_time for s in schedule_records)
    min_start = min(s.start_time for s in schedule_records)
    makespan_hours = (max_end - min_start).total_seconds() / 3600
    makespan_days = makespan_hours / 24

    machine_stats = {}
    for machine_name in machine_availability:
        used_hours = ProductionSchedule.objects.filter(
            machine__name=machine_name
        ).aggregate(total=Sum('duration_hours'))['total'] or 0

        utilization = (used_hours / makespan_hours * 100) if makespan_hours > 0 else 0
        machine_stats[machine_name] = {
            'used_hours': round(used_hours, 2),
            'utilization': round(utilization, 2)
        }

    total_units = Product.objects.filter(demand_2024__gt=0).aggregate(
        total=Sum('demand_2024')
    )['total'] or 0

    return {
        'total_makespan_hours': round(makespan_hours, 2),
        'total_makespan_days': round(makespan_days, 2),
        'machine_utilization': machine_stats,
        'total_operations': len(schedule_records),
        'throughput_units_per_day': round(
            total_units / makespan_days if makespan_days > 0 else 0, 2
        ),
        'total_units_scheduled': total_units
    }
