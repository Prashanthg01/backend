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