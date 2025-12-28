import pandas as pd

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