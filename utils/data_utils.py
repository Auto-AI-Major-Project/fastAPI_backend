# data_utils.py
import pandas as pd
from io import StringIO, BytesIO
from typing import List, Dict, Any, Tuple
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

# Global variable to store the dataframe in memory
global_data_store: Dict[str, Any] = {}

def load_data(file_content: bytes, filename: str) -> pd.DataFrame:
    """Loads and stores data from uploaded file content (CSV or Excel)."""
    if filename.endswith('.csv'):
        df = pd.read_csv(StringIO(file_content.decode('utf-8')))
    elif filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(BytesIO(file_content))
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
    
    global_data_store[filename] = df
    return df

def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Automatically detects and classifies columns into Numeric, Categorical, and Datetime."""
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []

    for col in df.columns:
        if is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        else:
            # Simple heuristic for categorical: low cardinality non-numeric
            if df[col].nunique() / len(df) < 0.2 and df[col].nunique() > 1:
                categorical_cols.append(col)
            elif df[col].nunique() / len(df) < 0.9: 
                 categorical_cols.append(col)
    
    return numeric_cols, categorical_cols, datetime_cols


def get_dataset_inspection(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
    """Generates a basic summary of the dataset for cleaning purposes."""
    numeric_cols, categorical_cols, datetime_cols = get_column_types(df)
    
    missing_values = df.isnull().sum().to_dict()
    # Provide text summary for Gemini
    summary_stats = df.describe(include='all').to_markdown(index=True)

    return {
        "filename": filename,
        "shape": list(df.shape),
        "missing_values": {k: int(v) for k, v in missing_values.items() if v > 0},
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "preview_first_5_rows": df.head(5).to_html(),
        "summary_statistics_markdown": summary_stats,
        "message": "Dataset loaded and inspected. Ready for cleaning and reporting."
    }