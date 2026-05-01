import numpy as np
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 10 static calendar features (Occam's Razor)."""
    df = df.copy()
    
    # 1. Base Calendar
    df['year']       = df['Date'].dt.year
    df['month']      = df['Date'].dt.month
    df['day']        = df['Date'].dt.day
    df['dayofweek']  = df['Date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['dayofyear']  = df['Date'].dt.dayofyear
    df['quarter']    = df['Date'].dt.quarter

    # 2. Insight 1 - Payday Effect
    df['is_payday']  = df['day'].isin([1, 2, 3, 28, 29, 30, 31]).astype(int)

    # 3. Insight 2 - Cyclic Encoding (Month)
    df['month_sin']  = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos']  = np.cos(2 * np.pi * df['month'] / 12.0)

    return df

def get_feature_columns(df: pd.DataFrame = None) -> list[str]:
    """Return feature column names."""
    return [
        'year', 'month', 'day', 'dayofweek', 'is_weekend',
        'dayofyear', 'quarter', 'is_payday', 'month_sin', 'month_cos'
    ]
