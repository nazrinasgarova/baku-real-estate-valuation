"""
Baku Real Estate Data Preprocessing Module
Contains parsing utilities for cleaning raw text-based columns into structured numerical values.
"""

import pandas as pd


def clean_area(series: pd.Series) -> pd.Series:
    """
    Extract numeric square meter values from raw string format (e.g., '145 m²' -> 145.0).

    Parameters:
        series (pd.Series): Raw string series containing area representations.

    Returns:
        pd.Series: Parsed floating-point area values.
    """
    return series.astype(str).str.extract(r'([\d\.]+)')[0].astype(float)


def parse_floors(series: pd.Series) -> pd.DataFrame:
    """
    Parse current floor and total building floors from string representation (e.g., '7 / 9' -> 7, 9).

    Parameters:
        series (pd.Series): Raw string series containing floor representations.

    Returns:
        pd.DataFrame: DataFrame with 'current_floor' and 'total_floors' as floats.
    """
    extracted = series.astype(str).str.extract(r'(\d+)\s*/\s*(\d+)')
    current_floor = extracted[0].astype(float)
    total_floors = extracted[1].astype(float)
    return pd.DataFrame({'current_floor': current_floor, 'total_floors': total_floors}, index=series.index)
