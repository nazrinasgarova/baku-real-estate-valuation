"""
Baku Real Estate Feature Engineering & Spatial Analytics Module
Provides geodesic distance calculations, structural ratio constructions,
and structured amenity signal extraction from listing text.
"""

import numpy as np
import pandas as pd


def haversine_distance(lat, lng, center_lat=40.3725, center_lng=49.8360):
    """
    Calculate geodesic distance in kilometers between coordinates and Baku City Center
    (Fountain Square / 28 May hub: 40.3725° N, 49.8360° E) using the spherical Haversine formula.

    Parameters:
        lat (float or pd.Series): Latitude coordinate(s) in decimal degrees.
        lng (float or pd.Series): Longitude coordinate(s) in decimal degrees.
        center_lat (float): Reference center latitude (default: 40.3725° N).
        center_lng (float): Reference center longitude (default: 49.8360° E).

    Returns:
        float or pd.Series: Distance in kilometers to the designated city center.
    """
    lat_r, lng_r = np.radians(lat), np.radians(lng)
    c_lat_r, c_lng_r = np.radians(center_lat), np.radians(center_lng)
    dlat = c_lat_r - lat_r
    dlng = c_lng_r - lng_r
    a = np.sin(dlat / 2.0)**2 + np.cos(lat_r) * np.cos(c_lat_r) * np.sin(dlng / 2.0)**2
    return 6371.0 * 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


def extract_amenities(text_series: pd.Series) -> pd.DataFrame:
    """
    Extract structured amenity signals from free-text listing descriptions.
    Uses strict word boundaries to avoid false positives (e.g., '100 kv.m.' matching 'm.').

    Parameters:
        text_series (pd.Series): Raw string series containing property descriptions.

    Returns:
        pd.DataFrame: Binary indicator columns: 'has_metro', 'has_sea_view', and 'has_gas_lift'.
    """
    text_clean = text_series.fillna('').astype(str).str.lower()
    has_metro = text_clean.str.contains(r'\b(?:metro|metronun|metrosu|metrostansiya|m/s)\b', regex=True).astype(int)
    has_sea_view = text_clean.str.contains(r'\b(?:dəniz|deniz|panoram|kaspi)\b', regex=True).astype(int)
    has_gas_lift = text_clean.str.contains(r'qaz.*lift|lift.*qaz', regex=True).astype(int)
    return pd.DataFrame({
        'has_metro': has_metro,
        'has_sea_view': has_sea_view,
        'has_gas_lift': has_gas_lift
    }, index=text_series.index)
