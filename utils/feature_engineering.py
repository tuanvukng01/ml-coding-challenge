import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def compute_geographical_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = (sin(d_lat / 2)**2
         + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2)**2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["DISTANCE_KM"] = df.apply(
        lambda row: compute_geographical_distance(
            row["USER_LAT"],
            row["USER_LONG"],
            row["VENUE_LAT"],
            row["VENUE_LONG"]),
        axis=1
    )
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df["HOUR_OF_DAY"] = df["TIMESTAMP"].dt.hour
    df["DAY_OF_WEEK"] = df["TIMESTAMP"].dt.dayofweek
    df["MONTH"] = df["TIMESTAMP"].dt.month
    df["RAIN_INDICATOR"] = (df["PRECIPITATION"] > 0).astype(int)

    for col in ["TEMPERATURE", "WIND_SPEED"]:
        min_val, max_val = df[col].min(), df[col].max()
        df[col + "_SCALED"] = (df[col] - min_val) / (max_val - min_val + 1e-9)
    return df