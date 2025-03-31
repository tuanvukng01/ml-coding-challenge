import pandas as pd
import numpy as np


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["ESTIMATED_DELIVERY_MINUTES",
                           "ACTUAL_DELIVERY_MINUTES",
                           "ITEM_COUNT",
                           "USER_LAT",
                           "USER_LONG",
                           "VENUE_LAT",
                           "VENUE_LONG"])
    df = df.fillna(0)
    df = df[(df["ESTIMATED_DELIVERY_MINUTES"] > 0) &
            (df["ESTIMATED_DELIVERY_MINUTES"] < 500)]
    df = df[(df["ACTUAL_DELIVERY_MINUTES"] > 0) &
            (df["ACTUAL_DELIVERY_MINUTES"] < 500)]
    return df

def train_test_split(df: pd.DataFrame,
                     test_ratio: float = 0.2,
                     random_state: int = 42):
    df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_size = int(len(df) * test_ratio)
    test_df = df_shuffled.iloc[:test_size]
    train_df = df_shuffled.iloc[test_size:]
    return train_df, test_df