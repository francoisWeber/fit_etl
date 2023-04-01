import pandas as pd
import numpy as np
from fit_etl import utils


def fit2lap(df: pd.DataFrame):
    df_laps = pd.json_normalize(
        df[df.name == "lap"].fields.dropna().apply(utils.fields2row)
    )
    # attribute a category to each lap
    df_laps["category"] = df_laps.apply(
        lambda s: utils.categorize_lap(s, len(df_laps)), axis=1
    )
    # guess end time of each lap (so we can slice according to datetimes later)
    df_laps["start_time"] = pd.to_datetime(df_laps.start_time)
    df_laps["stop_time"] = df_laps.start_time + pd.to_timedelta(
        df_laps.total_elapsed_time, unit="sec"
    )
    # drop useless
    df_laps.drop(
        columns=[
            "timestamp",
            "avg_stance_time",
            "avg_stance_time_percent",
            "avg_vertical_oscillation",
            "avg_vertical_ratio",
        ],
        inplace=True,
    )
    return df_laps


def fit2records(df: pd.DataFrame)
