import pandas as pd
import numpy as np

from fit_etl.utils import fields2row
from fit_etl.constants import *


class FitDataFrame:
    def __init__(self) -> None:
        pass

    @staticmethod
    def extract_laps(df: pd.DataFrame):
        # expand the dataframe to use it well
        df_laps = pd.json_normalize(
            df[df.name == "lap"].fields.dropna().apply(fields2row)
        )
        # attribute a category to each lap
        df_laps["category"] = df_laps.apply(
            lambda s: FitDataFrame._categorize_lap_serie(s, len(df_laps)), axis=1
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
            ]
        )
        return df_laps

    @staticmethod
    def _categorize_lap_serie(lap: pd.Series, last_lap_id: int):
        if lap.name == 0:
            return LAP_CATEGORY_WARMUP
        elif lap.name == last_lap_id - 1:
            return LAP_CATEGORY_WARMDOWN
        elif np.isclose(int(lap.total_distance), lap.total_distance, atol=1e-3):
            return LAP_CATEGORY_RUN
        elif np.isclose(int(lap.total_timer_time), lap.total_timer_time, atol=1e-1):
            return LAP_CATEGORY_BREAK
        else:
            return "unknown"

    @staticmethod
    def extract_session(df: pd.DataFrame):
        return pd.json_normalize(
            df[df.name == "session"].fields.dropna().apply(fields2row)
        )

    @staticmethod
    def extract_date(df: pd.DataFrame):
        session = FitDataFrame.extract_session(df)
        try:
            return session.start_time[0]
        except KeyError:
            return None

    @staticmethod
    def extract_records(df: pd.DataFrame):
        df_records = pd.json_normalize(
            df[df.name == "record"].fields.dropna().apply(fields2row)
        )
        df_records = df_records.set_index(pd.to_datetime(df_records["timestamp"]))
        return df_records
