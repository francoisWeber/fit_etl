from typing import List
from fit_etl.fidata import FitDataFrame
from fit_etl.constants import *
from fit_etl.fitfile import decode_fitfile

import pandas as pd
from matplotlib import pyplot as plt


class Workout:
    def __init__(
        self,
        path: str,
        date: str,
        laps: pd.DataFrame,
        records: pd.DataFrame,
        session=pd.DataFrame,
    ) -> None:
        self.path = path
        self.date = date
        self.laps = laps
        self.session = session
        self.records = records

    @classmethod
    def from_fitjson(clf, path_or_buf):
        df = pd.read_json(path_or_buf=path_or_buf)
        path = path_or_buf
        date = FitDataFrame.extract_date(df)
        laps = FitDataFrame.extract_laps(df)
        session = FitDataFrame.extract_session(df)
        records = FitDataFrame.extract_records(df)
        return clf(path=path, date=date, laps=laps, session=session, records=records)

    @classmethod
    def from_fit(clf, path_or_buf):
        df = pd.DataFrame(decode_fitfile(path_or_buf))
        path = path_or_buf
        date = FitDataFrame.extract_date(df)
        laps = FitDataFrame.extract_laps(df)
        session = FitDataFrame.extract_session(df)
        records = FitDataFrame.extract_records(df)
        return clf(path=path, date=date, laps=laps, session=session, records=records)

    def plot_basics(self, ax=None, figsize=(15, 5)):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        self.records.plot(y="heart_rate", c="r", ax=ax)
        ax_twin = ax.twinx()
        self.records.plot(y="speed", c="b", ax=ax_twin)
        for start_time in self.laps.start_time:
            ax.axvline(x=start_time, linestyle="--", c="k")
        ax.set_title(f"Basics for {self.date}")

    def get_every_breaks(self) -> List[pd.DataFrame]:
        n_breaks = len(self.laps[self.laps.category == LAP_CATEGORY_BREAK])
        return [
            self.get_ith_lap_of_category(i, LAP_CATEGORY_BREAK) for i in range(n_breaks)
        ]

    def get_ith_lap_of_category(self, i, category):
        try:
            laps = self.laps[self.laps.category == category]
        except pd.errors.IndexingError:
            raise KeyError(f"No category {category} in laps")
        try:
            lap = laps.iloc[i]
        except IndexError:
            raise KeyError(f"Observation {i} out of range for {category=}")
        return self.records.between_time(
            start_time=lap.start_time.time(), end_time=lap.stop_time.time()
        )
