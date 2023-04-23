from typing import List
from fit_etl.tooling.constants import *
from fit_etl.extract.fidata import FitDataFrame
from fit_etl.extract.fitfile import decode_fitfile

from scipy import ndimage

import pandas as pd
from matplotlib import pyplot as plt
import warnings

import seaborn as sns

warnings.filterwarnings("ignore", module="fitdecode")


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
        self.enhance_data()

    def enhance_data(self):
        # adding new quantities
        if "mps_per_bpm" not in self.records:
            self.records["mps_per_bpm"] = (
                self.records.speed * 1000 / 3600 / self.records.heart_rate
            )

        if "joules_per_bpm" not in self.records:
            self.records["joules_per_bpm"] = (
                0.5
                * BODY_MASS
                * (self.records.speed * 1000 / 3600) ** 2
                / self.records.heart_rate
            )

        # smooting some data
        for raw_key in ["heart_rate", "speed", "mps_per_bpm", "joules_per_bpm"]:
            smoothed_key = raw_key + "_smoothed"
            if smoothed_key not in self.records:
                self.records[smoothed_key] = ndimage.gaussian_filter1d(
                    self.records[raw_key], SMOOTHING_SIGMA
                )

        # naming laps
        if "lap_type" not in self.records:
            type2count = {}
            laps_names = []
            for lap_type in self.laps.category:
                type2count[lap_type] = type2count.get(lap_type, 0) + 1
                laps_names.append(f"{lap_type}-{type2count[lap_type]}")
            self.laps["lap_name"] = laps_names
            # report them on records
            ts = pd.to_datetime(self.records.timestamp)
            self.records["lap_type"] = None
            for _, lap in self.laps.iterrows():
                index = ts.between(lap.start_time, lap.stop_time)
                self.records.loc[index, "lap_type"] = lap.lap_name

    @classmethod
    def from_dataframe(clf, df: pd.DataFrame, path=None):
        date = FitDataFrame.extract_date(df)
        laps = FitDataFrame.extract_laps(df)
        session = FitDataFrame.extract_session(df)
        records = FitDataFrame.extract_records(df)
        return clf(path=path, date=date, laps=laps, session=session, records=records)

    @classmethod
    def from_fitjson(clf, path_or_buf):
        df = pd.read_json(path_or_buf=path_or_buf)
        return Workout.from_dataframe(df, path_or_buf)

    @classmethod
    def from_fit(clf, path_or_buf):
        df = pd.DataFrame(decode_fitfile(path_or_buf))
        return Workout.from_dataframe(df, path_or_buf)

    def plot_basics(self, ax=None, figsize=(15, 5), use_smoothed_version=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        hr_key = "heart_rate" + ("_smoothed" if use_smoothed_version else "")
        speed_key = "speed" + ("_smoothed" if use_smoothed_version else "")
        self.records.plot(y=hr_key, c="r", ax=ax)
        ax.legend(loc="upper left")
        self.records.plot(y=speed_key, c="b", ax=ax, secondary_y=True)
        for start_time in self.laps.start_time:
            ax.axvline(x=start_time, linestyle="--", c="k")
        ax.set_title(f"Basics for {self.date}")
        return fig

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
