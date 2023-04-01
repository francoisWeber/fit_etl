from fit_etl.fidata import FitDataFrame

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
    def from_json(clf, path_or_buf):
        df = pd.read_json(path_or_buf=path_or_buf)
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
