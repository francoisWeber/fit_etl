from abc import abstractmethod
from typing import Tuple, List
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

from fit_etl.tooling import utils as u


class AnalysisModel:
    def __init__(
        self,
        needed_feature: str,
        params_names,
        name,
        key_params,
        params_bounds=None,
        params_init=None,
        do_square_target=False,
    ) -> None:
        self.needed_feature = needed_feature
        self.params_names: Tuple[str] = params_names
        self.key_params = key_params
        if not isinstance(self.key_params, list):
            self.key_params = [self.key_params]
        self.name: str = name
        self.params_bounds: Tuple[List[float]] = params_bounds
        self.params_init: Tuple[float] = params_init
        self.params_estimated: List[float] = None
        self.do_square_target = do_square_target

    def extract_xy_from_df(self, df: pd.DataFrame):
        df = u.reset_time(df)
        time = df.index.to_numpy()
        target = df[self.needed_feature].to_numpy()
        if self.do_square_target:
            target = np.square(target)
        target = target - target.min()
        return time, target

    @abstractmethod
    def estimate_bounds(target):
        ...

    @abstractmethod
    def estimate_init(target):
        ...

    def get_params(self) -> dict:
        return {k: v for k, v in zip(self.params_names, self.params_estimated)}

    def get_key_params(self) -> dict:
        return {
            k: v
            for k, v in zip(self.params_names, self.params_estimated)
            if k in self.key_params
        }

    def fit(self, df: pd.DataFrame):
        time, target = self.extract_xy_from_df(df)
        self.estimate_bounds(target)
        self.estimate_init(target)
        params_estimated, _ = curve_fit(
            f=self,
            xdata=time,
            ydata=target,
            p0=self.params_init,
            bounds=self.params_bounds,
        )
        self.params_estimated = params_estimated

    def get_description(self):
        descr = f"{self.name} on "
        descr += "sqared " if self.do_square_target else " "
        descr += self.needed_feature
        return descr


class PiecewiseExpDecreaseModel(AnalysisModel):
    """PiecewiseExpDecrease represent a 1st HR decrease model to be calibrated"""

    def __init__(self, needed_feature: str, do_square_target=False) -> None:
        params_names = ("start_level", "t_start", "tau", "t_stop")
        name = "piecewise_exp"
        key_params = "tau"
        super().__init__(
            params_names=params_names,
            name=name,
            key_params=key_params,
            needed_feature=needed_feature,
            do_square_target=do_square_target,
        )

    def estimate_bounds(self, target):
        n_pts = len(target)
        params_lower_bound = [
            0,  # start_level
            0,  # t_start
            1,  # tau
            n_pts / 3,  # t_stop
        ]
        params_upper_bound = [
            2 * target[0],  # start_level
            n_pts * 30 / 100,  # t_start
            n_pts * 30 / 100,  # tau
            n_pts,  # t_stop
        ]
        params_bounds = (params_lower_bound, params_upper_bound)
        self.params_bounds = params_bounds

    def estimate_init(self, target):
        n_pts = len(target)
        self.params_init = (
            target[0],
            n_pts * 5 / 100,
            n_pts * 20 / 100,
            n_pts * 80 / 100,
        )

    @staticmethod
    def __call__(t, start_level, t_start, tau, t_stop):
        piece_1 = np.where(t < t_start, start_level, 0)
        piece_2 = np.where(
            (t_start <= t) * (t < t_stop),
            start_level * np.exp(-(t - t_start) / tau),
            0,
        )
        piece_3 = np.where(
            t_stop < t, start_level * np.exp(-(t_stop - t_start) / tau), 0
        )
        return piece_1 + piece_2 + piece_3


class PiecewiseLinearDecreaseModel(AnalysisModel):
    """PiecewiseExpDecrease represent a 1st HR decrease model to be calibrated"""

    def __init__(self, needed_feature: str, do_square_target=False) -> None:
        params_names = ("start_level", "t_start", "alpha", "t_stop")
        name = "piecewise_lin"
        key_params = "alpha"
        super().__init__(
            needed_feature=needed_feature,
            do_square_target=do_square_target,
            params_names=params_names,
            name=name,
            key_params=key_params,
        )

    def estimate_bounds(self, target):
        n_pts = len(target)
        params_lower_bound = [
            0,  # start_level
            0,  # t_start
            0,  # alpha
            n_pts / 3,  # t_stop
        ]
        params_upper_bound = [
            max(target),  # start_level
            n_pts * 30 / 100,  # t_start
            max(target) / (len(target) / 2),  # alpha
            n_pts,  # t_stop
        ]
        params_bounds = (params_lower_bound, params_upper_bound)
        self.params_bounds = params_bounds

    def estimate_init(self, target):
        n_pts = len(target)
        self.params_init = (
            target[0],  # start_level
            n_pts * 5 / 100,  # t_start
            (max(target) - min(target)) / len(target),  # alpha
            n_pts * 80 / 100,  # t_stop
        )

    @staticmethod
    def __call__(t, start_level, t_start, alpha, t_stop):
        piece_1 = np.where(t < t_start, start_level, 0)
        piece_2 = np.where(
            (t_start <= t) * (t < t_stop),
            start_level - alpha * (t - t_start),
            0,
        )
        piece_3 = np.where(
            t_stop <= t,
            start_level - alpha * (t_stop - t_start),
            0,
        )
        return piece_1 + piece_2 + piece_3


class BreakAnalyzer:
    def __init__(self) -> None:
        self.models: List[AnalysisModel] = [
            PiecewiseLinearDecreaseModel(
                needed_feature="heart_rate_smoothed", do_square_target=True
            ),
            PiecewiseExpDecreaseModel(
                needed_feature="heart_rate_smoothed", do_square_target=True
            ),
            PiecewiseLinearDecreaseModel(needed_feature="joules_per_bpm_smoothed"),
            PiecewiseExpDecreaseModel(needed_feature="joules_per_bpm_smoothed"),
        ]

    def fit(self, df: pd.DataFrame):
        for model in self.models:
            model.fit(df)

    def get_params(self):
        params = []
        for model in self.models:
            for name, value in model.get_params().items():
                params.append(
                    {
                        "input_feature": model.needed_feature,
                        "model_name": model.name,
                        "squared": model.do_square_target,
                        "param_name": name,
                        "param_value": value,
                    }
                )
        return params

    def get_key_params(self):
        params = []
        for model in self.models:
            for name, value in model.get_key_params().items():
                params.append(
                    {
                        "input_feature": model.needed_feature,
                        "model_name": model.name,
                        "squared": model.do_square_target,
                        "param_name": name,
                        "param_value": value,
                    }
                )
        return params

    def get_key_params_as_columns(self):
        params = {}
        for model in self.models:
            model_str = ["squared_" if model.do_square_target else ""]
            model_str += [model.needed_feature, model.name]
            for name, value in model.get_key_params().items():
                model_name = "_".join(model_str + [name])
                params[model_name] = value
        return params

    def predict(self, time):
        predictions = {}
        for model in self.models:
            params = model.get_params().values()
            predictor = lambda t: model(t, *params)
            predictions[model.name] = predictor(time)
        return predictions
