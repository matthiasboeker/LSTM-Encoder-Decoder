from typing import Union
import numpy as np
import pandas as pd


class ReverseStandardScaler:

    def __init__(self, mean: np.array, standard_deviation: np.array):
        self.mean = mean
        self.std = standard_deviation

    @classmethod
    def fit(cls, to_transform_ts: Union[pd.Series, np.array]):
        mean = np.nanmean(to_transform_ts, dtype=np.float64)
        std = np.nanstd(to_transform_ts, dtype=np.float64)
        return cls(mean, std)

    def reverse_transform(self, transformed_ts: np.array) -> np.array:
        return transformed_ts * self.std + self.mean


class ReverseMinMaxScaler:

    def __init__(self, max_val: np.array, min_val: np.array):
        self.max_val = max_val
        self.min_val = min_val

    @classmethod
    def fit(cls, to_transform_ts: Union[pd.Series, np.array]):
        mean = np.nanmax(to_transform_ts, dtype=np.float64)
        std = np.nanmin(to_transform_ts, dtype=np.float64)
        return cls(mean, std)

    def reverse_transform(self, transformed_ts: np.array) -> np.array:
        return transformed_ts * (self.max_val - self.min_val) + self.min_val
