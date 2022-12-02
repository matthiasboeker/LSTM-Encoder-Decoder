from typing import List
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TSTrainTest:
    train_ts: pd.DataFrame
    val_ts: pd.DataFrame
    test_ts: pd.DataFrame
    feature_names: List[str]

    @classmethod
    def train_test_split_ts(cls, time_series_df: pd.DataFrame, test_split: float, val_split: float):
        train_len = int(len(time_series_df)*(1-test_split))
        val_len = int((len(time_series_df)-train_len)*val_split)
        train_ts = time_series_df.iloc[:train_len, :]
        val_ts = time_series_df.iloc[train_len:train_len+val_len, :]
        test_ts = time_series_df.iloc[train_len+val_len:, :]
        return cls(train_ts, val_ts, test_ts, time_series_df.columns)