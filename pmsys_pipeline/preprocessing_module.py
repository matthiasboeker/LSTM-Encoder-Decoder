from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from pmsys_pipeline.pipeline_structure import Transformer


class TrainSetLargerThanValSet(Exception):
    pass


def strip_data(input_df: pd.DataFrame) -> pd.DataFrame:
    mask = input_df.replace(0, np.nan)
    f_idx = mask.first_valid_index()
    l_idx = mask.last_valid_index()
    return input_df.loc[f_idx:l_idx, :]


def ts_train_val_split(
    predictor_ts: Union[pd.DataFrame, np.array],
    dependent_ts: Union[pd.DataFrame, np.array],
    window_size: int,
    lag: int,
    validate_size: int,
) -> Tuple[np.array, np.array]:

    if isinstance(predictor_ts, pd.DataFrame):
        X_feature_names = predictor_ts.columns
    else:
        X_feature_names = range(0, predictor_ts.shape[1])

    if window_size - validate_size < validate_size:
        raise TrainSetLargerThanValSet(
            "The training set is smaller than validation set."
        )

    predictor_ts = np.array(predictor_ts)
    dependent_ts = np.array(dependent_ts)
    predictor_batches = [
        predictor_ts[i: i + window_size, :]
        for i in range(0, len(predictor_ts) - window_size, lag)
    ]
    dependent_batches = [
        dependent_ts[i: i + window_size, :]
        for i in range(0, len(dependent_ts) - window_size, lag)
    ]

    train_batches = []
    val_batches = []
    for pred_batch, dep_batch in zip(predictor_batches, dependent_batches):
        val_batch = pd.DataFrame(dep_batch[len(dep_batch) - validate_size :, :])
        train_batch = pd.DataFrame(
            pred_batch[: len(dep_batch) - validate_size, :], columns=X_feature_names
        )
        train_batches.append(train_batch)
        val_batches.append(val_batch)
    return train_batches, val_batches


class TSTrainValSplit:
    def __init__(self, window_size: int, lag: int, validation_size: int):
        self.window_size = window_size
        self.lag = lag
        self.validation_size = validation_size

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame):
        return ts_train_val_split(
            X, y, self.window_size, self.lag, self.validation_size
        )


def normalise_data(input_df: pd.DataFrame) -> pd.DataFrame:
    return (input_df - input_df.min()) / (input_df.max() - input_df.min())


class Normalise(Transformer):
    def __init__(self):
        super().__init__(
            normalise_data,
            {},
        )
