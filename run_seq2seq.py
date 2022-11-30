from enum import Enum
from typing import Any, Dict
from pathlib import Path
import pickle
import torch
import torch.nn as nn
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.reverse_transformation import ReverseStandardScaler, ReverseMinMaxScaler
from utils.data_wrangeling_funcs import TSTrainTest
from models.seq2seq import SEQ2SEQ
from torch.utils.data import DataLoader
from models.DataLoader import SequenceDataset

sys.path.append(str(Path.cwd().parent / "pmsys-pipeline"))
from pmsys_pipeline.pipeline_structure import (
    PreprocessingModule,
    TSTrainTestSplittingModule,
    Pipeline,
)
from pmsys_pipeline.preprocessing_module import strip_data


class Transformations(Enum):
    STANDARDSCALER = StandardScaler(), ReverseStandardScaler
    MINMAXSCALER = MinMaxScaler(), ReverseMinMaxScaler

    def __init__(self, transformation, reverse_transformation):
        self.transformation = transformation
        self.reverse_transformation = reverse_transformation


def load_in(path_to_players):
    with open(path_to_players, "rb") as f:
        players = pickle.load(f)
    return players


def init_pipelines(preprocessing):
    return {
        "train": Pipeline.initialise([preprocessing]),
        "val": Pipeline.initialise([preprocessing]),
        "test": Pipeline.initialise([preprocessing]),
    }

def run_pipeline(
    pipeline: Dict[str, Pipeline],
    data_obj: TSTrainTestSplittingModule,
    params: Dict[str, Any],
    show_sizes: bool = False,
):
    y_train = np.array(player_ts.train_ts[params["features_out"]]).reshape(-1, 1)
    y_val = np.array(player_ts.val_ts[params["features_out"]]).reshape(-1, 1)
    y_test = np.array(player_ts.test_ts[params["features_out"]]).reshape(-1, 1)
    if show_sizes:
        print("Training", y_train.shape)
        print("Validation", y_val.shape)
        print("Test", y_test.shape)

    X_train, _ = pipeline["train"].run(
        data_obj.train_ts, y_train, return_format="single-batch"
    )
    X_val, _ = pipeline["val"].run(data_obj.val_ts, y_val, return_format="single-batch")
    X_test, _ = pipeline["test"].run(
        data_obj.test_ts, y_test, return_format="single-batch"
    )

    X_train = pd.DataFrame(X_train, columns=data_obj.feature_names)
    X_val = pd.DataFrame(X_val, columns=data_obj.feature_names)
    X_test = pd.DataFrame(X_test, columns=data_obj.feature_names)
    return X_train, X_val, X_test


data_params = {
    "features_in": ["readiness", "index"],
    "features_out": "acwr",
    "test_split": 0.2,
    "val_split": 0.5,
    "sequence_length": 21,
    "output_length": 7,
}
model_params = {
    "batch_size": 32,
    "hidden_size": 128,
    "layers": 1,
    "epochs": 50,
    "dropout": 0,
    "l2_reg": 0,
    "learning_rate": 0.001,
    "loss_func": nn.MSELoss(),
}


if __name__ == "__main__":
    path_to_store_figures = Path(__file__).parent / "figures"
    players = load_in(Path(__file__).parent / "data" / "players.pkl")[:-6]
    players = pd.concat([strip_data(player).reset_index(drop=True).reset_index(drop=False) for player in players])

    player_ts = TSTrainTest.train_test_split_ts(
        players, test_split=data_params["test_split"], val_split=data_params["val_split"]
    )

    preprocessing = PreprocessingModule.init_module(
        [
            Transformations.STANDARDSCALER.transformation,
            SimpleImputer(missing_values=np.nan, strategy="mean"),
        ]
    )
    pipes = init_pipelines(preprocessing)

    X_train, X_val, X_test = run_pipeline(pipes, player_ts, data_params)

    rvs = Transformations.STANDARDSCALER.reverse_transformation.fit(player_ts.test_ts[data_params["features_out"]])

    np.random.seed(0)
    torch.manual_seed(0)

    features = data_params["features_in"]
    n_features = len(features)

    train_dataset = SequenceDataset(
        X_train,
        target=data_params["features_out"],
        features=features,
        input_length=data_params["sequence_length"],
        output_length=data_params["output_length"],
    )
    val_dataset = SequenceDataset(
        X_val,
        target=data_params["features_out"],
        features=features,
        input_length=data_params["sequence_length"],
        output_length=data_params["output_length"],
    )
    test_dataset = SequenceDataset(
        X_test,
        target=data_params["features_out"],
        features=features,
        input_length=data_params["sequence_length"],
        output_length=data_params["output_length"],
    )
    torch.manual_seed(99)
    train_loader = DataLoader(
        train_dataset, batch_size=model_params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=model_params["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=model_params["batch_size"], shuffle=False
    )

    model = SEQ2SEQ(
        n_features=n_features,
        hidden_size=model_params["hidden_size"],
        num_layers=model_params["layers"],
        target_len=data_params["output_length"],
        dropout=model_params["dropout"],
        loss_function=model_params["loss_func"],
    )
    model.train_model(
        train_loader,
        val_loader,
        n_epochs=model_params["epochs"],
        learning_rate=model_params["learning_rate"],
        l2_reg=model_params["l2_reg"],
        plot_loss=path_to_store_figures / "loss.png",
    )
    model.evaluate(test_loader, path_to_store_figures / "eval.png", reversed_transformer=rvs)
