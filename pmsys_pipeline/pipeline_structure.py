from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import pandas as pd
from sklearn.compose._column_transformer import _check_X

from pmsys_pipeline.summary_functions import get_correlation_summary
from pmsys_pipeline.summary_functions import get_feature_histogram_summary


def parse_to_numpy_tensor(data_obj: Union[np.array, pd.DataFrame]):
    if isinstance(data_obj, pd.DataFrame):
        data_obj = data_obj.to_records(index=False)
    return np.expand_dims(data_obj, axis=len(data_obj.shape))


@dataclass(frozen=True)
class Logger:
    name: str
    feature_correlation: List[Any]
    feature_distributions: List[Any]

    def plot_feature_histograms(self):
        pass


@dataclass(frozen=True)
class TransformerData:
    X: List[pd.DataFrame] = field(default_factory=List[pd.DataFrame])
    y: List[pd.DataFrame] = field(default_factory=List[pd.DataFrame])

    def get_summary(self):
        summary_statistics = {}
        summary_statistics.update(get_correlation_summary(self.X))
        summary_statistics.update(get_feature_histogram_summary(self.X))
        return summary_statistics


class Transformer:
    def __init__(self, function, arguments):
        self.function = function
        self.arguments = arguments

    def __call__(self, **arguments):
        self.arguments = arguments

    def fit_transform(
        self, to_transformed_data: pd.DataFrame, y: Union[None, np.array] = None
    ) -> pd.DataFrame:
        return self.function(to_transformed_data, **self.arguments)


@dataclass
class PipelineModule(ABC):

    transformers: Dict[str, Transformer] = field(default_factory=Dict[str, Transformer])
    module_summary: Dict[str, Logger] = field(default_factory=Dict[str, Logger])

    @classmethod
    def init_module(cls, input_transformers: List[Transformer]):
        transformers = {
            transformer.__class__.__name__: transformer
            for transformer in input_transformers
        }
        return cls(transformers, {})

    @abstractmethod
    def get_summary(self, data: TransformerData):
        pass

    @abstractmethod
    def run(self, transformer_data: TransformerData) -> TransformerData:
        pass


class PipelinePmsys:
    def __init__(self, modules: Dict[str, PipelineModule]):
        self.modules = modules

    @classmethod
    def initialise(cls, modules: List[PipelineModule]):
        return cls({module.__class__.__name__: module for module in modules})

    def run(self, X, y, return_format: str = "multi-batch"):
        X = _check_X(X)
        if y is not None:
            y = _check_X(y)
        data = TransformerData([X], [y])
        for name, module in self.modules.items():
            data = module.run(data)
        if return_format == "single-batch":
            (X,) = data.X
            (y,) = data.y
            return X, y.ravel()
        if return_format == "multi-batch":
            return data.X, data.y
        return data


def logger_transformation(X, y, name: str) -> Logger:
    correlation = get_correlation_summary(X)
    feature_distribution = get_feature_histogram_summary(X)
    return Logger(name, correlation, feature_distribution)


def run_module(input_data: TransformerData, transformers: Dict[str, Transformer]):
    X = input_data.X
    y = input_data.y
    loggers = {}
    for name, transformer in transformers.items():
        X = [transformer.fit_transform(batch) for batch in X]
        y = [transformer.fit_transform(batch) for batch in y]
        logger = logger_transformation(X, y, name)
        loggers.update({logger.name: logger})
    return TransformerData(X, y), loggers


class TSTrainTestSplittingModule(PipelineModule):
    @classmethod
    def init_module(cls, splitter):
        return cls({splitter.__class__.__name__: splitter}, {})

    def run(self, transformer_data) -> TransformerData:
        """Change the dict access"""
        splitter = list(self.transformers.values())[0]
        (X,) = transformer_data.X
        (y,) = transformer_data.y
        train_batches, val_batches = splitter.fit_transform(X, y)
        output_transformer_data = TransformerData(train_batches, val_batches)
        self.module_summary.update(output_transformer_data.get_summary())
        return output_transformer_data

    def get_summary(self, transformed_data: TransformerData):
        return transformed_data.get_summary()


class PreprocessingModule(PipelineModule):
    def run(self, transformer_data) -> TransformerData:
        output_transformer_data, logger = run_module(
            transformer_data, self.transformers
        )
        self.module_summary.update(logger)
        return output_transformer_data

    def get_summary(self, transformed_data: TransformerData):
        return transformed_data.get_summary()


class ImputationModule(PipelineModule):
    """Can be deleted!"""

    def run(self, transformer_data) -> TransformerData:
        output_transformer_data, logger = run_module(
            transformer_data, self.transformers
        )
        self.module_summary.update(logger)
        return output_transformer_data

    def get_summary(self, transformed_data: TransformerData):
        return transformed_data.get_summary()
