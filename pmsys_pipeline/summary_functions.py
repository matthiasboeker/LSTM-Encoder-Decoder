from typing import Any
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import pandas as pd


def get_feature_histogram_summary(
    batches: List[Union[pd.DataFrame, np.array]]
) -> List[Any]:
    feat_distributions = []
    for batch in batches:
        batch_histograms: Dict = {}
        for i in range(0, batch.shape[1]):
            feature = pd.DataFrame(batch).iloc[:, i]
            batch_histograms[feature.name] = feature.value_counts(bins=10)
        feat_distributions.append(batch_histograms)
    return feat_distributions


def get_correlation_summary(batches: List[Union[pd.DataFrame, np.array]]) -> List[Any]:
    correlations = []
    for batch in batches:
        correlations.append(pd.DataFrame(batch).corr())
    return correlations
