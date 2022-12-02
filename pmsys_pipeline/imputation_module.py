from typing import Union

import numpy as np
import pandas as pd

from pmsys_pipeline.pipeline_structure import Transformer


def forward_fill(input_df: Union[np.array, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(input_df, pd.DataFrame):
        return input_df.ffill()
    return pd.DataFrame(input_df).ffill().to_numpy()


class ForwardFill(Transformer):
    def __init__(self):
        super().__init__(
            forward_fill,
            {},
        )
