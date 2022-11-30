import numpy as np
import pandas as pd
from scipy.special import expit


def s_shaped_func(acwr_x: float):
    if acwr_x < 1.3:
        return 0.02
    else:
        return expit(3.5*(acwr_x-1.3)-3.89)
