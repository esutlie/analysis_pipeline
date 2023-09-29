import numpy as np
import pandas as pd


def min_dif(a, b, tolerance=0, return_index=False, rev=False):
    if type(a) == pd.core.series.Series:
        a = a.values
    if type(b) == pd.core.series.Series:
        b = b.values
    if rev:
        outer = -1 * np.subtract.outer(a, b)
        outer[outer <= tolerance] = np.nan
    else:
        outer = np.subtract.outer(b, a)
        outer[outer <= tolerance] = np.nan
    # noinspection PyBroadException
    mins = np.nanmin(outer, axis=0)
    if return_index:
        index = np.nanargmin(outer, axis=0)
        return index, mins
    return mins
