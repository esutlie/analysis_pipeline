from sklearn.manifold import Isomap
from create_bins_df import create_precision_df
import backend
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression, PoissonRegressor, GammaRegressor
import json
import os
from backend import get_data_path
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from thread_function import thread_function, model, get_isomap
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import metrics
from isomap import get_phase, get_block
from thread_function import get_x
from sklearn.model_selection import train_test_split


def main():
    files = backend.get_session_list()
    for session in files:
        [normalized_spikes, convolved_spikes, _, original_spikes], interval_ids, intervals_df = create_precision_df(
            session, regenerate=False)
        if normalized_spikes is None:
            continue
        blocks = intervals_df.block.unique()
        blocks.sort()
        phases = get_phase(interval_ids, intervals_df)
        phase_filter = np.where((phases == 1) | (phases == 2))[0]
        normalized_spikes = normalized_spikes[:, phase_filter]
        interval_ids = interval_ids[phase_filter]
        block = get_block(interval_ids, intervals_df)
        high_fr = np.where(np.mean(original_spikes, axis=1) * 1000 > 1)
        times, longest = get_x(interval_ids)
