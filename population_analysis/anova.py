from sklearn.manifold import Isomap
from create_bins_df import create_precision_df, get_phase, get_block, get_x
import backend
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression, PoissonRegressor, GammaRegressor
import json
import os
from backend import get_data_path
import time
from sklearn.preprocessing import minmax_scale
from scipy.ndimage import convolve1d
import math
from center_of_mass import get_mean


def main():
    color_sets = backend.get_color_sets()
    files = backend.get_session_list()
    for session in files:
        if session != 'ES029_2022-09-14_bot72_0_g0':
            continue
        spikes, pi_events, cluster_info = backend.load_data(session)
        # licks = pi_events[(pi_events.key == 'lick') & (pi_events.value == 1)].time.to_numpy()
        licks = backend.extract_event(pi_events, 'lick', 1)
        head = backend.extract_event(pi_events, 'head', 1)
        [normalized_spikes, convolved_spikes, _, original_spikes], interval_ids, intervals_df = create_precision_df(
            session, regenerate=False)
        if convolved_spikes is None:
            continue
        num_units = len(original_spikes)

        bin_size = 30
        for i in np.unique(interval_ids):
            interval_spikes = original_spikes[:, interval_ids == i]
            interval_licks = licks[(intervals_df.loc[i].interval_starts < licks) &
                                   (licks < intervals_df.loc[i].interval_ends)] - intervals_df.loc[i].interval_starts
            num_bins = math.ceil(interval_spikes.shape[1] / bin_size)
            binned_spikes = np.array(
                [np.mean(arr, axis=1) for arr in np.array_split(interval_spikes, num_bins, axis=1)])
            plt.imshow(binned_spikes.T, extent=[0, interval_spikes.shape[1] / 1000, num_units, 0], aspect='auto',
                       interpolation='none')
            plt.title(f'interval {i}')
            plt.show()
            plt.imshow(binned_spikes.T, extent=[0, interval_spikes.shape[1] / 1000, num_units, 0], aspect='auto',
                       interpolation='none')
            plt.vlines(interval_licks, 0, len(convolved_spikes), 'r')
            plt.title(f'interval {i}')
            plt.show()
            # pull out neural activity bins with different bin sizes. reduce interval_ids to match. get blocks and other modifiers.


if __name__ == '__main__':
    main()
