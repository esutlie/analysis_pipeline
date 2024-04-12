from sklearn.manifold import Isomap
from create_bins_df import create_precision_df, get_gaussian_kernel
import backend
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression, PoissonRegressor, GammaRegressor
import json
import os
from backend import get_data_path
import time
from population_analysis.old_stuff.isomap import get_phase, get_block
from thread_function import get_x
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
        [normalized_spikes, convolved_spikes, _, original_spikes], interval_ids, intervals_df = create_precision_df(
            session, regenerate=False)
        spikes_to_use = original_spikes
        kernel = get_gaussian_kernel(l=1000, sigma=1000 / 5)  # if using original spikes

        if spikes_to_use is None:
            continue

        intervals_df['length'] = intervals_df['interval_ends'] - intervals_df['interval_starts']
        blocks = intervals_df.block.unique()
        blocks.sort()
        block = get_block(interval_ids, intervals_df)

        phases = get_phase(interval_ids, intervals_df)
        phase_filter = np.where((phases == 2))[0]
        spikes_to_use = spikes_to_use[:, phase_filter]
        interval_ids = interval_ids[phase_filter]
        block = block[phase_filter]

        high_fr = np.where(np.mean(original_spikes, axis=1) * 1000 > 1)[0]
        times, longest = get_x(interval_ids, min_longest=1)
        lengths = intervals_df.length.loc[np.unique(interval_ids)]
        num_groups = 3
        time_groups = np.array_split(np.array(lengths.index[np.argsort(lengths.values)]), num_groups)
        colors = [color_sets['set2'], color_sets['set2_med_dark'], color_sets['set2_dark']]
        # means = []
        # stds = []
        # x_s = []
        # com_s = []
        # order = []
        for unit in high_fr:
            for i, b in enumerate(blocks):
                for j in range(num_groups):
                    time_cutoff = np.array([val in time_groups[j] for val in interval_ids])
                    spikes = spikes_to_use[:, (b == block) & time_cutoff]
                    spikes = spikes[unit]
                    t = times[(b == block) & time_cutoff]
                    mean_spikes, std_spikes, x = get_mean(spikes.T, t)
                    mean_spikes = convolve1d(mean_spikes.T, kernel).T  # if using original spikes
                    # means.append(mean_spikes * 1000)
                    # stds.append(std_spikes)
                    # x_s.append(x)
                    plt.plot(x, mean_spikes * 1000, c=list(colors[j][i]))
                plt.title(f'unit {unit}, block {b}')
                plt.show()


if __name__ == '__main__':
    main()
