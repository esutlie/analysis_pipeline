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
from isomap import get_phase, get_block
from thread_function import get_x
from sklearn.preprocessing import minmax_scale
from scipy.ndimage import convolve1d
import math
from center_of_mass import get_mean


def get_binned_quantiles(session, verbose=False, regenerate=False):
    local_path = os.path.join(backend.get_data_path(), session)
    binned_counts_path = os.path.join(local_path, 'binned_counts.npy')
    binned_x_path = os.path.join(local_path, 'binned_x.npy')
    binned_quantiles_path = os.path.join(local_path, 'binned_quantiles.npy')
    paths = [binned_x_path, binned_counts_path, binned_quantiles_path]
    if np.all([os.path.exists(path) for path in paths]) and not regenerate:
        return np.load(binned_counts_path), np.load(binned_x_path), np.load(binned_quantiles_path)

    bin_size = .3
    num_quantiles = 5
    [_, _, _, original_spikes], interval_ids, intervals_df = create_precision_df(session, regenerate=False)
    x, _ = get_x(interval_ids)
    if original_spikes is None:
        return False
    intervals_df['length'] = intervals_df['interval_ends'] - intervals_df['interval_starts']
    num_units = len(original_spikes)
    num_time_bins = math.ceil(intervals_df['length'].max() / bin_size)
    binned_counts = np.zeros([num_units, num_time_bins, num_quantiles + 1])
    binned_x = []
    binned_quantiles = []
    for unit in range(num_units):
        bin_counts = []
        for trial in np.unique(interval_ids):
            num_groups = math.floor(intervals_df.loc[trial].length / bin_size)
            if num_groups > 0:
                trial_spikes = original_spikes[unit, (interval_ids == trial)][:round(num_groups * bin_size * 1000)]
                groups = np.sum(np.stack(np.array_split(trial_spikes, num_groups)), axis=1)
                bin_counts.append(groups.astype(int))
        all_counts = np.hstack(bin_counts)
        non_zero_counts = all_counts[all_counts != 0]
        quants = [np.quantile(non_zero_counts, q) for q in np.linspace(0, 1 - 1 / num_quantiles, num_quantiles)]

        unit_quants = [[sum(val > np.array(quants)) for val in trial_count] for trial_count in bin_counts]
        binned_quantiles.append(backend.flatten_list(unit_quants))
        if not len(binned_x):
            unit_x = [list(range(len(trial_count))) for trial_count in bin_counts]
            binned_x = backend.flatten_list(unit_x)

        for b in range(num_time_bins):
            assigned_quants = [sum(count[b] > np.array(quants)) for count in bin_counts if len(count) > b]
            counts = np.bincount([sum(count[b] > np.array(quants)) for count in bin_counts if len(count) > b])
            counts = np.concatenate([counts, np.zeros([num_quantiles + 1])])
            binned_counts[unit, b] = counts[:num_quantiles + 1]
        if verbose:
            print(np.unique(all_counts))
            print(quants)
            print(np.bincount(all_counts)[np.unique(all_counts)])
            quant_counts = [sum(all_counts == 0)]
            for i in range(len(quants) - 1):
                quant_counts.append(sum((all_counts >= quants[i]) & (all_counts < quants[i + 1])))
            quant_counts.append(sum(all_counts >= quants[-1]))
            print(quant_counts)
            print()

    np.save(binned_counts_path, binned_counts)
    np.save(binned_x_path, np.array(binned_x))
    np.save(binned_quantiles_path, np.stack(binned_quantiles))
    return binned_counts, binned_x, binned_quantiles


def get_bayes(binned_quantiles):
    p_fr = np.sum(binned_quantiles, axis=1) / np.sum(binned_quantiles[0])
    bayes = binned_quantiles / np.sum(binned_quantiles[0]) / p_fr[:, None, :]
    bayes[np.isnan(bayes)] = 0
    return bayes


# def predict():

def main():
    color_sets = backend.get_color_sets()
    files = backend.get_session_list()
    for session in files:
        if session != 'ES029_2022-09-14_bot72_0_g0':
            continue
        binned_counts, binned_x, binned_quantiles = get_binned_quantiles(session)
        bayes = get_bayes(binned_counts)
        w = np.ones([binned_counts.shape[0]])
        bayes[np.arange(len(bayes)), :, binned_quantiles[:, 0]]


        print('done')

        # blocks = intervals_df.block.unique()
        # blocks.sort()
        # block = get_block(interval_ids, intervals_df)
        #
        # phases = get_phase(interval_ids, intervals_df)
        # phase_filter = np.where((phases == 2))[0]
        # spikes_to_use = spikes_to_use[:, phase_filter]
        # interval_ids = interval_ids[phase_filter]
        # block = block[phase_filter]
        #
        # high_fr = np.where(np.mean(original_spikes, axis=1) * 1000 > 1)[0]
        # times, longest = get_x(interval_ids, min_longest=1)
        # lengths = intervals_df.length.loc[np.unique(interval_ids)]
        # num_groups = 3
        # time_groups = np.array_split(np.array(lengths.index[np.argsort(lengths.values)]), num_groups)
        # colors = [color_sets['set2'], color_sets['set2_med_dark'], color_sets['set2_dark']]
        # # means = []
        # # stds = []
        # # x_s = []
        # # com_s = []
        # # order = []
        # for unit in high_fr:
        #     for i, b in enumerate(blocks):
        #         for j in range(num_groups):
        #             time_cutoff = np.array([val in time_groups[j] for val in interval_ids])
        #             spikes = spikes_to_use[:, (b == block) & time_cutoff]
        #             spikes = spikes[unit]
        #             t = times[(b == block) & time_cutoff]
        #             mean_spikes, std_spikes, x = get_mean(spikes.T, t)
        #             mean_spikes = convolve1d(mean_spikes.T, kernel).T  # if using original spikes
        #             # means.append(mean_spikes * 1000)
        #             # stds.append(std_spikes)
        #             # x_s.append(x)
        #             plt.plot(x, mean_spikes * 1000, c=list(colors[j][i]))
        #         plt.title(f'unit {unit}, block {b}')
        #         plt.show()


if __name__ == '__main__':
    main()
