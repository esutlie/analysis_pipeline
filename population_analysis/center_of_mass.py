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
        blocks = intervals_df.block.unique()
        blocks.sort()
        block = get_block(interval_ids, intervals_df)

        phases = get_phase(interval_ids, intervals_df)
        phase_filter = np.where((phases == 1) | (phases == 2))[0]
        spikes_to_use = spikes_to_use[:, phase_filter]
        interval_ids = interval_ids[phase_filter]
        block = block[phase_filter]

        high_fr = np.where(np.mean(original_spikes, axis=1) * 1000 > 1)[0]
        times, longest = get_x(interval_ids, min_longest=20)
        time_cutoff = times < longest.max()
        means = []
        stds = []
        x_s = []
        com_s = []
        order = []
        for i, b in enumerate(blocks):
            spikes = spikes_to_use[:, (b == block) & time_cutoff]
            spikes = spikes[high_fr]
            t = times[(b == block) & time_cutoff]
            mean_spikes, std_spikes, x = get_mean(spikes.T, t)
            mean_spikes = convolve1d(mean_spikes.T, kernel).T  # if using original spikes
            means.append(mean_spikes * 1000)
            stds.append(std_spikes)
            x_s.append(x)
            com_s.append(com(mean_spikes, x))
            order.append(np.argsort(np.argmax(mean_spikes, axis=0)))

        rel_pos = np.array([[[np.where(order[k] == val1)[0][0] > np.where(order[k] == val2)[0][0]
                              for val1 in range(len(high_fr))] for val2 in range(len(high_fr))] for k in range(2)])
        for j in range(len(high_fr)):
            order_fraction = sum(~(rel_pos[0, j] ^ rel_pos[1, j])) / len(high_fr)
            plt.plot(x_s[0], means[0][:, j], c=color_sets['set2'][0])
            plt.plot(x_s[1], means[1][:, j], c=color_sets['set2'][1])
            plt.scatter(com_s[0][j], 0, c=color_sets['set2'][0], s=15)
            plt.scatter(com_s[1][j], 0, c=color_sets['set2'][1], s=15)
            plt.legend(blocks)
            stds_scaled = [scaled_std(means[0][:, j]), scaled_std(means[1][:, j])]
            plt.title(f'Unit {j}, Order Conservation {order_fraction:.2f}, {stds_scaled[0]:.2f} {stds_scaled[1]:.2f}')
            plt.show()

        plt.scatter(com_s[1], com_s[0])
        plt.plot([0, 6], [0, 6], c=[.8, .8, .8])
        plt.xlabel(f'{blocks[1]} center of mass time (sec)')
        plt.ylabel(f'{blocks[0]} center of mass time (sec)')
        plt.title(f'units from {session}')
        plt.show()


def scaled_std(values):
    return np.std(values) / np.mean(values)


def com(spikes, x):
    return np.sum(minmax_scale(spikes).T * x, axis=1) / np.sum(minmax_scale(spikes).T, axis=1)


def get_mean(pred, x):
    times = np.unique(x)
    times.sort()
    mean_neural_time = np.zeros([len(times), pred.shape[-1]])
    std_neural_time = np.zeros([len(times), pred.shape[-1]])
    for j, x_i in enumerate(times):
        mean_neural_time[j] = np.mean(pred[np.where(x == x_i)[0]], axis=0)
        std_neural_time[j] = np.std(pred[np.where(x == x_i)[0]], axis=0)
    return mean_neural_time, std_neural_time, times


if __name__ == '__main__':
    main()
