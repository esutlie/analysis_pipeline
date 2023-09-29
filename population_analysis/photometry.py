import pandas as pd
from sklearn.manifold import Isomap
from create_bins_df import create_precision_df, get_phase, get_block
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
from thread_function import thread_function, model, get_isomap, get_x
import math
import itertools
from scipy.optimize import curve_fit
from RegscorePy.bic import bic
from inspect import signature
from sklearn.metrics import r2_score
from scipy.interpolate import griddata
from scipy.stats import ttest_ind
import seaborn as sns


def reward_plots(split):
    save_path = os.path.join(os.getcwd(), 'figures', 'photometry', split)
    # session_data_columns = ['session', 'centers', 'unfiltered_centers', 'sig', 'unfiltered_sig']
    # if os.path.exists(session_data_path) and not regenerate:
    #     session_data = pd.read_pickle(session_data_path)
    # else:
    #     session_data = pd.DataFrame(columns=session_data_columns)
    color_sets = backend.get_color_sets()
    files = backend.get_session_list(photometry=True)
    # points_all = [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []]]
    xy_range = [0, 5]
    # heat_map_data = []
    # heat_map_x = np.linspace(0, 10, 1000)
    for session in files:
        # if session != 'ES029_2022-09-14_bot72_0_g0':
        #     continue
        [_, _, _, original_spikes], interval_ids, intervals_df = create_precision_df(
            session, regenerate=False, photometry=True)
        spikes_to_use = original_spikes
        if spikes_to_use is None:
            continue
        # flex_bin_origin = get_flex_bins(intervals_df, bin_time=50)
        # flex_bin_centers = (flex_bin_origin[:-1] + flex_bin_origin[1:]) / 2
        # flex_bins = np.round(flex_bin_origin[1:] * 1000)
        intervals_df['trial_starts'] = [intervals_df[(intervals_df.interval_trial == row[1].interval_trial) & (
                (intervals_df.interval_phase == 0) | (intervals_df.interval_phase == 3.))].interval_starts.iloc[0]
                                        for row in intervals_df.iterrows()]
    #     activity_levels = []
    #     coefs = []
    #     best_functions = []
    #     flex_bin_counts_list = []
    #     significant = []
    #     for unit in range(len(spikes_to_use)):
    #         spikes = spikes_to_use[unit]
    #         activity_levels.append(np.mean(spikes) * 1000)
    #         best_coef, best_function, flex_bin_counts, sig = compare_plot(spikes, unit, interval_ids, intervals_df,
    #                                                                       session,
    #                                                                       plot=plot_fit, plot_best=plot_best,
    #                                                                       split=split,
    #                                                                       flex_bins=flex_bins)
    #         coefs.append(best_coef)
    #         best_functions.append(best_function)
    #         flex_bin_counts_list.append(flex_bin_counts)
    #         significant.append(sig)
    #         if np.mean(spikes) * 1000 < .8:
    #             print()
    #     # global unit_data
    #     # unit_df = pd.read_pickle(unit_data.path)
    #     # gridx, gridy = np.mgrid[0:100, 0:100]
    #
    #     for unit_data in flex_bin_counts_list:
    #         if unit_data is not None:
    #             fig, ax = plt.subplots(1, 1, figsize=[7, 5])
    #             ax.plot(flex_bin_centers, unit_data)
    #             plt.show()
    #             heat_map_data.append(griddata(flex_bin_centers, unit_data, heat_map_x, method='nearest'))
    #             # plt.plot(np.linspace(0, flex_bin_origin[-1], 1000),griddata(flex_bin_centers, unit_data, (np.linspace(0, flex_bin_origin[-1], 1000)), method='nearest'))
    #             # plt.show()
    #     significant = np.array(significant)
    #     best_functions = np.array([f.__name__ for f in best_functions])
    #     centers = np.array([[c0[0] for c0 in c] for c in coefs])
    #     height = np.array([np.min([abs(c0[2]) for c0 in c]) for c in coefs])
    #     max_spread = np.array([np.max([c0[1] for c0 in c]) for c in coefs])
    #     filtered_centers = centers[(max_spread < (height * 70 - 9)) & (np.all(abs(centers) < 5, axis=1))]
    #     filtered_significant = significant[(max_spread < (height * 70 - 9)) & (np.all(abs(centers) < 5, axis=1))]
    #     session_data = session_data.append(
    #         pd.DataFrame([[session, filtered_centers, centers, filtered_significant, significant]],
    #                      columns=session_data_columns),
    #         ignore_index=True)
    #     if len(centers.T) == 2:
    #         sets = [[0, 1]]
    #     elif len(centers.T) == 3:
    #         sets = [[0, 1], [1, 2], [0, 2]]
    #     else:
    #         sets = [[i, i + 1] for i in range(len(centers.T) - 1)]
    #     for k, [i, j] in enumerate(sets):
    #         x, y = centers.T[i], centers.T[j]
    #         height_filter = max_spread < (height * 70 - 9)
    #         filter = (x < xy_range[1]) & (y < xy_range[1]) & height_filter
    #         # filter = (x < xy_range[1]) & (y < xy_range[1]) & (spread_dif < 25) & (height > .15) & (
    #         #         best_functions != 'exponential')
    #
    #         if sum(filter) == 0:
    #             continue
    #         print(f'{session}: {np.sum(filter)}/{len(filter)} units kept')
    #         x, y = x[filter], y[filter]
    #         points_all[k][0].append(x)
    #         points_all[k][1].append(y)
    #         # x_all.append(x)
    #         # y_all.append(y)
    #         if len(x) > 75:
    #             fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    #
    #             ax.plot(xy_range, xy_range, c='k', label='no change', alpha=.5)
    #             coef, _ = curve_fit(fixed_linear, x, y)
    #             ax.plot(xy_range, fixed_linear(np.array(xy_range), *coef), label='regression', alpha=.5)
    #             ax.plot(x, y, '.', c=color_sets['set1'][1], label='centers')
    #
    #             y_predicted = fixed_linear(x, *coef)
    #             ax.annotate("r-squared = {:.3f}".format(r2_score(y, y_predicted)), (3, .1))
    #
    #             ax.set_xlim(xy_range)
    #             ax.set_ylim(xy_range)
    #             ax.set_xlabel(f'{split_labels[split][i]}\ncenter of field (sec)')
    #             ax.set_ylabel(f'{split_labels[split][j]}\ncenter of field (sec)')
    #             ax.legend()
    #
    #             ax.set_title(f'Relative Field Shifts, {session}, {split}, {np.sum(filter)}/{len(filter)}')
    #             ax.set_aspect('equal', adjustable='box')
    #             plt.subplots_adjust(top=0.88)
    #             backend.save_fig(fig, session + '_shift_' + str(k), sub_folder=os.path.join('curve_fit', split))
    #             plt.show()
    # session_data.to_pickle(session_data_path)
    #
    # heat_map_data = np.array(heat_map_data)
    # normalized = heat_map_data / np.max(heat_map_data, axis=1, keepdims=True)
    # sorted_normalized = normalized[np.argsort(np.argmax(np.mean(normalized, axis=2), axis=1))]
    # for i in range(sorted_normalized.shape[-1]):
    #     heat_map_split = sorted_normalized[:, :, i]
    #     # normalized = heat_map_split.T / np.max(heat_map_split, axis=1)
    #     # sorted_normalized = normalized[:, np.argsort(np.argmax(normalized, axis=0))]
    #     heatmap_df = pd.DataFrame(heat_map_split, columns=np.round(heat_map_x, 2),
    #                               index=list(range(len(heat_map_split))))
    #     fig, ax = plt.subplots(1, 1, figsize=[7, 5])
    #     sns.heatmap(heatmap_df, ax=ax, xticklabels=100, yticklabels=False)
    #     ax.set_ylabel('Units')
    #     ax.set_xlabel('Seconds')
    #     ax.set_title(f'Heat Map: {split_labels[split][i]} ')
    #     backend.save_fig(fig, f'heat_map_{split}_{i}', sub_folder=os.path.join('curve_fit', split))
    #     plt.show()
    #
    # # plt.hist(activity_levels, bins=2000)
    # # plt.xlim([-.2, 2])
    # # plt.show()
    #
    # for i in range(len(points_all)):
    #     if not len(points_all[i][0]):
    #         continue
    #     x_all = np.concatenate(points_all[i][0])
    #     y_all = np.concatenate(points_all[i][1])
    #     coef, _ = curve_fit(fixed_linear, x_all, y_all)
    #     fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    #     ax.plot(x_all, y_all, '.', c=color_sets['set1'][1], label='centers', alpha=.5)
    #     ax.plot(xy_range, xy_range, c='k', label='no change')
    #     ax.plot(xy_range, fixed_linear(np.array(xy_range), *coef), label='regression')
    #     y_predicted = fixed_linear(x_all, *coef)
    #     ax.annotate("r-squared = {:.3f}".format(r2_score(y_all, y_predicted)), (3, .1))
    #     ax.set_xlim(xy_range)
    #     ax.set_ylim(xy_range)
    #     ax.set_xlabel(f'{split_labels[split][sets[i][0]]}\ncenter of field (sec)')
    #     ax.set_ylabel(f'{split_labels[split][sets[i][1]]}\ncenter of field (sec)')
    #     ax.legend()
    #     ax.set_title(f'Relative Field Shifts, All Units, {split}')
    #     ax.set_aspect('equal', adjustable='box')
    #     plt.subplots_adjust(top=0.88)
    #
    #     backend.save_fig(fig, f'all_session_shifts_{i}', sub_folder=os.path.join('curve_fit', split))
    #     plt.show()


if __name__ == '__main__':
    split_labels = {
        'block': ['low rate block', 'high rate block'],
        'leave': ['late leave', 'medium leave', 'early leave'],
        'time': ['early in trial', 'middle of trial', 'late in trial']
    }
    for key in split_labels.keys():
        reward_plots(key)
