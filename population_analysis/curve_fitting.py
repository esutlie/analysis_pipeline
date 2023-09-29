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
from inspect import currentframe

plasma = matplotlib.cm.get_cmap('plasma')


class Timer:
    def __init__(self):
        self.time = time.time()

    def check(self):
        now = time.time()
        line = currentframe().f_back.f_lineno
        print(f'line {line}: {now - self.time} seconds')
        self.time = now


class UnitData:
    def __init__(self, regenerate=False):
        self.columns = ['unit', 'height', 'function', 'spread_dif', 'max_spread', 'filter']
        self.path = os.path.join(os.getcwd(), 'figures', 'curve_fit', 'unit_data.pkl')
        if regenerate:
            self.df = pd.DataFrame(columns=self.columns)
            self.df.to_pickle(self.path)
        else:
            self.df = pd.read_pickle(self.path)

    def update(self, data):
        self.df = pd.read_pickle(self.path)
        self.df = self.df.append(pd.DataFrame(data=[data], columns=self.columns), ignore_index=True)
        self.df.to_pickle(self.path)


def disqualified():
    return [
        'SZ030_2023-05-16_18-33-50',
        'SZ030_2023-05-16_19-34-51',
        'SZ030_2023-05-17_16-47-15',
        'SZ030_2023-05-22_13-41-42',
        'SZ030_2023-05-23_20-58-45',
        'SZ033_2023-07-28_16-24-34',
        'SZ033_2023-08-01_12-39-53',
        'SZ033_2023-08-01_16-51-05',
        'SZ033_2023-08-02_13-08-08',
        'SZ033_2023-08-02_14-23-16',
        'SZ033_2023-08-03_13-27-38',
        'SZ033_2023-08-03_15-02-01',
        'SZ034_2023-08-01_13-05-46',
        'SZ034_2023-08-01_17-12-50',
        'SZ034_2023-08-02_13-31-19',
        'SZ034_2023-08-02_14-44-31',
        'SZ034_2023-08-03_15-30-21',
        'SZ035_2023-08-01_13-41-58',
        'SZ035_2023-08-01_17-36-25',
        'SZ035_2023-08-02_13-53-28',
        'SZ035_2023-08-03_15-51-39',

    ]


def gaussian_pos(x, x0, sigma, A, H):  # x0: x shift, sigma: spread, A: height, H: y shift
    A = abs(A) if abs(A) > .05 else .05
    sigma = abs(sigma) if abs(sigma) < 5 else 5
    return H + abs(A) * np.exp(-(x - abs(x0)) ** 2 / (2 * sigma ** 2))


def a_gaussian_pos(x, x0, sigma, A, H1, H2):  # x0: x shift, sigma: spread, A: height, H: y shift
    A = abs(A) if abs(A) > .05 else .05
    sigma = abs(sigma) if abs(sigma) < 5 else 5
    H1 = abs(H1)
    H2 = abs(H2)
    # mean = np.mean([H1, H2])
    if H1 - H2 > abs(A):
        print()
        #     H1 = mean + abs(A) / 4
        H2 = H1 - abs(A)
    if H2 - H1 > abs(A) / 2:
        # H1 = mean - abs(A) / 4
        H2 = H1 + abs(A) / 2

    H = (x > x0) * H1 + (x <= x0) * H2
    A = (x > x0) * A + (x <= x0) * (A + H1 - H2)
    return H + abs(A) * np.exp(-(x - abs(x0)) ** 2 / (2 * sigma ** 2))


# plt.plot(a_gaussian_pos(np.linspace(0, 6, 100), x0, sigma, A, H1, H2))
# plt.show()


def gaussian_neg(x, x0, sigma, A, H):
    A = abs(A) if abs(A) > .05 else .05
    sigma = abs(sigma) if abs(sigma) < 5 else 5
    return H + -abs(A) * np.exp(-(x - abs(x0)) ** 2 / (2 * sigma ** 2))


def a_gaussian_neg(x, x0, sigma, A, H1, H2):
    A = abs(A) if abs(A) > .05 else .05
    sigma = abs(sigma) if abs(sigma) < 5 else 5
    H1 = abs(H1)
    H2 = abs(H2)
    if H1 - H2 > abs(A) / 2:
        # H1 = mean + abs(A) / 4
        H2 = H1 - abs(A) / 2
    elif H2 - H1 > abs(A):
        print()
        #     H1 = mean - abs(A) / 4
        H2 = H1 + abs(A)
    H = (x > x0) * H1 + (x <= x0) * H2
    A = (x > x0) * A + (x <= x0) * (A + H2 - H1)
    return H + -abs(A) * np.exp(-(x - abs(x0)) ** 2 / (2 * sigma ** 2))


def logistic_pos(x, x0, k, A, off):  # x0: x shift, k: steepness/flip, A: span, off: y shift
    return A / (1 + np.exp(-abs(k) * (x - abs(x0)))) + off


def logistic_neg(x, x0, k, A, off):  # x0: x shift, k: steepness/flip, A: span, off: y shift
    return A / (1 + np.exp(abs(k) * (x - abs(x0)))) + off


def exponential(x, tau, y0, A):  # tau: spread, A: height, y0: y shift
    return y0 + abs(A) * np.exp(-np.log(2) / abs(tau) * x)


def exp_gauss(x, x0, sigma, A, tau, a, y0):
    A = abs(A) if abs(A) > .05 else .05
    sigma = abs(sigma) if abs(sigma) < 5 else 5
    return y0 + abs(a) * np.exp(-np.log(2) / abs(tau) * x) + abs(A) * np.exp(-(x - abs(x0)) ** 2 / (2 * sigma ** 2))


def linear(x, m, b):
    return x * m + b


def fixed_linear(x, m):
    return x * m


def fit(f, x, y):
    try:
        coef, _ = curve_fit(f, x, y)
    except RuntimeError:
        sig = signature(f)
        len(sig.parameters)
        coef = [.5] * (len(sig.parameters) - 1)
    score = bic(y, f(x, *coef), len(coef))
    return coef, score


def get_intersections(interval_sets):
    interval_intersections = []
    for interval_set in interval_sets:
        intervals = interval_set[0]
        for interval in interval_set[1:]:
            intervals = np.intersect1d(intervals, interval)
        interval_intersections.append(intervals)
    return interval_intersections


def get_flex_bins(intervals_df, bin_time=.5):
    quanta = 100000
    intervals_df['length'] = intervals_df.interval_ends - intervals_df.interval_starts
    arr = np.zeros([len(intervals_df), math.ceil(intervals_df.length.max() * quanta)])
    for i, l in enumerate(intervals_df.length.values):
        arr[i, 0:int(l * quanta)] = 1
    arr_sum = np.cumsum(np.sum(arr, axis=0))
    max_val = math.ceil(max(arr_sum) / (quanta * bin_time)) * (quanta * bin_time)
    bounds = np.linspace((quanta * bin_time), max_val - (quanta * bin_time), int(max_val / (quanta * bin_time)) - 1)
    bins = (np.array([np.min(np.where(arr_sum > b)[0]) for b in bounds]) + 1) / quanta
    bins = np.concatenate([[0], bins, [len(arr_sum) / quanta]])
    return bins


def split_block(interval_ids, intervals_df):
    blocks = intervals_df.block.unique()
    blocks.sort()
    block_ids = get_block(interval_ids, intervals_df)
    return [np.unique(interval_ids[block_ids == block]).astype(int) for block in blocks]


def split_time(interval_ids, intervals_df):
    intervals_df['trial_time'] = intervals_df['interval_starts'] - intervals_df['trial_starts']
    return apply_quantile(intervals_df, 'trial_time')

    # bins = 3
    # edges = [np.quantile(intervals_df['trial_time'], q) for q in np.linspace(0, 1, bins + 1)]
    # edges[0] = -1.
    # return [np.array(intervals_df[(intervals_df['trial_time'] > s) & (intervals_df['trial_time'] <= e)].index) for s, e
    #         in zip(edges[:-1], edges[1:])]


def split_num_rewards(interval_ids, intervals_df):
    num_rewards = []
    for i, row in intervals_df.iterrows():
        trial = row.interval_trial
        start = row.interval_starts
        num_rewards.append(
            len(intervals_df[(intervals_df.interval_trial == trial) & (intervals_df.interval_starts < start)]))
    intervals_df['num_rewards'] = num_rewards
    return apply_quantile(intervals_df, 'num_rewards')
    # bins = 3
    # edges = [np.quantile(intervals_df['num_rewards'], q) for q in np.linspace(0, 1, bins + 1)]
    # edges[0] = -1.
    # return [np.array(intervals_df[(intervals_df['num_rewards'] > s) & (intervals_df['num_rewards'] <= e)].index)
    #         for s, e in zip(edges[:-1], edges[1:])]


def split_prev_interval(interval_ids, intervals_df):
    prev_interval = []
    for trial in intervals_df.interval_trial.unique():
        trial_df = intervals_df[intervals_df.interval_trial == trial]
        trial_df = trial_df.reset_index()
        for i, row in trial_df.iterrows():
            if i == 0:
                prev_interval.append(row.interval_starts - row.trial_starts)
            else:
                prev_interval.append(trial_df.loc[i - 1].length)

    intervals_df['prev_interval'] = prev_interval
    return apply_quantile(intervals_df, 'prev_interval')
    # bins = 3
    # edges = [np.quantile(intervals_df['prev_interval'], q) for q in np.linspace(0, 1, bins + 1)]
    # edges[0] = -1.
    # return [np.array(intervals_df[(intervals_df['prev_interval'] > s) & (intervals_df['prev_interval'] <= e)].index)
    #         for s, e in zip(edges[:-1], edges[1:])]


def split_leave(interval_ids, intervals_df):
    return apply_quantile(intervals_df, 'leave_times')[::-1]
    # bins = 3
    # edges = [np.quantile(intervals_df['leave_times'], q) for q in np.linspace(0, 1, bins + 1)]
    # edges[0] = -1.
    # return [np.array(intervals_df[(intervals_df['leave_times'] > s) & (intervals_df['leave_times'] <= e)].index) for
    #         s, e in zip(edges[:-1], edges[1:])][::-1]


def apply_quantile(df, column, bins=3):
    bins = 3
    edges = [np.quantile(df[column], q) for q in np.linspace(0, 1, bins + 1)]
    edges[0] = -1.
    return [np.array(df[(df[column] > s) & (df[column] <= e)].index) for s, e in zip(edges[:-1], edges[1:])]


def get_splits(interval_ids, intervals_df, splits=None):
    if splits is None:
        splits = ['block']
    intervals = []
    for s in splits:
        if s == 'block':
            intervals.append(split_block(interval_ids, intervals_df))
        if s == 'leave':
            intervals.append(split_leave(interval_ids, intervals_df))
        if s == 'time':
            intervals.append(split_time(interval_ids, intervals_df))
        if s == 'num_rewards':
            intervals.append(split_num_rewards(interval_ids, intervals_df))
        if s == 'prev_interval':
            intervals.append(split_prev_interval(interval_ids, intervals_df))
    return intervals, splits


def fit_plot(spikes, unit, interval_ids, interval_set, func_list, session, bin_size=100, plot=True):
    if unit == 10 and session == 'ES024_2023-02-14_bot192_1_g0':
        print()
        plot = True
    quant = .9
    color_sets = backend.get_color_sets()
    interval_spikes = [spikes[interval_ids == i] for i in interval_set]
    norm_func = np.mean if use_photometry_data else np.sum
    binned_interval_spikes = [norm_func(np.array(
        np.split(interval, np.arange(1, math.floor(len(interval) / bin_size) + 1) * bin_size)[:-1]).T, axis=0)
                              for interval in interval_spikes]
    binned_interval_spikes = [b if hasattr(b, '__len__') else np.array([]) for b in binned_interval_spikes]
    lengths = np.array([len(arr) for arr in binned_interval_spikes])
    max_x = math.floor(np.quantile(lengths, quant)) * bin_size / 1000
    flattened_spikes = np.array(backend.flatten_list(binned_interval_spikes))
    flattened_x = np.array(backend.flatten_list(
        [(np.linspace(bin_size, len(l) * bin_size, len(l)) - bin_size / 2) / 1000 for l in
         binned_interval_spikes]))
    flattened_spikes = flattened_spikes[flattened_x <= max_x]
    flattened_x = flattened_x[flattened_x <= max_x]
    if plot:
        for group in binned_interval_spikes:
            x = (np.linspace(bin_size, len(group) * bin_size, len(group)) - bin_size / 2) / 1000
            add_random = np.random.random() / 5 - .1 if not use_photometry_data else 0
            plt.plot(x, group + add_random, '.', c='k', alpha=.1, label='_nolegend_')
        pivot = np.array(list(itertools.zip_longest(*binned_interval_spikes, fillvalue=np.nan)))
        means = np.nanmean(pivot, axis=1)
        x = (np.linspace(bin_size, len(means) * bin_size, len(means)) - bin_size / 2) / 1000
        plt.plot(x, means, linewidth=3, c='k', label='mean')
        y_lim = plt.gca().get_ylim()
        y_range = y_lim[1] - y_lim[0]

    colors = color_sets['set1']
    scores = []
    centers = []
    lines = []
    v_lines = []
    coefs = []
    for i, func in enumerate(func_list):
        coef, score = fit(func, flattened_x, flattened_spikes)
        coef[0] = abs(coef[0])
        scores.append(score)
        centers.append(coef[0])
        coefs.append(coef)
        if plot:
            lines.append(plt.plot(x, func(x, *coef), linewidth=2, c=colors[i], alpha=.7,
                                  label=f'{func.__name__}: {score:.2f}'))
            p = func(coef[0], *coef)
            v_lines.append(
                plt.vlines(coef[0], p - y_range * .1, p + y_range * .1, color=colors[i], label='_nolegend_'))
    if plot:
        plt.xlim([-.5, 6])
        plt.ylim(y_lim)
        lines[np.argmin(scores)][0].set_alpha(1)
        lines[np.argmin(scores)][0].set_linewidth(3)
        v_lines[np.argmin(scores)].set_linewidth(3)
        plt.title(
            f'unit {unit} best fit: {func_list[np.argmin(scores)].__name__} regression '
            f'at {centers[np.argmin(scores)]:.2f} sec')
        ylabel = f'spikes per {bin_size} ms bin' if not use_photometry_data else 'df/f0'
        plt.ylabel(ylabel)
        plt.xlabel(f'time (sec)')
        plt.legend()
        # backend.save_fig(plt.gcf(), f'unit{unit}_all_fit', sub_folder=os.path.join('curve_fit', session))
        if np.argmin(scores) == 4 or np.argmin(scores) == 3:
            print()
        plt.show()

    return scores, coefs


def best_fit_plot(interval_sets, spikes, unit, interval_ids, f, coefs, session, split, interval_df,
                  bin_size=100, flex_bins=None, show_fig=True):
    # if (f.__name__ == 'a_gaussian_pos') or (f.__name__ == 'a_gaussian_neg'):
    show_fig = True
    color_sets = backend.get_color_sets()
    colors = color_sets['set2'] if split == 'block' else color_sets['set1']
    colors_dark = color_sets['set2_med_dark'] if split == 'block' else color_sets['set1_med_dark']
    flex_binned_split = []
    x_all_split = []
    y_all_split = []
    for i, intervals in enumerate(interval_sets):
        interval_spikes = [arr[unit] for arr in interval_df.loc[intervals].spikes.values]
        # interval_spikes = [spikes[interval_ids == i] for i in intervals]
        norm_func = np.mean if use_photometry_data else np.sum
        binned_interval_spikes = [norm_func(np.array(
            np.split(interval, np.arange(1, math.floor(len(interval) / bin_size) + 1) * bin_size)[:-1]).T, axis=0)
                                  for interval in interval_spikes]
        binned_interval_spikes = [b if hasattr(b, '__len__') else np.array([]) for b in binned_interval_spikes]
        flex_binned_spikes = [np.split(interval, flex_bins.astype(int)) for interval in interval_spikes]
        for j, a in enumerate(flex_binned_spikes):
            if len(a[-1]) > 0:
                print()

        # flex_bin_sizes = np.array([len(a) for a in flex_binned_spikes[0]])
        flex_binned_spikes = np.array([[np.sum(b) for b in a] for a in flex_binned_spikes])
        flex_bin_counts = norm_func(flex_binned_spikes, axis=0)[:-1]
        flex_binned_split.append(flex_bin_counts)
        x_all = []
        y_all = []
        for group in binned_interval_spikes:
            x = (np.linspace(bin_size, len(group) * bin_size, len(group)) - bin_size / 2) / 1000
            # add_random = np.random.random() / 5 - .1 if not use_photometry_data else 0
            # plt.plot(x, group + add_random, '.', c=colors[i], alpha=.1, label='_nolegend_')
            x_all.append(x)
            y_all.append(group)
        x_all = np.concatenate(x_all)
        y_all = np.concatenate(y_all)
        add_rand = np.random.random(size=len(y_all)) / 5 - .1 if not use_photometry_data else 0
        plt.plot(x_all, y_all + add_rand, '.', c=colors[i], alpha=.1, label='_nolegend_')
        x_all_split.append(x_all)
        y_all_split.append(y_all)
        pivot = np.array(list(itertools.zip_longest(*binned_interval_spikes, fillvalue=np.nan)))
        means = np.nanmean(pivot, axis=1)
        x = (np.linspace(bin_size, len(means) * bin_size, len(means)) - bin_size / 2) / 1000
        plot_gauss = False if use_photometry_data else True
        if plot_gauss:
            plt.plot(x, means, linewidth=3, c=colors[i], alpha=.7, label='_nolegend_')
            plt.plot(x, f(x, *coefs[i]), linewidth=2, c=colors_dark[i], alpha=1,
                     label=f'{split_labels[split][i]}: {coefs[i][0]:.2f} sec')
            p = f(coefs[i][0], *coefs[i])
            y_lim = plt.gca().get_ylim()
            y_range = y_lim[1] - y_lim[0]
            plt.vlines(coefs[i][0], p - y_range * .1, p + y_range * .1, color=colors_dark[i],
                       label='_nolegend_')
        else:
            plt.plot(x, means, linewidth=3, c=colors[i], alpha=1,
                     label=f'{split_labels[split][i]}: {coefs[i][0]:.2f} sec')

    p_vals = []
    for point in np.unique(np.concatenate(x_all_split)):
        arrays = []
        for i, arr in enumerate(y_all_split):
            arrays.append(arr[np.where(x_all_split[i] == point)[0]])
        p_vals_splits = []
        for (a, b) in itertools.combinations(arrays, 2):
            ttest = ttest_ind(a, b)
            p_vals_splits.append(ttest.pvalue)
        p_vals.append(p_vals_splits)
    p_vals = np.array(p_vals)
    p_vals[np.isnan(p_vals)] = 1
    sig_section = np.any([len(np.flatnonzero(np.convolve(p_vals[i] < .05, np.ones(3, dtype=int), 'valid') >= 3)) > 0
                          for i in range(p_vals.shape[1])])

    plt.xlim([-.5, 6])
    # spread_dif = abs(coefs[0][1] - coefs[1][1])
    spread_dif = np.max([abs(coefs[i][1] - coefs[i + 1][1]) for i in range(len(coefs) - 1)])
    height = np.min([abs(c[2]) for c in coefs])
    max_spread = np.max([c[1] for c in coefs])
    if use_photometry_data:
        height_filter = max_spread < (height * 70000 - 9)
    else:
        height_filter = max_spread < (height * 70 - 9)
    # plt.title(f'unit {unit} shift: {coefs[0][0] - coefs[1][0]:.2f} sec, spread: {spread_dif:.3f}, '
    #           f'height: {height:.2f}, max spread: {max_spread:.2f}')
    ylabel = f'spikes per {bin_size} ms bin' if not use_photometry_data else 'df/f0'
    plt.ylabel(ylabel)
    plt.xlabel(f'time (sec)')
    plt.legend()
    save_folder = 'photometry' if use_photometry_data else 'curve_fit'
    if use_photometry_data:
        side = 'right' if unit == 0 else 'left'
        plt.title(f'{session} {side}, split by: {split}')
        backend.save_fig(plt.gcf(), f'{session}_{side}_best_fit',
                         sub_folder=os.path.join(save_folder, split, 'units'))
    else:
        plt.title(f'unit {unit} shift: {coefs[0][0] - coefs[1][0]:.2f} sec, keep? {height_filter}')
        backend.save_fig(plt.gcf(), f'unit{unit}_best_fit',
                         sub_folder=os.path.join(save_folder, split, 'units', f'unit_{session}_shifts'))

    if show_fig:
        plt.show()
    else:
        plt.close()

    if height_filter:
        return np.array(flex_binned_split).T, sig_section
    else:
        return None, sig_section
    # res = input('good unit? (y/n) ')
    # global unit_data
    # unit_data.update([unit, height, f.__name__, spread_dif, max_spread, res == 'y'])


def compare_plot(spikes, unit, interval_ids, intervals_df, session, split='block', plot=True, plot_best=True,
                 flex_bins=None):
    intervals, splits = get_splits(interval_ids, intervals_df, splits=[split])
    func_list = [gaussian_pos, gaussian_neg, exp_gauss, a_gaussian_pos, a_gaussian_neg]
    score_list = []
    coef_list = []
    interval_sets = get_intersections(list(itertools.product(*intervals)))
    for interval_set in interval_sets:
        scores, coefs = fit_plot(spikes, unit, interval_ids, interval_set, func_list, session, plot=plot)
        score_list.append(scores)
        coef_list.append(coefs)
    best_fit = np.argmin(sum([np.array(l) for l in score_list]))
    best_fit_coefs = [c[best_fit] for c in coef_list]

    flex_bin_counts, sig = best_fit_plot(interval_sets, spikes, unit, interval_ids, func_list[best_fit],
                                         best_fit_coefs, session, split, intervals_df, flex_bins=flex_bins,
                                         show_fig=plot_best)
    return best_fit_coefs, func_list[best_fit], flex_bin_counts, sig


def curve_fitting(split, plot_fit=False, plot_best=False, photometry=False):
    timer.check()
    std_list = []
    folder = 'photometry' if photometry else 'curve_fit'
    session_data_path = os.path.join(os.getcwd(), 'figures', folder, split, 'session_data.pkl')

    session_data_columns = ['session', 'centers', 'unfiltered_centers', 'sig', 'unfiltered_sig']
    session_data = pd.DataFrame(columns=session_data_columns)
    color_sets = backend.get_color_sets()
    files = backend.get_session_list(photometry=photometry)
    points_all = [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []]]
    xy_range = [0, 5]
    heat_map_data = []
    heat_map_x = np.linspace(0, 10, 1000)
    for session in files:
        if session[:5] == 'ES024':
            continue
        # if session in disqualified():
        #     continue
        # if session != 'ES029_2022-09-14_bot72_0_g0':
        #     continue
        [_, _, _, original_spikes], interval_ids, intervals_df = create_precision_df(
            session, regenerate=False, photometry=photometry)
        _, _, cluster_info = backend.load_data(session, photometry=photometry)
        if cluster_info is None or 'area' in cluster_info.keys():
            area = 'Caudoputamen'
            original_spikes = original_spikes[np.where(cluster_info.area == area)[0]]

        if intervals_df is None:
            continue
        intervals_df['trial_starts'] = [intervals_df[(intervals_df.interval_trial == row[1].interval_trial) & (
                (intervals_df.interval_phase == 0) | (intervals_df.interval_phase == 3.))].interval_starts.iloc[0]
                                        for row in intervals_df.iterrows()]
        end_intervals = intervals_df[(intervals_df.interval_phase == 2) | (intervals_df.interval_phase == 3)]
        end_intervals['leave_times'] = end_intervals.interval_ends - end_intervals.interval_starts
        end_intervals = end_intervals.set_index('interval_trial')
        intervals_df['leave_times'] = end_intervals.loc[intervals_df.interval_trial].reset_index().leave_times
        intervals_df = intervals_df[(intervals_df.interval_phase == 1) | (intervals_df.interval_phase == 2)]
        original_spikes = original_spikes[:, np.in1d(interval_ids, intervals_df.index)]
        interval_ids = interval_ids[np.in1d(interval_ids, intervals_df.index)]
        spikes_to_use = original_spikes
        intervals_df['spikes'] = [spikes_to_use[:, interval_ids == i] for i in intervals_df.index]
        if photometry:
            std_list.append(np.std(spikes_to_use, axis=1))
            print(np.std(spikes_to_use, axis=1))
            spikes_to_use = spikes_to_use[np.std(spikes_to_use, axis=1) > .002]
        if spikes_to_use is None or len(spikes_to_use) == 0:
            continue
        flex_bin_origin = get_flex_bins(intervals_df, bin_time=30)
        flex_bin_centers = (flex_bin_origin[:-1] + flex_bin_origin[1:]) / 2
        flex_bins = np.round(flex_bin_origin[1:] * 1000)

        activity_levels = []
        coefs = []
        best_functions = []
        flex_bin_counts_list = []
        significant = []
        for unit in range(len(spikes_to_use)):
            spikes = spikes_to_use[unit]
            activity_levels.append(np.mean(spikes) * 1000)
            timer.check()
            if unit == 10 and session == 'ES024_2023-02-14_bot192_1_g0':
                print()
            best_coef, best_function, flex_bin_counts, sig = compare_plot(spikes, unit, interval_ids, intervals_df,
                                                                          session,
                                                                          plot=plot_fit, plot_best=plot_best,
                                                                          split=split,
                                                                          flex_bins=flex_bins)

            timer.check()
            coefs.append(best_coef)
            best_functions.append(best_function)
            flex_bin_counts_list.append(flex_bin_counts)
            significant.append(sig)
            if np.mean(spikes) * 1000 < .8:
                print()

        # global unit_data
        # unit_df = pd.read_pickle(unit_data.path)
        # gridx, gridy = np.mgrid[0:100, 0:100]

        for unit_data in flex_bin_counts_list:
            if unit_data is not None:
                # fig, ax = plt.subplots(1, 1, figsize=[7, 5])
                # ax.plot(flex_bin_centers, unit_data)
                # plt.show()
                heat_map_data.append(griddata(flex_bin_centers, unit_data, heat_map_x, method='nearest'))
                # plt.plot(np.linspace(0, flex_bin_origin[-1], 1000),griddata(flex_bin_centers, unit_data, (np.linspace(0, flex_bin_origin[-1], 1000)), method='nearest'))
                # plt.show()
        significant = np.array(significant)
        best_functions = np.array([f.__name__ for f in best_functions])
        centers = np.array([[c0[0] for c0 in c] for c in coefs])
        height = np.array([np.min([abs(c0[2]) for c0 in c]) for c in coefs])
        max_spread = np.array([np.max([c0[1] for c0 in c]) for c in coefs])
        if use_photometry_data:
            height_filter = max_spread < (height * 70000 - 9)
        else:
            height_filter = max_spread < (height * 70 - 9)
        filtered_centers = centers[height_filter & (np.all(abs(centers) < 5, axis=1))]
        filtered_significant = significant[height_filter & (np.all(abs(centers) < 5, axis=1))]
        session_data = session_data.append(
            pd.DataFrame([[session, filtered_centers, centers, filtered_significant, significant]],
                         columns=session_data_columns),
            ignore_index=True)
        if len(centers.T) == 2:
            sets = [[0, 1]]
        elif len(centers.T) == 3:
            sets = [[0, 1], [1, 2], [0, 2]]
        else:
            sets = [[i, i + 1] for i in range(len(centers.T) - 1)]
        for k, [i, j] in enumerate(sets):
            x, y = centers.T[i], centers.T[j]
            if use_photometry_data:
                height_filter = max_spread < (height * 70000 - 9)
            else:
                height_filter = max_spread < (height * 70 - 9)
            filter = (x < xy_range[1]) & (y < xy_range[1]) & (x > .01) & (y > .01) & height_filter
            # filter = (x < xy_range[1]) & (y < xy_range[1]) & (spread_dif < 25) & (height > .15) & (
            #         best_functions != 'exponential')

            if sum(filter) == 0:
                continue
            print(f'{session}: {np.sum(filter)}/{len(filter)} units kept')
            x, y = x[filter], y[filter]
            points_all[k][0].append(x)
            points_all[k][1].append(y)
            # x_all.append(x)
            # y_all.append(y)
            if len(x) > 75:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))

                ax.plot(xy_range, xy_range, c='k', label='no change', alpha=.5)
                coef, _ = curve_fit(fixed_linear, x, y)
                ax.plot(xy_range, fixed_linear(np.array(xy_range), *coef), label='regression', alpha=.5)
                ax.plot(x, y, '.', c=color_sets['set1'][1], label='centers')

                y_predicted = fixed_linear(x, *coef)
                ax.annotate("r-squared = {:.3f}".format(r2_score(y, y_predicted)), (3, .1))

                ax.set_xlim(xy_range)
                ax.set_ylim(xy_range)
                ax.set_xlabel(f'{split_labels[split][i]}\ncenter of field (sec)')
                ax.set_ylabel(f'{split_labels[split][j]}\ncenter of field (sec)')
                ax.legend()

                ax.set_title(f'Relative Field Shifts, {session}, {split}, {np.sum(filter)}/{len(filter)}')
                ax.set_aspect('equal', adjustable='box')
                plt.subplots_adjust(top=0.88)
                backend.save_fig(fig, session + '_shift_' + str(k), sub_folder=os.path.join(folder, split))
                plt.show()

    session_data.to_pickle(session_data_path)

    heat_map_data = np.array(heat_map_data)
    normalized = heat_map_data / np.max(heat_map_data, axis=1, keepdims=True)
    sorted_normalized = normalized[np.argsort(np.argmax(np.mean(normalized, axis=2), axis=1))]
    for i in range(sorted_normalized.shape[-1]):
        heat_map_split = sorted_normalized[:, :, i]
        # normalized = heat_map_split.T / np.max(heat_map_split, axis=1)
        # sorted_normalized = normalized[:, np.argsort(np.argmax(normalized, axis=0))]
        heatmap_df = pd.DataFrame(heat_map_split, columns=np.round(heat_map_x, 2),
                                  index=list(range(len(heat_map_split))))
        fig, ax = plt.subplots(1, 1, figsize=[7, 5])
        sns.heatmap(heatmap_df, ax=ax, xticklabels=100, yticklabels=False)
        ax.set_ylabel('Units')
        ax.set_xlabel('Seconds')
        ax.set_title(f'Heat Map: {split_labels[split][i]} ')
        backend.save_fig(fig, f'heat_map_{split}_{i}', sub_folder=os.path.join(folder, split))
        plt.show()

    # plt.hist(activity_levels, bins=2000)
    # plt.xlim([-.2, 2])
    # plt.show()

    for i in range(len(points_all)):
        if not len(points_all[i][0]):
            continue
        x_all = np.concatenate(points_all[i][0])
        y_all = np.concatenate(points_all[i][1])
        coef, _ = curve_fit(fixed_linear, x_all, y_all)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(x_all, y_all, '.', c=color_sets['set1'][1], label='centers', alpha=.5)
        ax.plot(xy_range, xy_range, c='k', label='no change')
        ax.plot(xy_range, fixed_linear(np.array(xy_range), *coef), label='regression')
        y_predicted = fixed_linear(x_all, *coef)
        ax.annotate("r-squared = {:.3f}".format(r2_score(y_all, y_predicted)), (3, .1))
        ax.set_xlim(xy_range)
        ax.set_ylim(xy_range)
        ax.set_xlabel(f'{split_labels[split][sets[i][0]]}\ncenter of field (sec)')
        ax.set_ylabel(f'{split_labels[split][sets[i][1]]}\ncenter of field (sec)')
        ax.legend()
        ax.set_title(f'Relative Field Shifts, All Units, {split}')
        ax.set_aspect('equal', adjustable='box')
        plt.subplots_adjust(top=0.88)

        backend.save_fig(fig, f'all_session_shifts_{i}', sub_folder=os.path.join(folder, split))
        plt.show()
    timer.check()

    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # ax.hist(np.concatenate(std_list), bins=100)
    # plt.show()


def unit_filtering():
    unit_data = UnitData()
    unit_df = unit_data.df
    functions = unit_df.function.unique()
    x = np.linspace(0, 4)
    # y = x * 70.175 - 9.12
    y = x * 70 - 9
    for f in functions:
        data = unit_df[unit_df.function == f]
        xy_pairs = [['height', 'max_spread']]
        # xy_pairs = [['height', 'max_spread'], ['height', 'spread_dif'], ['max_spread', 'spread_dif']]
        for [x_label, y_label] in xy_pairs:
            fig, ax = plt.subplots(1, 1)
            for i, row in data.iterrows():
                color = 'k' if row['filter'] else 'r'
                ax.plot(row[x_label], row[y_label], '.', color=color)
            ax.plot(x, y, 'k')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if x_label == 'height':
                ax.set_xlim([-.1, .4])
                # ax.set_xlim([-.5, 4])
            ax.set_ylim([-5, 100])
            ax.set_title(f'units for {f}')
            plt.show()


def bar_plots(split, photometry=False):
    color_sets = backend.get_color_sets()
    folder = 'photometry' if photometry else 'curve_fit'
    session_data_path = os.path.join(os.getcwd(), 'figures', folder, split, 'session_data.pkl')
    data = pd.read_pickle(session_data_path)
    data_filtered = data[[len(l) > 0 for l in data.centers]]
    # filtered_sig=[]
    # for i, row in data_filtered.iterrows():
    #     sig=np.array(row.sig)
    #     filtered_sig.append(sig[np.where(np.in1d(row.unfiltered_centers[:, 0],row.centers[:, 0]))[0]])
    # sig = np.concatenate(filtered_sig)
    centers = np.concatenate(data_filtered.centers.values)
    pass_filter = np.all(np.array([np.all(abs(centers) < 10, axis=1), np.all(abs(centers) > .001, axis=1)]), axis=0)
    centers = centers[pass_filter]
    # sig = sig[pass_filter]
    if split != 'leave':
        shifts = [centers.T[0] - centers.T[c] for c in range(1, len(centers.T))]
    else:
        shifts = [centers.T[-c - 1] - centers.T[-1] for c in range(1, len(centers.T))]
    avg_center = np.mean(centers, axis=1)
    shift_split_labels = {
        'block': ['low rate - high rate'],
        'leave': ['medium - early', 'late - early'],
        'time': ['early - medium', 'early - late'],
        'num_rewards': ['few - some', 'few - many'],
        'prev_interval': ['short - medium', 'short - long']
    }
    fig, ax = plt.subplots(1, 1, figsize=[7, 5])
    # legend = ['no shift']
    legend = []
    for i in range(len(shifts)):
        # sig_shifts=shifts[i][sig]
        # non_sig_shifts=shifts[i][~sig]
        # ax.hist([non_sig_shifts, sig_shifts], bins=100, alpha=.6, range=(-3, 3), stacked=True)
        ax.hist(shifts[i], bins=100, alpha=.6, range=(-3, 3), color=color_sets['set1'][i])
        legend.append(shift_split_labels[split][i])
        ax.vlines(np.mean(shifts[i]), *ax.get_ylim(), color=color_sets['set1'][i])
    ax.vlines(0, *ax.get_ylim(), 'k', linewidth=1, alpha=.5)

    ax.set_xlim([-3, 3])
    ax.set_ylabel('Count')
    ax.set_xlabel('Center Shifts (sec)')
    ax.legend(legend)
    ax.set_title(f'Shifts for Split: {split}')
    backend.save_fig(fig, f'shifts_hist', sub_folder=os.path.join(folder, split))
    plt.show()

    shifts = np.array(shifts)
    fig, ax = plt.subplots(1, 1, figsize=[7, 5])
    sort_i = np.argsort(avg_center)
    x = avg_center[sort_i]
    y = shifts[:, sort_i]
    y = np.array([np.mean(arr, axis=1) for arr in np.array_split(y, min([len(x), 100]), axis=1)])[:-2]
    x = np.array([np.mean(arr) for arr in np.array_split(x, min([len(x), 100]))])[:-2]

    # x = np.round(avg_center * 10) / 10
    # x_vals = np.unique(x)
    # y = np.array([np.mean(shifts[:, np.where(x == x_i)[0]], axis=1) for x_i in x_vals])
    # ax.plot(x_vals, y)
    xy_range = [0, 5]
    for i in range(len(y.T)):
        ax.scatter(x, y[:, i], c=color_sets['set1'][i], label=shift_split_labels[split][i])
        coef, _ = curve_fit(linear, x, y[:, i])
        if split != 'time':
            ax.plot(xy_range, linear(np.array(xy_range), *coef), alpha=.5, c=color_sets['set1'][i])
            ax.annotate("r-squared = {:.3f}".format(r2_score(y[:, i], linear(x, *coef))), (3.5, .1 * (i + .5)))

    # ax.set_ylim([-.1, 1])
    ax.set_xlim(xy_range)
    ax.set_ylabel('Average Shift (sec)')
    ax.set_xlabel('Time From Reward (sec)')
    ax.set_title(f'Shifts by Time from Reward: {split}')
    ax.legend()
    backend.save_fig(fig, f'shifts_regression', sub_folder=os.path.join(folder, split))
    plt.show()

    # fig, ax = plt.subplots(1, 1, figsize=[7, 5])
    # ax.hist(centers.T[0], alpha=.5, bins=50, range=(0, 10))
    # ax.hist(centers.T[1], alpha=.5, bins=50, range=(0, 10))
    # # ax.set_xlim([-3, 3])
    # ax.set_ylabel('Count')
    # ax.set_xlabel('Field Centers (sec)')
    # plt.show()

    # fig, ax = plt.subplots(1, 1, figsize=[4, 10])
    # ax.plot(centers.T, alpha=.5)
    # # ax.set_xlim([-3, 3])
    # ax.set_xlabel('Group')
    # ax.set_ylabel('Field Centers (sec)')
    # plt.show()
    print()


use_photometry_data = False
timer = Timer()

if __name__ == '__main__':
    # unit_filtering()
    # unit_data = UnitData()
    split_labels = {
        'block': ['low rate block', 'high rate block'],
        'leave': ['late leave', 'medium leave', 'early leave'],
        'time': ['early in trial', 'middle of trial', 'late in trial'],
        'prev_interval': ['short', 'medium', 'long'],
        'num_rewards': ['few rewards', 'some rewards', 'several rewards'],
    }
    for key in split_labels.keys():
        timer.check()
        # curve_fitting(key, photometry=use_photometry_data)
        timer.check()
        if not use_photometry_data:
            timer.check()
            bar_plots(key, photometry=use_photometry_data)
            timer.check()
