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
from itertools import repeat
import pandas as pd


def get_binned_quantiles(session, verbose=False, regenerate=False):
    local_path = os.path.join(backend.get_data_path(), session)
    binned_counts_path = os.path.join(local_path, 'binned_counts.npy')
    binned_x_path = os.path.join(local_path, 'binned_x.npy')
    binned_quantiles_path = os.path.join(local_path, 'binned_quantiles.npy')
    binned_intervals_path = os.path.join(local_path, 'binned_intervals.npy')
    paths = [binned_x_path, binned_counts_path, binned_quantiles_path, binned_intervals_path]
    if np.all([os.path.exists(path) for path in paths]) and not regenerate:
        return np.load(binned_counts_path), np.load(binned_x_path), np.load(binned_quantiles_path), np.load(
            binned_intervals_path)

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
    binned_intervals = []
    for unit in range(num_units):
        bin_counts = []
        for trial in np.unique(interval_ids):
            num_groups = math.floor(intervals_df.loc[trial].length / bin_size)
            if num_groups > 0:
                trial_spikes = original_spikes[unit, (interval_ids == trial)][:round(num_groups * bin_size * 1000)]
                groups = np.sum(np.stack(np.array_split(trial_spikes, num_groups)), axis=1)
                bin_counts.append(groups.astype(int))
                if unit == 0:
                    binned_intervals.append([trial] * num_groups)
        all_counts = np.hstack(bin_counts)
        non_zero_counts = all_counts[all_counts != 0]
        quants = [np.quantile(non_zero_counts, q) for q in np.linspace(0, 1 - 1 / num_quantiles, num_quantiles)]

        unit_quants = [[sum(val > np.array(quants)) for val in trial_count] for trial_count in bin_counts]
        binned_quantiles.append(backend.flatten_list(unit_quants))
        if not len(binned_x):
            unit_x = [list(range(len(trial_count))) for trial_count in bin_counts]
            binned_x = backend.flatten_list(unit_x)
            binned_intervals = backend.flatten_list(binned_intervals)

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
    np.save(binned_intervals_path, np.array(binned_intervals))
    return binned_counts, binned_x, binned_quantiles, binned_intervals


def get_bayes(binned_quantiles):
    p_fr = np.sum(binned_quantiles, axis=1) / np.sum(binned_quantiles[0])
    bayes = binned_quantiles / np.sum(binned_quantiles[0]) / p_fr[:, None, :]
    bayes[np.isnan(bayes)] = 0
    return bayes


def predict(bayes, quantiles, w):
    return np.sum(bayes[np.arange(len(bayes)), :, quantiles] * w[:, None], axis=0) / sum(w)


def center_of_mass(pred):
    return np.sum(np.arange(pred.shape[1]) * pred, axis=1)


def score_model(x, y, plot=False, log=True):
    if log:
        x = np.log(x.reshape(-1, 1) + 1)
    else:
        x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    score = model.score(x, y)
    model_pred = model.predict(x)
    if plot:
        if log:
            plt.scatter(np.exp(x) - 1, y)
            plt.scatter(np.exp(x) - 1, model_pred)
        else:
            plt.scatter(x, y)
            plt.scatter(x, model_pred)
        plt.show()
    return score


def prediction_model(x, y, plot=True):
    y = np.log(y + 1)
    x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    score = model.score(x, y)
    model_pred = model.predict(x)
    if plot:
        plt.scatter(x, np.exp(y) - 1)
        plt.scatter(x, np.exp(model_pred) - 1)
        plt.ylabel('time (seconds)')
        plt.xlabel('bayesian prediction variable')
        plt.show()
    return np.exp(model_pred) - 1


def score_by_local(x):
    top = np.argsort(x)[-3:] if np.argmax(x) != 0 else np.argsort(x)[-2:]
    return sum(top * x[top]) / sum(x[top])


def score_by_accuracy(pred, binned_x):
    main_score = pred[np.arange(len(pred)), binned_x]
    left_score = pred[np.arange(len(pred)), binned_x - 1]
    right_score = pred[np.arange(len(pred)), binned_x + 1]
    score = main_score + .3 * right_score + .3 * left_score
    return np.mean(score)


def fit_by_random(bayes, binned_counts, binned_x, binned_quantiles):
    scores = []
    weights = []
    for i in range(50):
        w = np.random.random([binned_counts.shape[0]])
        pred = np.array(list(map(predict, repeat(bayes), binned_quantiles.T, repeat(w))))
        com = center_of_mass(pred)
        score = score_model(binned_x, com, plot=False, log=True)
        scores.append(score)
        weights.append(w)
    score = np.max(scores)
    print(f'starting score: {score}')
    score_memory = score
    w = weights[np.argmax(scores)]
    for i in range(10000):
        history = []
        deltas = [.5, 1, 2]
        for delta in deltas:
            temp_weights = w.copy()
            temp_weights[i % len(w)] *= delta
            temp_weights /= np.max(temp_weights)
            pred = np.array(list(map(predict, repeat(bayes), binned_quantiles.T, repeat(temp_weights))))
            # score = score_by_accuracy(pred, binned_x)
            com = center_of_mass(pred)
            score = score_model(binned_x, com, plot=False, log=True)
            history.append(score)
        w[i % len(w)] *= deltas[np.argmax(history)]
        w /= np.max(w)
        print(f'{i} score: {np.max(history)}     {deltas[np.argmax(history)]}')
        if i % len(bayes) == 0:
            np.save(os.path.join(os.getcwd(), 'weights2.npy'), w)
            # plt.plot(w)
            # plt.show()
            print('score improved by: ' + str(abs(score - score_memory)))
            if abs(score - score_memory) < .001 and i != 0:
                break
            score_memory = score
    return w


def fit_by_exclude(bayes, binned_counts, binned_x, binned_quantiles):  # Not done
    w = np.ones([binned_counts.shape[0]])
    pred = np.array(list(map(predict, repeat(bayes), binned_quantiles.T, repeat(w))))
    com = center_of_mass(pred)
    score = score_model(binned_x, com, plot=True, log=True)
    print(f'starting error: {score}')
    scores = []
    for i in range(len(w)):
        weights = w.copy()
        weights[i] = 0
        pred = np.array(list(map(predict, repeat(bayes), binned_quantiles.T, repeat(w))))
        com = center_of_mass(pred)
        score = score_model(binned_x, com, plot=True, log=True)
        scores.append(score)
        print(score)


def compare(session, binned_pred, binned_x, binned_intervals):
    color_sets = backend.get_color_sets()
    local_path = os.path.join(backend.get_data_path(), session)
    intervals_df_path = os.path.join(local_path, 'intervals_df.pkl')
    intervals_df = pd.read_pickle(intervals_df_path)
    blocks = intervals_df.block.unique()
    blocks.sort()
    binned_blocks = get_block(binned_intervals, intervals_df)
    block_means = []
    for i, b in enumerate(blocks):
        b_filter = b == binned_blocks
        plt.scatter(binned_x[b_filter] * .3 + np.random.random(np.sum(b_filter)) * .1, binned_pred[b_filter] * .3,
                    c=color_sets['set2'][i], s=1)
        means = []
        for j, bin_num in enumerate(np.unique(binned_x)):
            bin_filter = binned_x == bin_num
            means.append(np.mean(binned_pred[b_filter & bin_filter] * .3))
        block_means.append(means)
        plt.plot(np.unique(binned_x) * .3, means, c=color_sets['set2'][i])

    plt.xlabel('Real Time Bins')
    plt.ylabel('Predicted Time Bins')
    plt.show()

    print()


def main():
    color_sets = backend.get_color_sets()
    files = backend.get_session_list()
    for session in files:
        if session != 'ES029_2022-09-14_bot72_0_g0':
            continue
        """I think the problem is that I've lost too much sensitivity in the blocking. Find a way to do this with the 
        continuous values and it might work."""
        binned_counts, binned_x, binned_quantiles, binned_intervals = get_binned_quantiles(session)
        non_zeros = binned_x != 0
        binned_x = binned_x[non_zeros]
        binned_quantiles = binned_quantiles[:, non_zeros]
        binned_intervals = binned_intervals[non_zeros]

        bayes = get_bayes(binned_counts)
        # w = fit_by_random(bayes, binned_counts, binned_x, binned_quantiles)

        w = np.load(os.path.join(os.getcwd(), 'weights2.npy'))
        plt.plot(w)
        plt.show()
        pred = np.array(list(map(predict, repeat(bayes), binned_quantiles.T, repeat(w))))
        com = center_of_mass(pred)
        score = score_model(binned_x, com, plot=True, log=True)

        # coms = []
        # for b in np.unique(binned_x):
        #     b_filter = b == binned_x
        #     # plt.plot(pred[b_filter].T)
        #     # plt.title(f'bin {b}')
        #     # plt.show()
        #     plt.plot(np.mean(pred[b_filter], axis=0))
        #     plt.title(f'bin {b}')
        #     plt.show()
        #     coms.append(center_of_mass(np.mean(pred[b_filter], axis=0)[None, :]))
        # plt.plot(coms)
        # plt.show()

        binned_pred = prediction_model(com, binned_x)
        compare(session, binned_pred, binned_x, binned_intervals)


if __name__ == '__main__':
    main()
