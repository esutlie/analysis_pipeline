"""This file is for processing the neural spike times into rates with different levels of smoothing and normalization"""

import pandas as pd
import backend
from behavior import get_trial_events
import numpy as np
import math
from scipy.ndimage import convolve1d
import os
from scipy.interpolate import griddata


def get_trial_group(events):
    pi_events_shortened = events[(events.phase != 'setup') & ~np.isnan(events.trial)]
    phases = pi_events_shortened.phase.values
    transition_trials = pi_events_shortened.iloc[np.where(phases != np.roll(phases, 1))[0]].trial.values
    block_idx = [np.sum(trial >= transition_trials) for trial in np.unique(pi_events_shortened.trial)]
    return pd.Series(block_idx, index=np.unique(pi_events_shortened.trial))


def create_precision_df(session, kernel_size=1000, regenerate=False, photometry=False):
    local_path = os.path.join(backend.get_data_path(photometry=photometry), session)
    normalized_spikes_path = os.path.join(local_path, 'normalized_spikes.npy')
    convolved_spikes_path = os.path.join(local_path, 'convolved_spikes.npy')
    boxcar_spikes_path = os.path.join(local_path, 'boxcar_spikes.npy')
    original_spikes_path = os.path.join(local_path, 'original_spikes.npy')
    interval_ids_path = os.path.join(local_path, 'interval_ids.npy')
    intervals_df_path = os.path.join(local_path, 'intervals_df.pkl')
    paths = [normalized_spikes_path, convolved_spikes_path, boxcar_spikes_path, original_spikes_path, interval_ids_path,
             intervals_df_path]
    if np.all([os.path.exists(path) for path in paths]) and not regenerate:
        normalized_spikes = np.load(normalized_spikes_path)
        convolved_spikes = np.load(convolved_spikes_path)
        boxcar_spikes = np.load(boxcar_spikes_path)
        original_spikes = np.load(original_spikes_path)
        interval_ids = np.load(interval_ids_path)
        intervals_df = pd.read_pickle(intervals_df_path)
        return [normalized_spikes, convolved_spikes, boxcar_spikes, original_spikes], interval_ids, intervals_df
    kernel = get_gaussian_kernel(l=kernel_size, sigma=kernel_size / 5)
    boxcar_kernel = get_boxcar_kernel(w=100)
    # plt.plot(kernel)
    # plt.show()
    spikes, pi_events, cluster_info = backend.load_data(session, photometry=photometry)

    if len(spikes) == 0 or np.isnan(pi_events.trial.max()):
        return [None, None, None, None], None, None
    trial_blocks = pi_events.groupby('trial').phase.max()
    trial_group = get_trial_group(pi_events)
    interval_columns = ['interval_starts', 'interval_ends', 'interval_trial', 'interval_phase']
    intervals_df = pd.DataFrame(columns=interval_columns)
    for entry_time, exit_time, reward_list, trial, _ in zip(*get_trial_events(pi_events)):
        interval_starts = [entry_time] + reward_list.tolist()
        interval_ends = reward_list.tolist() + [exit_time]
        interval_trial = [trial] * len(interval_starts)
        interval_phase = [0] + [1] * (len(reward_list) - 1) + [2] if len(reward_list) else [3]
        interval_data = np.array([interval_starts, interval_ends, interval_trial, interval_phase]).T
        intervals_df = intervals_df.append(pd.DataFrame(interval_data, columns=interval_columns))
    intervals_df.reset_index(inplace=True, drop=True)
    intervals_df['block'] = trial_blocks.loc[intervals_df.interval_trial.values].values
    intervals_df['group'] = trial_group.loc[intervals_df.interval_trial.values].values
    filtered_interval_arrays = []
    boxcar_interval_arrays = []
    original_interval_arrays = []
    interval_ids = []
    interval_rates = []
    for interval_idx, row in enumerate(intervals_df.values):
        [start, end, trial, phase, block, group] = row
        interval_spikes = spikes[(spikes.time > start) & (spikes.time < end)]
        if photometry:
            sensor_vals = interval_spikes[['green_right', 'green_left']].to_numpy()
            x_origin = interval_spikes.time.values - start
            if len(x_origin):
                x_len = math.ceil(x_origin.max() * 1000)
                x_map = np.linspace(0, x_len / 1000, x_len + 1)
                if len(x_origin) > 1:
                    spike_rates = griddata(x_origin, sensor_vals, x_map, method='nearest').T
                else:
                    spike_rates = np.repeat(sensor_vals, [len(x_map)], axis=0).T
            else:
                spike_rates = np.array([[np.nan], [np.nan]])
            interval_rates.append(np.mean(spike_rates, axis=1) / (end - start))
        else:
            clusters = np.unique(spikes.cluster)
            spike_times = interval_spikes.time.values
            bins = ((spike_times - start).round(decimals=3) * 1000).astype(int)
            interval_spikes['bin'] = bins
            spike_rates = np.zeros([len(clusters), 1 + math.ceil((end - start) * 1000)])
            for i, cluster in enumerate(clusters):
                counts = interval_spikes[interval_spikes.cluster == cluster].groupby('bin').count().cluster
                for index in counts.index:
                    spike_rates[i, index] = counts.loc[index]
            interval_rates.append(np.sum(spike_rates, axis=1) / (end - start))
        filtered_interval_arrays.append(convolve1d(spike_rates, kernel))
        boxcar_interval_arrays.append(convolve1d(spike_rates, boxcar_kernel))
        original_interval_arrays.append(spike_rates)
        interval_ids.append(interval_idx * np.ones([len(spike_rates[0])]))
    convolved_spikes = np.concatenate(filtered_interval_arrays, axis=1)
    boxcar_spikes = np.concatenate(boxcar_interval_arrays, axis=1)
    original_spikes = np.concatenate(original_interval_arrays, axis=1)
    interval_ids = np.concatenate(interval_ids, axis=0)
    centered_spikes = np.subtract(convolved_spikes,
                                  np.expand_dims(np.mean(convolved_spikes, axis=1), axis=1))
    normalized_spikes = np.divide(centered_spikes, np.expand_dims(np.std(centered_spikes, axis=1), axis=1))
    intervals_df['rate'] = interval_rates
    np.save(normalized_spikes_path, normalized_spikes)
    np.save(convolved_spikes_path, convolved_spikes)
    np.save(boxcar_spikes_path, boxcar_spikes)
    np.save(original_spikes_path, original_spikes)
    np.save(interval_ids_path, interval_ids)
    intervals_df.to_pickle(intervals_df_path)
    return [normalized_spikes, convolved_spikes, boxcar_spikes, original_spikes], interval_ids, intervals_df


def get_phase(interval_ids, intervals_df):
    x_total = []
    for i in np.unique(interval_ids):
        num = len(np.where(interval_ids == i)[0])
        phase = intervals_df.loc[i].interval_phase
        x_total.append([phase] * num)
    phase = np.concatenate(x_total)
    return phase


def get_block(interval_ids, intervals_df):
    x_total = []
    for i in np.unique(interval_ids):
        num = len(np.where(interval_ids == i)[0])
        block = intervals_df.loc[i].block
        x_total.append([block] * num)
    block = np.concatenate(x_total)
    return block


def get_x(interval_ids, min_longest=1):
    x_total = []
    for i in np.unique(interval_ids):
        num = len(np.where(interval_ids == i)[0])
        x_total.append(np.linspace(0, (num - 1) / 1000, num))
    longest = np.argsort(np.array([len(x_i) for x_i in x_total]), axis=0)[-min_longest]
    # longest = np.argmax(np.array([len(x_i) for x_i in x_total]))
    longest = x_total[longest]
    x = np.concatenate(x_total)
    x = np.around(x, 3)
    return x, longest


def create_bins_df(session):
    spikes, pi_events, cluster_info = backend.load_data(session)
    if len(spikes) == 0:
        return False
    spikes['bins'] = spikes.time.round(decimals=1)
    clusters = spikes.cluster.unique()
    clusters.sort()
    spike_rates = np.zeros([len(clusters), 1 + int(spikes.bins.max() * 10)])
    for i, cluster in enumerate(clusters):
        counts = spikes[spikes.cluster == cluster].groupby('bins').count().cluster
        for index in counts.index:
            spike_rates[i, int(index * 10)] = counts.loc[index]
    spike_rates_list = np.split(spike_rates.T, len(spike_rates.T))
    spike_rates_list = [[np.squeeze(l)] for l in spike_rates_list]
    session_df = pd.DataFrame(spike_rates_list, columns=['spike_rates'])
    entries, exits, rewards, trial_numbers = get_trial_events(pi_events)
    trial_blocks = pi_events.groupby('trial').phase.max()
    all_rewards = np.concatenate(rewards)
    session_df = session_df[session_df.index / 10 > min(entries)]
    session_df = session_df[session_df.index / 10 < max(exits)]
    session_df['closest_entry'] = np.array(entries)[
        backend.min_dif(entries, session_df.index / 10, rev=True, return_index=True)[0]]
    session_df['closest_exit'] = np.array(exits)[
        backend.min_dif(session_df.index / 10, exits, rev=False, return_index=True)[0]]
    session_df['closest_entry_trial'] = np.array(trial_numbers)[
        backend.min_dif(entries, session_df.index / 10, rev=True, return_index=True)[0]]
    session_df['closest_exit_trial'] = np.array(trial_numbers)[
        backend.min_dif(session_df.index / 10, exits, rev=False, return_index=True)[0]]
    session_df = session_df[session_df.closest_entry_trial == session_df.closest_exit_trial]
    session_df['time_from_entry'] = backend.min_dif(entries, session_df.index / 10, rev=True)
    session_df['time_from_reward'] = backend.min_dif(np.concatenate([all_rewards, np.array(entries)]),
                                                     session_df.index / 10, rev=True)
    session_df['time_before_exit'] = backend.min_dif(session_df.index / 10, exits, rev=False)
    session_df['mouse'] = [session[:5]] * len(session_df)
    session_df['session'] = [session] * len(session_df)
    session_df['time_in_session'] = session_df.index / 10
    session_df['trial'] = session_df['closest_entry_trial']
    session_df['block'] = trial_blocks.loc[session_df.trial].values
    return session_df


def get_gaussian_kernel(l=5, sigma=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    # plt.plot(gauss)
    # plt.show()
    return gauss / np.sum(gauss)


def get_boxcar_kernel(w=10):
    return np.ones(w)


# def test_kernel():
#     i = 12
#     plt.plot(spike_rates[i] / 20)
#     plt.plot(convolve1d(spike_rates, get_gaussian_kernel(l=50, sigma=50 / 5))[i])
#     plt.plot(convolve1d(spike_rates, get_gaussian_kernel(l=100, sigma=100 / 5))[i])
#     plt.plot(convolve1d(spike_rates, get_gaussian_kernel(l=300, sigma=300 / 5))[i])
#     plt.show()

def regen_all():
    files = backend.get_session_list()
    for session in files:
        [_, _, _, _], interval_ids, intervals_df = create_precision_df(session, regenerate=True)

if __name__ == '__main__':
    # first_session = backend.get_session_list()[0]
    # create_bins_df(first_session)
    # create_precision_df(first_session)
    regen_all()

