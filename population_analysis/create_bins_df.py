import pandas as pd
import backend
from behavior import get_trial_events
import numpy as np
import math
from scipy.ndimage import convolve1d


def get_trial_group(events):
    pi_events_shortened = events[(events.phase != 'setup') & ~np.isnan(events.trial)]
    phases = pi_events_shortened.phase.values
    transition_trials = pi_events_shortened.iloc[np.where(phases != np.roll(phases, 1))[0]].trial.values
    block_idx = [np.sum(trial >= transition_trials) for trial in np.unique(pi_events_shortened.trial)]
    return pd.Series(block_idx, index=np.unique(pi_events_shortened.trial))


def create_precision_df(session, kernel_size=300):
    kernel = np.ones([kernel_size])
    spikes, pi_events, cluster_info = backend.load_data(session)
    if len(spikes) == 0:
        return False
    clusters = np.unique(spikes.cluster)
    trial_blocks = pi_events.groupby('trial').phase.max()
    trial_group = get_trial_group(pi_events)
    interval_columns = ['interval_starts', 'interval_ends', 'interval_trial', 'interval_phase']
    intervals_df = pd.DataFrame(columns=interval_columns)
    for entry_time, exit_time, reward_list, trial, in zip(*get_trial_events(pi_events)):
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
    interval_ids = []
    for interval_idx, row in enumerate(intervals_df.values):
        [start, end, trial, phase, block, group] = row
        interval_spikes = spikes[(spikes.time > start) & (spikes.time < end)]
        interval_spikes['bin'] = ((interval_spikes.time.values - start).round(decimals=3) * 1000).astype(int)
        spike_rates = np.zeros([len(clusters), 1 + math.ceil((end - start) * 1000)])
        for i, cluster in enumerate(clusters):
            counts = interval_spikes[interval_spikes.cluster == cluster].groupby('bin').count().cluster
            for index in counts.index:
                spike_rates[i, index] = counts.loc[index]
        filtered_interval_arrays.append(convolve1d(spike_rates, kernel))
        interval_ids.append(interval_idx * np.ones([len(spike_rates[0])]))
    filtered_interval_arrays = np.concatenate(filtered_interval_arrays, axis=1)
    interval_ids = np.concatenate(interval_ids, axis=0)
    centered_spikes = np.subtract(filtered_interval_arrays, np.expand_dims(np.mean(filtered_interval_arrays, axis=1), axis=1))
    normalized_spikes = np.divide(centered_spikes, np.expand_dims(np.std(centered_spikes, axis=1), axis=1))
    return normalized_spikes, interval_ids, intervals_df


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


if __name__ == '__main__':
    first_session = backend.get_session_list()[0]
    # create_bins_df(first_session)
    create_precision_df(first_session)
