import pandas as pd
import backend
from behavior import get_trial_events
import numpy as np


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
    create_bins_df(first_session)
