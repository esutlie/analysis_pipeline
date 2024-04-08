"""
Step 1:
Create a master data frame with all of the neural and behavioral data across sessions.
Maybe separate sessions across different files if they take a long time to load and combine the ones you want
Save it so we don't have to reprocess all the time
"""

from step1_calc_functions import *
import pandas as pd
import os
import matplotlib.pyplot as plt


def recent_rate_kernel(x):
    return np.exp(-x / 30)


def optimal_kernel(x, single=False):
    if not single:
        return -8 * np.log(x)
    else:
        reward_level = 0.5994974874371859  # cumulative for an 8 reward version
        starting_prob = 0.1301005025125628
        return np.log(((1 / x) - reward_level) / (1 - reward_level)) * (reward_level / starting_prob)


def exp_func(x, single=False):
    if not single:
        return np.exp(-x / 8)
    else:
        reward_level = 0.5994974874371859  # cumulative for an 8 reward version
        starting_prob = 0.1301005025125628
        return 1 / ((1 - reward_level) * np.exp(starting_prob / reward_level * x) + reward_level)


def make_intervals_df(pi_events):
    all_reward_times = pi_events[(pi_events.key == 'reward') & (pi_events.value == 1)].time.values
    single = bool(len(pi_events[pi_events.key == 'reward_time']))
    recent_rewards = []
    recent_rates = []
    recent_rate_t = np.sum(recent_rate_kernel(np.linspace(.5, 59.5, 60)))
    interval_columns = ['start', 'end', 'trial', 'phase', 'trial_time', 'leave_from_reward', 'leave_from_entry',
                        'optimal_leave']
    intervals_df = pd.DataFrame(columns=interval_columns)
    for entry_time, exit_time, reward_list, trial, _ in zip(*backend.get_trial_events(pi_events)):
        all_interval_starts = np.concatenate([[entry_time], reward_list])
        interval_starts = [entry_time] + reward_list.tolist()
        interval_ends = reward_list.tolist() + [exit_time]
        interval_trial = [trial] * len(interval_starts)
        interval_phase = [0] + [1] * (len(reward_list) - 1) + [2] if len(reward_list) else [3]
        trial_time = all_interval_starts - entry_time
        leave_from_reward = [exit_time - reward_list[-1]] * len(interval_starts) if len(reward_list) else [np.nan]
        leave_from_entry = [exit_time - entry_time] * len(interval_starts)
        rewards = entry_time - all_reward_times
        recent_rewards_arr = np.array([r for r in rewards if 0 < r < 60])
        recent_rate = np.sum(recent_rate_kernel(recent_rewards_arr)) / recent_rate_t
        optimal_leave = optimal_kernel(recent_rate, single=single)
        for t in all_interval_starts:
            rewards = t - all_reward_times
            recent_rewards_arr = np.array([r for r in rewards if 0 < r < 60])
            recent_rewards.append(recent_rewards_arr)
            recent_rates.append(np.sum(recent_rate_kernel(recent_rewards_arr)) / recent_rate_t)
        interval_data = np.array(
            [interval_starts, interval_ends, interval_trial, interval_phase, trial_time,
             leave_from_reward, leave_from_entry, optimal_leave]).T
        # if len(interval_data.shape) == 1:
        #     interval_data = np.expand_dims(interval_data, 0)
        # else:
        #     interval_data = interval_data.T
        intervals_df = pd.concat([intervals_df, pd.DataFrame(interval_data, columns=interval_columns)])
        # intervals_df = intervals_df.append(pd.DataFrame(interval_data, columns=interval_columns))
    intervals_df.reset_index(inplace=True, drop=True)

    trial_blocks = pi_events.groupby('trial').phase.max()
    intervals_df['block'] = trial_blocks.loc[intervals_df.trial.values].values
    trial_group = backend.get_trial_group(pi_events)
    intervals_df['group'] = trial_group.loc[intervals_df.trial.values].values
    intervals_df['recent_rewards'] = recent_rewards
    intervals_df['recent_rate'] = recent_rates
    return intervals_df


def make_unit_df(cluster_info, global_start_index):
    global_start_index = global_start_index if not np.isnan(global_start_index) else 0
    cluster_info['global_id'] = list(range(global_start_index, global_start_index + len(cluster_info)))
    return cluster_info


def make_session_df(intervals_df, unit_df, spikes):
    # Get session and global ids for the units
    unit_session_ids = unit_df.id.to_numpy()
    unit_global_ids = unit_df.global_id.to_numpy()
    intervals_df['unit_session_ids'] = [unit_session_ids] * len(intervals_df)
    intervals_df['unit_global_ids'] = [unit_global_ids] * len(intervals_df)

    # Get the lists of spike times for each interval for each spike
    intervals_df['spike_times'] = get_spike_times(intervals_df, spikes)

    # Get the center of mass for all the units
    intervals_df['bins500ms'] = bins500ms(intervals_df)

    return intervals_df


def get_master_df(regenerate=False, add=True):
    # Load master_df and return if not adding anything. If it doesn't exist yet, make an empty one.
    master_data_path = os.path.join(os.getcwd(), 'data', 'master_data.pkl')
    unit_global_path = os.path.join(os.getcwd(), 'data', 'unit_global.pkl')
    if os.path.exists(master_data_path) and not regenerate:
        master_df = pd.read_pickle(master_data_path)
    else:
        master_df = pd.DataFrame()
    if os.path.exists(unit_global_path) and not regenerate:
        unit_global_df = pd.read_pickle(unit_global_path)
    else:
        unit_global_df = pd.DataFrame()
    if not regenerate and not add:
        return master_df, unit_global_df

    # If we are regenerating or adding sessions, get a list of all the sessions and iterate through them
    t = backend.Timer()
    session_list = ['ES041_2024-03-08_bot096_1_g0']
    # session_list = ['ES039_2024-03-08_bot144_1_g0']
    # session_list = backend.get_session_list()
    # if 'session' in master_df.keys():
    #     session_list = [session for session in session_list if session not in master_df['session']]

    for session in session_list:
        spikes, pi_events, cluster_info = backend.load_data(session, photometry=False)
        if len(spikes) == 0 or np.isnan(pi_events.trial.max()):
            print(f'session: {session} is empty or broken')
            continue

        t.tic('load_data')
        # Construct the intervals df for the session
        intervals_df = make_intervals_df(pi_events)
        intervals_df = add_session_columns(intervals_df, session)
        t.tic('make_intervals_df')

        # Construct the unit df for the session
        unit_df = make_unit_df(cluster_info, unit_global_df.index.max() + 1)
        unit_df = add_session_columns(unit_df, session)
        t.tic('make_unit_df')

        # Construct the master df for the session from the interval df
        session_df = make_session_df(intervals_df, unit_df, spikes)
        t.tic('make_session_df')

        master_df = pd.concat([master_df, session_df])
        master_df.to_pickle(master_data_path)

        unit_global_df = pd.concat([unit_global_df, unit_df])
        unit_global_df.to_pickle(unit_global_path)

    return master_df, unit_global_df


if __name__ == '__main__':
    get_master_df(regenerate=True, add=True)
