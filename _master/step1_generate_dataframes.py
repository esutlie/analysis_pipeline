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


def exp_func(x, single=False):
    if not single:
        return np.exp(-x / 8)
    else:
        reward_level = 0.5994974874371859  # cumulative for an 8 reward version
        starting_prob = 0.1301005025125628
        return 1 / ((1 - reward_level) * np.exp(starting_prob / reward_level * x) + reward_level)


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
    # session_list = ['ES041_2024-03-08_bot096_1_g0']
    # session_list = ['ES039_2024-03-08_bot144_1_g0']
    session_list = backend.get_session_list()
    if 'session' in master_df.keys():
        session_list = [session for session in session_list if session not in master_df['session'].values]

    for session in session_list:
        spikes, pi_events, cluster_info = backend.load_data(session, photometry=False)
        if len(spikes) == 0 or np.isnan(pi_events.trial.max()):
            print(f'session: {session} is empty or broken')
            continue

        t.tic('load_data')
        # Construct the intervals df for the session
        intervals_df = backend.make_intervals_df(pi_events)
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
    get_master_df(regenerate=False, add=True)
