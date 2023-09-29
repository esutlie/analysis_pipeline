import numpy as np
from .min_dif import min_dif


def remove(df, key, tolerance, port):
    on_times = df[(df.key == key) & (df.value == 1) & (df.port == port)].session_time.to_numpy()
    off_times = df[(df.key == key) & (df.value == 0) & (df.port == port)].session_time.to_numpy()
    forward = min_dif(on_times, off_times)
    forward_off = min_dif(on_times, off_times, rev=True)
    forward[np.isnan(forward)] = tolerance
    forward_off[np.isnan(forward_off)] = tolerance
    on_times = on_times[forward >= tolerance]
    off_times = off_times[forward_off >= tolerance]

    back = min_dif(off_times, on_times, rev=True)
    back_off = min_dif(off_times, on_times)
    back[np.isnan(back)] = tolerance
    back_off[np.isnan(back_off)] = tolerance
    on_times = on_times[back >= tolerance]
    off_times = off_times[back_off >= tolerance]

    df = df[((df.key != key) | (df.value != 1) | (df.port != port)) | (df.session_time.isin(on_times))]
    df = df[((df.key != key) | (df.value != 0) | (df.port != port)) | (df.session_time.isin(off_times))]
    return df


def data_reduction(df, lick_tol=.01, head_tol=.2):
    df = df[df.key != 'camera']
    df = df[df.phase != 'setup']
    df = remove(df, 'head', head_tol, port=1)
    df = remove(df, 'head', head_tol, port=2)
    df = remove(df, 'lick', lick_tol, port=1)
    df = remove(df, 'lick', lick_tol, port=2)
    return df


def get_entry_exit(df, trial):
    is_trial = df.trial == trial
    start = df.value == 1
    end = df.value == 0
    port1 = df.port == 1
    port2 = df.port == 2
    reward = df.key == 'reward'
    lick = df.key == 'lick'

    trial_start = df[is_trial & start & (df.key == 'trial')].session_time.values[0]
    trial_middle = df[is_trial & end & (df.key == 'LED') & port2].session_time.values[0]
    trial_end = df[is_trial & end & (df.key == 'trial')].session_time.values[0]

    bg_entries = df[is_trial & port2 & start & (df.key == 'head')].session_time.to_numpy()
    bg_exits = df[is_trial & port2 & end & (df.key == 'head')].session_time.to_numpy()

    if len(bg_entries) == 0 or bg_entries[0] > bg_exits[0]:
        bg_entries = np.concatenate([[trial_start], bg_entries])
    if trial_end - bg_entries[-1] < .1:
        bg_entries = bg_entries[:-1]
    if len(bg_exits) == 0 or bg_entries[-1] > bg_exits[-1]:
        bg_entries = np.concatenate([bg_exits, [trial_middle]])

    exp_entries = df[is_trial & port1 & start & (df.key == 'head') &
                     (df.session_time > trial_middle)].session_time.to_numpy()
    exp_exits = df[is_trial & port1 & end & (df.key == 'head') &
                   (df.session_time > trial_middle)].session_time.to_numpy()

    if not (len(exp_entries) == 0 and len(exp_exits) == 0):
        if len(exp_entries) == 0:
            exp_entries = np.concatenate([[trial_middle], exp_entries])
        if len(exp_exits) == 0:
            exp_exits = np.concatenate([exp_exits, [trial_end]])

        if exp_entries[0] > exp_exits[0]:
            exp_entries = np.concatenate([[trial_middle], exp_entries])
        if exp_entries[-1] > exp_exits[-1]:
            exp_exits = np.concatenate([exp_exits, [trial_end]])

    early_exp_entries = df[is_trial & port1 & start & (df.key == 'head') &
                           (df.session_time < trial_middle)].session_time.to_numpy()
    early_exp_exits = df[is_trial & port1 & end & (df.key == 'head') &
                         (df.session_time < trial_middle)].session_time.to_numpy()

    if not (len(early_exp_entries) == 0 and len(early_exp_exits) == 0):
        if len(early_exp_entries) == 0:
            early_exp_entries = np.concatenate([[trial_start], early_exp_entries])
        if len(early_exp_exits) == 0:
            early_exp_exits = np.concatenate([early_exp_exits, [trial_middle]])

        if early_exp_entries[0] > early_exp_exits[0]:
            early_exp_entries = np.concatenate([[trial_start], early_exp_entries])
        if early_exp_entries[-1] > early_exp_exits[-1]:
            early_exp_exits = np.concatenate([early_exp_exits, [trial_middle]])

    if len(bg_entries) != len(bg_exits):
        print()
    if len(exp_entries) != len(exp_exits):
        print()
    if len(early_exp_entries) != len(early_exp_exits):
        print()

    # first_entry = min(exp_entries)
    # first_exit = min(exp_exits)
    #
    # reward_times = df[reward & port1 & start & is_trial].time.values
    # lick_times = df[lick & port1 & start &is_trial].time.values
    #
    # reward_times = reward_times[(max(lick_times) - reward_times) > 0]
    # reward_times = lick_times[min_dif(reward_times, lick_times, return_index=True)[0]]
    # reward_times = reward_times[(reward_times - first_entry) > 0]
    # reward_times = reward_times[(first_exit - reward_times) > 0]

    return bg_entries, bg_exits, exp_entries, exp_exits, early_exp_entries, early_exp_exits

