import numpy as np


def get_entry_exit(df, trial):
    is_trial = df.trial == trial
    start = df.value == 1
    end = df.value == 0
    port1 = df.port == 1
    port2 = df.port == 2

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

    return bg_entries, bg_exits, exp_entries, exp_exits, early_exp_entries, early_exp_exits
