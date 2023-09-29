import numpy as np

import backend


def get_trial_events(data, entry_tolerance=.5, exit_tolerance=2, include_unrewarded=True, travel_time=.5):
    [head, trial, cue, reward, lick, leave, start, port1, port2] = backend.get_bools(data)
    entries = []
    exits = []
    rewards = []
    trial_numbers = []
    active_time = []
    for trial_number in range(1, int(data.trial.max())):
        available_time = data[cue & start & port2 & (data.trial == trial_number)].time.values[0]
        entry_times = data[head & port1 & start & (data.trial == trial_number)].time.values
        exit_times = data[head & port1 & leave & (data.trial == trial_number)].time.values
        reward_times = data[reward & port1 & start & (data.trial == trial_number)].time.values
        lick_times = data[lick & port1 & start & (data.trial == trial_number)].time.values
        trial_block = data[(data.trial == trial_number)].phase.iloc[0]
        port2_time = 10 if trial_block in ['0.4', '.4'] else 5

        if not len(exit_times):
            continue
        entry_times = entry_times[(entry_times - available_time) > 0]
        entry_times = entry_times[backend.min_dif(entry_times, exit_times) > entry_tolerance]
        if not len(entry_times) or not len(lick_times) or not (len(reward_times) or include_unrewarded):
            continue
            # This triggers if there are no exits after the main entry, which happens if they stick
            # their butt in the opposite side and trigger the end of the trial early. Just omit these trials
        first_entry = entry_times.min()

        exit_times = exit_times[(exit_times - first_entry) > 0]
        exit_times = exit_times[backend.min_dif(exit_times, np.concatenate([entry_times, [np.inf]])) > exit_tolerance]
        first_exit = exit_times.min()
        if first_exit - first_entry > 30:
            continue

        reward_times = reward_times[(max(lick_times) - reward_times) > 0]

        reward_times = lick_times[backend.min_dif(reward_times, lick_times, return_index=True)[0]]
        reward_times = reward_times[(reward_times - first_entry) > 0]
        reward_times = reward_times[(first_exit - reward_times) > 0]

        entries.append(first_entry)
        exits.append(first_exit)
        rewards.append(reward_times)  # actually first lick after reward times
        trial_numbers.append(trial_number)
        active_time.append(port2_time + travel_time * 2 + first_exit - first_entry)
    return entries, exits, rewards, trial_numbers, active_time
