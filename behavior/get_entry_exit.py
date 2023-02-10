import numpy as np

import backend


def get_entry_exit(data, entry_tolerance=.5, exit_tolerance=1):
    [head, trial, cue, reward, lick, leave, start, port1, port2] = backend.get_bools(data)
    entries = []
    exits = []
    trial_numbers = []
    for trial_number in range(1, int(data.trial.max())):
        available_time = data[cue & start & port2 & (data.trial == trial_number)].time.values[0]
        entry_times = data[head & port1 & start & (data.trial == trial_number)].time.values
        exit_times = data[head & port1 & leave & (data.trial == trial_number)].time.values

        entry_times = entry_times[(entry_times - available_time) > 0]
        entry_times = entry_times[backend.min_dif(entry_times, exit_times) > entry_tolerance]
        if not len(entry_times):
            continue
            # This triggers if there are no exits after the main entry, which happens if they stick
            # their butt in the opposite side and trigger the end of the trial early. Just omit these trials
        first_entry = entry_times.min()

        exit_times = exit_times[(exit_times - first_entry) > 0]
        exit_times = exit_times[backend.min_dif(exit_times, np.concatenate([entry_times, [np.inf]])) > exit_tolerance]
        first_exit = exit_times.min()

        entries.append(first_entry)
        exits.append(first_exit)
        trial_numbers.append(trial_number)
    return entries, exits, trial_numbers
