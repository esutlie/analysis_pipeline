import numpy as np


def extract_intervals(spikes, intervals):
    spike_arrays = []
    for i, row in intervals.iterrows():
        start = int(1000 * row.start)
        end = int(1000 * row.end)
        spike_arrays.append(spikes[:, start:end:10])
    # intervals.loc[:, 'activity'] = spike_arrays
    intervals['activity'] = spike_arrays
    spike_arrays = np.concatenate(spike_arrays, axis=1)
    return spike_arrays, intervals
