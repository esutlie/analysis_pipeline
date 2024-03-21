import backend
import numpy as np

def add_session_columns(df, session):
    [mouse, date, num_in_day, recording_block] = backend.unpack_session_name(session)
    df['mouse'] = [mouse] * len(df)
    df['date'] = [date] * len(df)
    df['session_num_in_day'] = [num_in_day] * len(df)
    df['bot_row'] = [recording_block] * len(df)
    df['session'] = [session] * len(df)
    return df


def get_spike_times(df, spikes):
    spike_lists = []
    starts = [np.searchsorted(spikes.time, df.start)][0]
    ends = [np.searchsorted(spikes.time, df.end)][0]
    for i, (_, row) in enumerate(df.iterrows()):
        interval_spikes = spikes.iloc[starts[i]:ends[i]]
        groups = interval_spikes.groupby('cluster').groups
        spike_times = [interval_spikes.loc[groups[unit_id]].time.to_numpy() if unit_id in groups.keys() else
                       np.array([]) for unit_id in row.unit_session_ids]
        spike_lists.append(spike_times)
    return spike_lists


def bins500ms(df):
    interval_length_max = 10  # seconds
    bin_size = .05  # milliseconds
    slider_width = .5  # milliseconds
    t = np.linspace(0, interval_length_max, 1 + int(interval_length_max / bin_size))
    slider = np.ones([int(slider_width/bin_size)])
    result = []
    if 'spike_times' in df.keys():
        for _, row in df.iterrows():  # for each interval
            interval_bins = round((row.end-row.start)/bin_size)
            activity = np.zeros([len(row.spike_times), interval_bins])
            # activity = np.zeros([len(row.spike_times), len(t)])
            for j, arr in enumerate(row.spike_times):  # for each unit
                for spike in arr:
                    start = np.searchsorted(t, spike-row.start)-1
                    end = min(start+len(slider), interval_bins)
                    activity[j, start:end] += slider[:end-start]
                print()
            result.append(activity)
        return result
    else:
        print('cannot calculate until get_spike_times has been added to df as \'spike_times\'')

# i dont think center of mas is going to be useful the way I'd hoped, since it will always be skewed toward time when
# there are more samples or toward the few spikes bringing up the average when there are few samples.
# def get_spike_time_metrics(df):
#     com = []
#     std = []
#     if 'spike_times' in df.keys():
#         for i, row in df.iterrows():
#             row_com = np.empty([len(row.spike_times)])
#             row_com[:] = np.nan
#             row_std = np.empty([len(row.spike_times)])
#             row_std[:] = np.nan
#             for j, arr in enumerate(row.spike_times):
#                 if len(arr) > 0:
#                     row_com[j] = np.mean(arr - row.start)
#                     row_std[j] = np.std(arr - row.start)
#             com.append(row_com)
#             std.append(row_std)
#         return com, std
#     else:
#         print('cannot get center of mass until get_spike_times has been added to df as \'spike_times\'')
