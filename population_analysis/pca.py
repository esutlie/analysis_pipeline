import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.preprocessing import normalize
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from shapely import LineString
from shapely.geometry import Point
from shapely.ops import nearest_points

# Change
plt.style.use('tableau-colorblind10')

file_name = 'phy_folder_for_ES029_2022-09-14_bot72_0'

spikes = pd.read_pickle(
    f'/Users/baldomeroramirez/Documents/Shuler Rotation/neuropixel-analysis-main/data_saves/{file_name}/spikes.pkl')
events = pd.read_pickle(
    f'/Users/baldomeroramirez/Documents/Shuler Rotation/neuropixel-analysis-main/data_saves/{file_name}/pi_events.pkl')
cluster_info = pd.read_pickle(
    f'/Users/baldomeroramirez/Documents/Shuler Rotation/neuropixel-analysis-main/data_saves/{file_name}/cluster_info.pkl')


# Returns array of length 'time_vector' populated with 1's where spiking events occurred. By cluster.
def get_all_spikes(good_clusters, time_vector, spikes, step_size=0.001):
    min_time = min(spikes['time'].to_numpy())
    all_bools = []
    for i in good_clusters:
        bool_vector = np.zeros(len(time_vector))
        for t in spikes[spikes.cluster == i].time.to_numpy():
            bool_vector[int((t - min_time) / step_size)] = 1
        all_bools.append(bool_vector)
    return all_bools


# Returns array of length time_vector populated with 1's where lick events occurred
def get_all_licks(time_vector, step_size=0.001):
    min_time = min(spikes['time'].to_numpy())
    lick_bools = []
    bool_vector = np.zeros(len(time_vector))
    for t in events[(events['key'] == 'lick') & (events['value'] == 1) & (events['port'] == 1)].time.to_numpy():
        bool_vector[int((t - min_time) / step_size)] = 1
    lick_bools.append(bool_vector)
    return lick_bools


# Get session information
def get_session_info(cluster_info, events, spikes):
    good_clusters = cluster_info.loc[cluster_info['group'] == 'good'].id.to_numpy()
    num_trials = int(np.nanmax(events.trial.values))
    min_time = min(spikes['time'].to_numpy())
    session_start = np.round(min_time, 4)
    session_end = np.round(max(spikes['time'].to_numpy()), 4)
    step_size = 0.001  # ms
    time_vector = np.arange(session_start, session_end, step=step_size)
    exp_events = events.loc[events['port'] == 1]
    prob_df = events[(events['key'] == 'probability')]
    spikes_by_cluster = get_all_spikes(good_clusters, time_vector, spikes)
    rewards = get_all_rewards(time_vector)[0]
    rewards_idxs = np.where(rewards == 1)[0]
    stacked_spikes = np.vstack(spikes_by_cluster)
    tot_spikes_by_cluster = np.mean(stacked_spikes, axis=1)
    return good_clusters, num_trials, time_vector, prob_df, tot_spikes_by_cluster, min_time, rewards_idxs, stacked_spikes


# Returns array of length time_vector populated with 1's where lick events occurred
def get_all_licks(time_vector, step_size=0.001):
    min_time = min(spikes['time'].to_numpy())
    lick_bools = []
    bool_vector = np.zeros(len(time_vector))
    for t in events[(events['key'] == 'lick') & (events['value'] == 1) & (events['port'] == 1)].time.to_numpy():
        bool_vector[int((t - min_time) / step_size)] = 1
    lick_bools.append(bool_vector)
    return lick_bools


def get_all_rewards(time_vector, step_size=0.001):
    min_time = min(spikes['time'].to_numpy())
    reward_bools = []
    bool_vector = np.zeros(len(time_vector))
    for t in events[(events['key'] == 'reward') & (events['value'] == 1)].time.to_numpy():
        bool_vector[int((t - min_time) / step_size)] = 1
    reward_bools.append(bool_vector)
    return reward_bools


# Returns array of length time_vector populated with 1's where reward events occurred.
def get_rewards(time_vector, step_size=0.001):
    min_time = min(spikes['time'].to_numpy())
    reward_bools = []
    bool_vector = np.zeros(len(time_vector))
    for t in events[(events['key'] == 'reward') & (events['value'] == 1) & (events['port'] == 1)].time.to_numpy():
        bool_vector[int((t - min_time) / step_size)] = 1
    reward_bools.append(bool_vector)

    licks = get_all_licks(time_vector)[0]
    where_rwrds = np.where(reward_bools[0] == True)[0]
    where_licks = np.where(licks == True)[0]
    reward_aft_times = []
    for i in range(len(where_rwrds)):
        out = min(np.where(where_licks > where_rwrds[i])[0])
        reward_aft_times.append(out)

    zeros = np.zeros(len(time_vector))
    zeros[where_licks[reward_aft_times]] = 1
    adjusted_reward_times = []
    adjusted_reward_times.append(zeros)

    return reward_bools, adjusted_reward_times


# Time since last event.
def min_dif(a, b, tolerance=0, return_index=False, rev=False):
    if type(a) == pd.core.series.Series:
        a = a.values
    if type(b) == pd.core.series.Series:
        b = b.values
    if rev:
        outer = -1 * np.subtract.outer(a, b)
        outer[outer <= tolerance] = np.nan
    else:
        outer = np.subtract.outer(b, a)
        outer[outer <= tolerance] = np.nan
    # Noinspection PyBroadException
    mins = np.nanmin(outer, axis=0)
    if return_index:
        index = np.nanargmin(outer, axis=0)
        return index, mins
    return mins


# Returns values in ms for indexing spike trains.
def return_ms_indexing(vals):
    idx = (vals * 1e3).astype(int)
    return idx


# Given a trial index returns, all trial times, tr_start, and tr_end.
def get_trial_times(trial_idx):
    trial_times = events[events['trial'] == trial_idx].time.values
    trial_start = np.min(trial_times)
    trial_end = np.max(trial_times)
    return trial_times, trial_start, trial_end


# Returns instances where a rewards occurred in a given trial.
def rewards_by_trial(start, end, reward_idxs):
    mask = np.logical_and(reward_idxs > start, reward_idxs <= end)
    reward_times = reward_idxs[mask]
    return reward_times, start, end


# Bins matrix by given bin_size value.
def bin_matrix(matrix_in, bin_size=100):
    num_bins = round(len(matrix_in[0]) / bin_size)
    matrix_out = np.array_split(matrix_in, num_bins, axis=1)
    matrix_list = []
    for i in range(num_bins):
        matrix_list.append(np.mean(matrix_out[i], axis=1))
    matrix_out = np.asarray(matrix_list).T
    return matrix_out


# Sorts, pads, and stacks array of arrays.
def pad_stack_array(array):
    sorted_list = list(sorted(array, key=len))
    array_lengths = [len(arr) for arr in sorted_list]
    to_pad = []
    for i in range(len(sorted_list)):
        out = np.max(array_lengths) - len(sorted_list[i])
        to_pad.append(out)

    padded_and_sorted = []
    for i in range(len(sorted_list)):
        c = np.zeros(to_pad[i])
        out = np.concatenate((sorted_list[i], c))
        padded_and_sorted.append(out)

    out = np.vstack(padded_and_sorted)
    summed_out = np.sum(out, axis=0)
    return out, summed_out


# Defines active timebins across trials.
def get_actives(array_i):
    array_lengths = [len(arr) for arr in array_i]
    active_intervals = [sum([val > i for val in array_lengths]) for i in range(max(array_lengths))]
    return active_intervals


# Slices stacked spikes.
def slice_spikes(start, end):
    sliced_stack = stacked_spikes[:, start:end + 1]
    return sliced_stack


# Gives matrix based on quantitative parameters
def get_quantitative_matrix(df, column, lower_bound, upper_bound):
    column = column
    lower_bound = lower_bound
    upper_bound = upper_bound
    by_units = []
    df_i = df[(df[column] > lower_bound) & (df[column] < upper_bound)]['spike_trains']
    for i in range(len(good_clusters)):
        cluster_now = []
        for tr in range(len(df_i) - 1):
            first = df_i.values[tr][i]
            cluster_now.append(first)
        by_units.append(cluster_now)

    im_matrix = []
    actives = []
    for i in range(len(good_clusters)):
        bulk, summed = pad_stack_array(by_units[i])
        active = get_actives(by_units[i])
        im_matrix.append(summed / active)
        actives.append(active)

    return im_matrix, actives


# Gives matrix based on qualitative  parameters
def get_categorical_matrix(df, column, parse_by, return_whole=False):
    column = column
    parse_by = parse_by
    by_units = []

    if return_whole:
        spike_arrays = df['spike_trains'].values
    else:
        spike_arrays = df[df[column] == parse_by]['spike_trains'].values
    array_lengths = [len(arr[0]) for arr in spike_arrays]
    active_intervals = [sum([val > i for val in array_lengths]) for i in range(max(array_lengths))]
    im_matrix = np.zeros((len(spike_arrays[0]), len(active_intervals)))

    for arr in spike_arrays:
        im_matrix[:len(arr), :len(arr[0])] += arr

    im_matrix = im_matrix / np.array(active_intervals)

    return im_matrix, active_intervals


# Returns 'true' entries/exits into an exponential port
def get_exp_entries_exits(events, tolerance=1):
    # When the exponential port becomes avaibale.
    exp_available = events.loc[(events['key'] == 'forced_switch')].time.to_numpy()
    # When the mouse enters the expornential port.
    exp_entries = events.loc[(events['key'] == 'head') & (events['value'] == 1) & (events['port'] == 1)].time.to_numpy()
    # When the mouse exists the exponential port.
    exp_exits = events.loc[(events['key'] == 'head') & (events['value'] == 0) & (events['port'] == 1)].time.to_numpy()
    tol = tolerance
    print(exp_available)
    print(exp_entries)
    dif = min_dif(exp_entries, exp_exits)
    # print(len(exp_entries), len(exp_exits))
    exp_entries = exp_entries[:-1]
    # print(len(exp_entries), len(exp_exits))
    exp_entries = exp_entries[np.where(dif > tol)]
    # print(len(exp_entries), len(exp_entries))
    ind = min_dif(exp_available, exp_entries, return_index=True)[0]
    # print(ind)
    exp_first_entries = exp_entries[ind]

    exp_all_entries = events.loc[
        (events['key'] == 'head') & (events['value'] == 1) & (events['port'] == 1)].time.to_numpy()
    exp_exits = events.loc[(events['key'] == 'head') & (events['value'] == 0) & (events['port'] == 1)].time.to_numpy()
    dif = min_dif(exp_exits, exp_all_entries)
    dif[np.where(np.isnan(dif))] = tol + .1
    exp_exits = exp_exits[np.where(dif > tol)]
    if exp_exits[-1] < exp_all_entries[-1]:
        max_time = events.time.max()
        exp_exits = np.concatenate([exp_exits, np.array([max_time])])
    ind = min_dif(exp_first_entries, exp_exits, return_index=True)[0]
    exp_first_exits = exp_exits[ind]

    exponential_first_entries = return_ms_indexing(exp_first_entries)
    exponential_first_exits = return_ms_indexing(exp_first_exits)
    return exponential_first_entries, exponential_first_exits


# Returns list with the information of all trials (strt, end, rwrd_times, etc.)
def get_trial_information(num_trials, exponential_first_entries, exponential_first_exits, prob_df, tolerance=0.01):
    prob_rewards = np.where(get_rewards(time_vector)[0][0] == 1)[0]
    lick_adjusted_rewards = np.where(get_rewards(time_vector)[1][0] == 1)[0]

    trial_information = []
    prob_information = []
    for tr in range(num_trials - 1):
        tr_idx = tr
        tr_reward_times, tr_start, tr_end = rewards_by_trial(exponential_first_entries[tr], exponential_first_exits[tr],
                                                             lick_adjusted_rewards)
        block = np.unique(events[(events['trial'] == tr) & (events['port'] == 1)]['phase'].values)
        trial_information.append([tr_reward_times, tr_start, tr_end, block, tr_idx])
        prob_information.append(
            rewards_by_trial(exponential_first_entries[tr], exponential_first_exits[tr], prob_rewards))

    tolerance = 0.05
    trial_probs = []
    for tr in range(num_trials - 1):
        trial_rwrds = prob_information[tr][0]
        probs_tr = []
        for i in range(len(trial_rwrds)):
            rwrd_time_now = round(trial_rwrds[i] / 1000, 3)
            prob_now = prob_df[
                (prob_df['time'] >= rwrd_time_now) & (prob_df['time'] < (rwrd_time_now + tolerance))].value.values
            if not len(prob_now) == 1:
                print(prob_now)
                print(rwrd_time_now)
                print(prob_df[(prob_df['time'] >= rwrd_time_now - 1) & (prob_df['time'] < (rwrd_time_now + 1))])
            probs_tr.append(prob_now)
        trial_probs.append(probs_tr)

    for tr in range(num_trials - 1):
        flat_probs = [item for sublist in trial_probs[tr] for item in sublist]
        trial_information[tr].append(flat_probs)

    return trial_information


# Cretates session's dataframe
def create_data_frame(num_trials, trial_information, look_back_time=60):
    columns = ['start_time', 'stop_time', 'trial_number', 'trial_time', 'interval_type', 'block', 'reward_probability',
               'spike_trains', 'recent_reward_rate']
    df = pd.DataFrame(columns=columns)
    look_back_time = look_back_time

    for tr in range(1, (num_trials - 1)):
        [rwrd_times, tr_start, tr_end, tr_block, tr_idx, rwrd_probs] = trial_information[tr]
        tr_block = tr_block[0]
        for j in range(len(rwrd_times) + 1):
            if j == 0 and len(rwrd_times):
                data = [tr_start, rwrd_times[0], tr_idx, 0, 'start_reward', tr_block, 0.1]
            elif j == len(rwrd_times) and len(rwrd_times):
                data = [rwrd_times[-1], tr_end, tr_idx, rwrd_times[-1] - tr_start, 'reward_end', tr_block,
                        rwrd_probs[-1]]
            elif len(rwrd_times):
                #                 print(rwrd_times)
                #                 print(rwrd_probs)
                data = [rwrd_times[j - 1], rwrd_times[j], tr_idx, rwrd_times[j - 1] - tr_start, 'reward_reward',
                        tr_block, rwrd_probs[j - 1]]
            else:
                data = [tr_start, tr_end, tr_idx, 0, 'start_end', tr_block, 0.1]
            data.append(slice_spikes(data[0], data[1]))
            recent_rewards = len(events[(events['key'] == 'reward') & (events['value'] == 1) & (
                    events['time'] < (data[0]) / 1000) & (events['time'] > (data[0]) / 1000 - look_back_time)])
            recent_time = min([look_back_time, data[0] / 1000 - events[
                (events['key'] == 'reward') & (events['value'] == 1)].time.min()])
            data.append(recent_rewards / recent_time)
            df = pd.concat([df, pd.DataFrame([data], columns=columns)])

    normalized_spikes = []
    for i in range(len(df)):
        normalized_spikes.append(df['spike_trains'].iloc[i].T / tot_spikes_by_cluster)

    df['normalized_spike_trains'] = normalized_spikes

    where_change_arr = df['block'].values[:-1] != df['block'].values[1:]
    block_changes = np.where(where_change_arr == 1)[0]
    block_idxs = np.zeros(len(df['block']))
    block_idxs[:block_changes[1]] = 1
    block_idxs[block_changes[1]:block_changes[3]] = 2
    block_idxs[block_changes[3]:] = 3
    df['block_idxs'] = block_idxs
    return df


# Get cluters indices by earliest firing rate time.
def get_inds(df):
    im_matrix, actives = get_categorical_matrix(df, 'normalized_spike_trains', 'start_reward', return_whole=True)
    last_bin = np.max(np.where(np.array(actives) > actives[0] * (0.2)))
    filtered_matrix = np.vstack([gaussian_filter(vector, 50) for vector in im_matrix[:, :last_bin]])
    inds = np.argsort(np.argmax(filtered_matrix, axis=1))
    to_imshow = im_matrix[inds]
    return inds, to_imshow, im_matrix


# Pads matrices for plotting
def pad_heatmap_matrices(matrices, good_clusters):
    durations = []
    for matrix in matrices:
        matrix_time = np.shape(matrix)[1]
        durations.append(matrix_time)

    longest = max(durations)
    padded_matrices = []
    for i in range(len(durations)):
        pad_by = np.zeros((len(good_clusters), longest - durations[i]))
        padded_matrices.append(np.concatenate((matrices[i], pad_by), axis=1))
    return padded_matrices


# Plots matrices against each other by type.
def plot_heat_map(df, good_clusters, split_by='block', split_type='cat'):
    inds, to_imshow, im_matrix = get_inds(df)
    if split_type == 'cat':
        cat_vals = np.unique(df[split_by])
        sorted_matrices = []
        for i in cat_vals:
            matrix_i, actives_i = get_categorical_matrix(df, split_by, i)
            matrix_sorted_i = np.array(matrix_i)[inds]
            sorted_matrices.append(matrix_sorted_i)
        to_plot = pad_heatmap_matrices(sorted_matrices, good_clusters)
        num_subs = len(to_plot)
        figure, axis = plt.subplots(num_subs, figsize=(15, 15))
        for s in range(num_subs):
            axis[s].imshow(to_plot[s], vmin=0, vmax=np.quantile(to_imshow, .99), aspect='auto')
    # If values are quantitative (range of vals) rather than categorical
    else:
        sorted_matrices = []
        first_q = np.quantile(df[split_by], [0, 0.25])
        second_q = np.quantile(df[split_by], [0.25, 0.50])
        third_q = np.quantile(df[split_by], [0.50, .75])
        fourth_q = np.quantile(df[split_by], [0.75, 1.0])
        all_qs = [first_q, second_q, third_q, fourth_q]

        for i in range(len(all_qs)):
            matrix_i, actives_i = get_quantitative_matrix(df, split_by, all_qs[i][0], all_qs[i][1])
            matrix_sorted_i = np.array(matrix_i)[inds]
            sorted_matrices.append(matrix_sorted_i)
        to_plot = pad_heatmap_matrices(sorted_matrices, good_clusters)
        num_subs = len(to_plot)
        figure, axis = plt.subplots(num_subs, figsize=(15, 15))
        for s in range(num_subs):
            axis[s].imshow(to_plot[s], vmin=0, vmax=np.quantile(to_imshow, .99), aspect='auto')
    return None


# P.C.A.
def pca(data_bins):
    x = data_bins
    scaler = StandardScaler()
    x = scaler.fit_transform(x.T)
    scaler = StandardScaler()
    x = scaler.fit_transform(x.T)
    pca_model = PCA(n_components=0.99)
    X_r = pca_model.fit_transform(x)
    return X_r, x, pca_model


# Get p.c.a. labels
def pca_labels(df, num_clusters=4):
    im_matrix = get_inds(df)[2]
    pca_matrix = bin_matrix(im_matrix)
    pca_results, scaled_data, pca_model = pca(pca_matrix)
    labels = AgglomerativeClustering(num_clusters).fit_predict(pca_results)
    label_masks = []
    for i in range(len(np.unique(labels))):
        label_masks.append(labels == i)
    return labels, label_masks, pca_matrix


# Plots activity by pca label
def average_activity_by_label(df, label_masks, split_by='block', split_type='cat'):
    if split_by == 'block_idxs':
        splits = np.unique(df[split_by])
        matrices = []
        for i in splits:
            m, a = get_categorical_matrix(df, split_by, i)
            matrices.append(m)
        fig, axs = plt.subplots(2, 2, figsize=(10, 5), sharey=True)
        axs = axs.ravel()
        for i in range(len(label_masks)):
            axs[i].plot(np.mean(bin_matrix(matrices[0])[label_masks[i]], axis=0), label='ts1')
            axs[i].plot(np.mean(bin_matrix(matrices[1])[label_masks[i]], axis=0), label='ts2')
            axs[i].plot(np.mean(bin_matrix(matrices[2])[label_masks[i]], axis=0), label='ts3')
            axs[i].grid(axis="y")

        fig.suptitle("%s -- %s" % (split_by, file_name))
        axs[1].legend()
        fig.tight_layout()
        fig.savefig('graphs_hushu_rotation/{}_{}.png'.format(file_name, split_by))


    elif split_type == 'cat':
        splits = np.unique(df[split_by])
        matrices = []
        for i in splits:
            m, a = get_categorical_matrix(df, split_by, i)
            matrices.append(m)
        fig, axs = plt.subplots(2, 2, figsize=(10, 5), sharey=True)
        axs = axs.ravel()
        for i in range(len(label_masks)):
            axs[i].plot(np.mean(bin_matrix(matrices[0])[label_masks[i]], axis=0), label='B1 0.4')
            axs[i].plot(np.mean(bin_matrix(matrices[1])[label_masks[i]], axis=0), label='B2 0.8')
            axs[i].grid(axis="y")
        fig.suptitle("%s -- %s" % (split_by, file_name))
        axs[1].legend()
        fig.tight_layout()
        fig.savefig('graphs_hushu_rotation/{}_{}.png'.format(file_name, split_by))


    else:
        first_q = np.quantile(df[split_by], [0, 0.25])
        second_q = np.quantile(df[split_by], [0.25, 0.50])
        third_q = np.quantile(df[split_by], [0.50, 0.75])
        fourth_q = np.quantile(df[split_by], [0.75, 1.0])
        all_qs = [first_q, second_q, third_q, fourth_q]

        matrices = []
        for i in range(len(all_qs)):
            m, a = get_quantitative_matrix(df, split_by, all_qs[i][0], all_qs[i][1])
            matrices.append(m)
        fig, axs = plt.subplots(2, 2, figsize=(10, 5), sharey=True)
        axs = axs.ravel()
        for i in range(len(label_masks)):
            m1 = np.array(matrices[0])[label_masks[i]]
            m2 = np.array(matrices[1])[label_masks[i]]
            m3 = np.array(matrices[2])[label_masks[i]]
            m4 = np.array(matrices[3])[label_masks[i]]
            axs[i].plot(np.mean(bin_matrix(m1), axis=0), label='q1')
            axs[i].plot(np.mean(bin_matrix(m2), axis=0), label='q2')
            axs[i].plot(np.mean(bin_matrix(m3), axis=0), label='q3')
            axs[i].plot(np.mean(bin_matrix(m4), axis=0), label='q4')
            tots_units = np.sum(label_masks[i])
            axs[i].title.set_text("PCA_label_%d Number of Units: %d" % (i, tots_units))
            axs[i].grid(axis="y")
        axs[1].legend()
        fig.suptitle("%s -- %s" % (split_by, file_name))
        # fig.suptitle(split_by)
        fig.tight_layout()
        fig.savefig('graphs_hushu_rotation/{}_{}.png'.format(file_name, split_by))

    return None


def fit_curve(principal_component, poly_degree=7):
    x = np.arange(len(principal_component))
    z = np.polyfit(x, principal_component, poly_degree)
    p = np.poly1d(z)
    return p, x


def get_categorical_matrix_two_params(df, column_1, column_2, parse_by_1, parse_by_2, return_whole=False):
    # column = column
    # parse_by = parse_by
    by_units = []

    if return_whole:
        spike_arrays = df['spike_trains'].values
    else:
        spike_arrays = df[(df[column_1] == parse_by_1) & (df[column_2] == parse_by_2)]['spike_trains'].values
    array_lengths = [len(arr[0]) for arr in spike_arrays]
    active_intervals = [sum([val > i for val in array_lengths]) for i in range(max(array_lengths))]
    im_matrix = np.zeros((len(spike_arrays[0]), len(active_intervals)))

    for arr in spike_arrays:
        im_matrix[:len(arr), :len(arr[0])] += arr

    im_matrix = im_matrix / np.array(active_intervals)

    return im_matrix, active_intervals

def fit_pcs(pca_results):
    poly_results = []
    time_poly = np.arange(0, 50, 0.01)
    polynomials = []
    for pc in range(2):
        p, x = fit_curve(pca_results[:,pc])
        poly_results.append(p(time_poly))
        polynomials.append(p)
    return polynomials


good_clusters, num_trials, time_vector, prob_df, tot_spikes_by_cluster, min_time, rewards_idxs, stacked_spikes = get_session_info(
    cluster_info, events, spikes)
exp_entry, exp_exit = get_exp_entries_exits(events)
trial_info = get_trial_information(num_trials, exp_entry, exp_exit, prob_df)
df = create_data_frame(num_trials, trial_info)

if __name__ == '__main__':
    run_pca_analysis()
