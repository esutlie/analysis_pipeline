import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.ndimage import gaussian_filter

plt.style.use('tableau-colorblind10')

file_name = 'phy_folder_for_ES029_2022-09-14_bot72_0'

spikes = pd.read_pickle(
    f'/Users/baldomeroramirez/Documents/Shuler Rotation/neuropixel-analysis-main/data_saves/{file_name}/spikes.pkl')
events = pd.read_pickle(
    f'/Users/baldomeroramirez/Documents/Shuler Rotation/neuropixel-analysis-main/data_saves/{file_name}/pi_events.pkl')
cluster_info = pd.read_pickle(
    f'/Users/baldomeroramirez/Documents/Shuler Rotation/neuropixel-analysis-main/data_saves/{file_name}/cluster_info.pkl')

# In[3]:


good_clusters = cluster_info.loc[cluster_info['group'] == 'good'].id.to_numpy()
num_trials = int(np.nanmax(events.trial.values))
min_time = min(spikes['time'].to_numpy())
session_start = np.round(min_time, 4)
session_end = np.round(max(spikes['time'].to_numpy()), 4)
step_size = 0.001  # ms
time_vector = np.arange(session_start, session_end, step=step_size)
exp_events = events.loc[events['port'] == 1]


# In[4]:


# Returns array of length 'time_vector' populated with 1's where spiking events occurred. By cluster.
def get_all_spikes(good_clusters, time_vector):
    all_bools = []
    for i in good_clusters:
        bool_vector = np.zeros(len(time_vector))
        for t in spikes[spikes.cluster == i].time.to_numpy():
            bool_vector[int((t - min_time) / step_size)] = 1
        all_bools.append(bool_vector)
    return all_bools


# Returns array of length time_vector populated with 1's where reward events occurred.
def get_all_rewards(time_vector):
    reward_bools = []
    bool_vector = np.zeros(len(time_vector))
    for t in events[(events['key'] == 'reward') & (events['value'] == 1)].time.to_numpy():
        bool_vector[int((t - min_time) / step_size)] = 1
    reward_bools.append(bool_vector)
    return reward_bools


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
    # noinspection PyBroadException
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
def rewards_by_trial(start, end, rewards_idxs):
    start = start
    end = end
    mask = np.logical_and(rewards_idxs > start, rewards_idxs < end)
    rewards_times = rewards_idxs[mask]
    return rewards_times, start, end


# Bins.
def binning(arr, bin_size=10):
    binned_array = np.round(arr / bin_size).astype(int)
    return binned_array


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


###
def plot_heat_map(column, parse_by, gauss_sigma, last_bin):
    column = column
    parse_by = parse_by
    by_units = []
    for i in range(len(good_clusters)):
        cluster_now = []
        for tr in range(len(df[df['interval_type'] == parse_by]) - 1):
            first = df[df[column] == parse_by]['spike_trains'].values[tr][i]
            cluster_now.append(first)
        by_units.append(cluster_now)

    im_matrix = []
    for i in range(len(good_clusters)):
        bulk, summed = pad_stack_array(by_units[i])
        im_matrix.append(summed / get_actives(by_units[i]))

    normalized_matrix = normalize(im_matrix, axis=1)
    filtered_matrix = np.vstack([gaussian_filter(vector, gauss_sigma) for vector in normalized_matrix])
    inds = np.argsort(np.argmax(filtered_matrix, axis=1))
    im_matrix = normalized_matrix[inds]
    im_matrix = im_matrix[:, :last_bin]

    plt.figure(figsize=(15, 10))
    plt.imshow(im_matrix, vmin=0, vmax=(np.quantile(im_matrix[:, :int(len(im_matrix[0]) / 2)], .99)), aspect='auto')
    plt.colorbar()
    plt.show()


###
def get_quantitative_matrix(column, lower_bound, upper_bound):
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


###
def get_categorical_matrix(column, parse_by, return_whole=False):
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


#     for i in range(len(good_clusters)):
#         cluster_now = []
#         for tr in range(len(df[df[column] == parse_by]) - 1):
#             first = df[df[column] == parse_by]['spike_trains'].values[tr][i]
#             cluster_now.append(first)
#         by_units.append(cluster_now)

#     im_matrix = []
#     actives = []
#     for i in range(len(good_clusters)):
#         bulk, summed = pad_stack_array(by_units[i])
#         active = get_actives(by_units[i])
#         im_matrix.append(summed / active)
#         actives.append(active)

#     return im_matrix, actives


# In[5]:


spikes_by_cluster = get_all_spikes(good_clusters, time_vector)
rewards = get_all_rewards(time_vector)[0]
rewards_idxs = np.where(rewards == 1)[0]
stacked_spikes = np.vstack(spikes_by_cluster)
tot_spikes_by_cluster = np.mean(stacked_spikes, axis=1)

# In[6]:


# When the exponential port becomes avaibale.
exp_available = events.loc[(events['key'] == 'forced_switch')].time.to_numpy()
# When the mouse enters the expornential port.
exp_entries = events.loc[(events['key'] == 'head') & (events['value'] == 1) & (events['port'] == 1)].time.to_numpy()
# When the mouse exists the exponential port.
exp_exits = events.loc[(events['key'] == 'head') & (events['value'] == 0) & (events['port'] == 1)].time.to_numpy()

# Get clean first entry/exit times to exponential port.
tol = 1
dif = min_dif(exp_entries, exp_exits)
exp_entries = exp_entries[np.where(dif > tol)]
ind = min_dif(exp_available, exp_entries, return_index=True)[0]
exp_first_entries = exp_entries[ind]

exp_all_entries = events.loc[(events['key'] == 'head') & (events['value'] == 1) & (events['port'] == 1)].time.to_numpy()
exp_exits = events.loc[(events['key'] == 'head') & (events['value'] == 0) & (events['port'] == 1)].time.to_numpy()
dif = min_dif(exp_exits, exp_all_entries)
dif[np.where(np.isnan(dif))] = tol + .1
exp_exits = exp_exits[np.where(dif > tol)]
if exp_exits[-1] < exp_all_entries[-1]:
    max_time = events.time.max()
    exp_exits = np.concatenate([exp_exits, np.array([max_time])])
ind = min_dif(exp_first_entries, exp_exits, return_index=True)[0]
exp_first_exits = exp_exits[ind]

# In[7]:


exponential_first_entries = return_ms_indexing(exp_first_entries)
exponential_first_exits = return_ms_indexing(exp_first_exits)
prob_df = events[(events['key'] == 'probability')]

# In[8]:


trial_information = []
for tr in range(num_trials - 1):
    tr_idx = tr
    tr_reward_times, tr_start, tr_end = rewards_by_trial(exponential_first_entries[tr], exponential_first_exits[tr],
                                                         rewards_idxs)
    block = np.unique(events[(events['trial'] == tr) & (events['port'] == 1)]['phase'].values)
    trial_information.append([tr_reward_times, tr_start, tr_end, block, tr_idx])

# In[9]:


tolerance = 0.01
trial_probs = []
for tr in range(num_trials - 1):
    trial_rwrds = trial_information[tr][0]
    probs_tr = []
    for i in range(len(trial_rwrds)):
        rwrd_time_now = round(trial_rwrds[i] / 1000, 3)
        prob_now = prob_df[
            (prob_df['time'] >= rwrd_time_now) & (prob_df['time'] < (rwrd_time_now + tolerance))].value.values
        probs_tr.append(prob_now)
    trial_probs.append(probs_tr)

for tr in range(num_trials - 1):
    flat_probs = [item for sublist in trial_probs[tr] for item in sublist]
    trial_information[tr].append(flat_probs)

# In[10]:


columns = ['start_time', 'stop_time', 'trial_number', 'trial_time', 'interval_type', 'block', 'reward_probability',
           'spike_trains', 'recent_reward_rate']
df = pd.DataFrame(columns=columns)
look_back_time = 60

for tr in range(1, (num_trials - 1)):
    [rwrd_times, tr_start, tr_end, tr_block, tr_idx, rwrd_probs] = trial_information[tr]
    tr_block = tr_block[0]
    for j in range(len(rwrd_times) + 1):
        if j == 0 and len(rwrd_times):
            data = [tr_start, rwrd_times[0], tr_idx, 0, 'start_reward', tr_block, 0.1]
        elif j == np.max(len(rwrd_times)) and len(rwrd_times):
            data = [rwrd_times[-1], tr_end, tr_idx, rwrd_times[-1] - tr_start, 'reward_end', tr_block, rwrd_probs[-1]]
        elif len(rwrd_times):
            data = [rwrd_times[j - 1], rwrd_times[j], tr_idx, rwrd_times[j - 1] - tr_start, 'reward_reward', tr_block,
                    rwrd_probs[j - 1]]
        else:
            data = [tr_start, tr_end, tr_idx, 0, 'start_end', tr_block, 0.1]
        data.append(slice_spikes(data[0], data[1]))
        recent_rewards = len(events[(events['key'] == 'reward') & (events['value'] == 1) & (
                events['time'] < (data[0]) / 1000) & (events['time'] > (data[0]) / 1000 - look_back_time)])
        recent_time = min(
            [look_back_time, data[0] / 1000 - events[(events['key'] == 'reward') & (events['value'] == 1)].time.min()])
        data.append(recent_rewards / recent_time)
        df = pd.concat([df, pd.DataFrame([data], columns=columns)])

# In[11]:


df.head()

# In[12]:


normalized_spikes = []
for i in range(len(df)):
    normalized_spikes.append(df['spike_trains'].iloc[i].T / tot_spikes_by_cluster)

# In[13]:


df['normalized_spike_trains'] = normalized_spikes

# In[14]:


df['normalized_spike_trains1'] = df['spike_trains'].map(lambda x: x.T / tot_spikes_by_cluster)

# In[15]:


im_matrix, actives = get_categorical_matrix('normalized_spike_trains', 'start_reward', return_whole=True)

# In[16]:


last_bin = np.max(np.where(np.array(actives) > actives[0] * (0.2)))

# In[17]:


np.shape(im_matrix[:, :last_bin])

# In[18]:


filtered_matrix = np.vstack([gaussian_filter(vector, 50) for vector in im_matrix[:, :last_bin]])
inds = np.argsort(np.argmax(filtered_matrix, axis=1))
to_imshow = im_matrix[inds]

# In[19]:


plt.figure(figsize=(5, 5))
plt.imshow(to_imshow, vmin=0, vmax=np.quantile(to_imshow, .99), aspect='auto')
plt.show()

# ## Plotting

# In[20]:


matrix_sr, actives_sr = get_categorical_matrix('interval_type', 'start_reward')
matrix_rr, actives_rr = get_categorical_matrix('interval_type', 'reward_reward')
matrix_er, actives_er = get_categorical_matrix('interval_type', 'reward_end')

# In[21]:


matrix_sorted_sr = np.array(matrix_sr)[inds]
matrix_sorted_rr = np.array(matrix_rr)[inds]
matrix_sorted_er = np.array(matrix_er)[inds]

# In[22]:


sr = np.shape(matrix_sorted_sr)[1]
rr = np.shape(matrix_sorted_rr)[1]
er = np.shape(matrix_sorted_er)[1]
sr_pad = rr - sr
er_pad = rr - er
sr_to_pad = np.zeros((len(good_clusters), sr_pad))
er_to_pad = np.zeros((len(good_clusters), er_pad))
matrix_sorted_sr = np.concatenate((matrix_sorted_sr, sr_to_pad), axis=1)
matrix_sorted_er = np.concatenate((matrix_sorted_er, er_to_pad), axis=1)

# In[23]:


figure, axis = plt.subplots(3, figsize=(15, 15))
axis[0].imshow(matrix_sorted_sr, vmin=0, vmax=np.quantile(to_imshow, .99), aspect='auto')
axis[0].set_title("strt_reward")
axis[1].imshow(matrix_sorted_rr, vmin=0, vmax=np.quantile(to_imshow, .99), aspect='auto')
axis[1].set_title("reward_reward")
axis[2].imshow(matrix_sorted_er, vmin=0, vmax=np.quantile(to_imshow, .99), aspect='auto')
axis[2].set_title("end_reward")
plt.show()

# In[24]:


matrix_b1, actives_b1 = get_categorical_matrix('block', '0.4')
matrix_b2, actives_b2 = get_categorical_matrix('block', '0.8')

# In[25]:


matrix_sorted_b1 = np.array(matrix_b1)[inds]
matrix_sorted_b2 = np.array(matrix_b2)[inds]

# In[26]:


b1 = np.shape(matrix_sorted_b1)[1]
b2 = np.shape(matrix_sorted_b2)[1]
b2_pad = b1 - b2
b2_to_pad = np.zeros((len(good_clusters), b2_pad))
matrix_sorted_b2 = np.concatenate((matrix_sorted_b2, b2_to_pad), axis=1)

# In[27]:


figure, axis = plt.subplots(2, figsize=(15, 15))
axis[0].imshow(matrix_sorted_b1, vmin=0, vmax=np.quantile(to_imshow, .99), aspect='auto')
axis[0].set_title("block_1")
axis[1].imshow(matrix_sorted_b2, vmin=0, vmax=np.quantile(to_imshow, .99), aspect='auto')
axis[1].set_title("block_2")
plt.show()

# ## Dimensionality Reduction - PCA

# In[28]:


# 100ms time bins
pca_matrix = im_matrix[inds]
num_bins = round(len(pca_matrix[0]) / 100)
pca_matrix = np.array_split(pca_matrix, num_bins, axis=1)
pca_list = []
for i in range(num_bins):
    pca_list.append(np.mean(pca_matrix[i], axis=1))

# In[29]:


pca_matrix = np.asarray(pca_list).T

# In[30]:


plt.imshow(im_matrix[inds, :1000], aspect='auto')

# In[31]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


# In[34]:


def pca(data_bins):
    x = data_bins
    scaler = StandardScaler()
    x = scaler.fit_transform(x.T)
    scaler = StandardScaler()
    x = scaler.fit_transform(x.T)
    pca_model = PCA(n_components=0.99)
    X_r = pca_model.fit_transform(x)
    return X_r, x, pca_model


# In[35]:


pca_results, scaled_data, pca_model = pca(pca_matrix)
labels = AgglomerativeClustering(4).fit_predict(pca_results)

# In[36]:


plt.scatter(pca_results[:, 0], pca_results[:, 1])
plt.plot()
plt.grid()

# In[37]:


# Label to color dict
label_color_dict = {0: 'orange', 1: 'black', 2: 'blue', 3: 'magenta'}
# Color vector creation
cvec = [label_color_dict[label] for label in labels]

# In[38]:


for i in range(len(good_clusters)):
    plt.scatter(pca_results[i, 0], pca_results[i, 1], color=cvec[i])
plt.grid()

# ## Average activity of units by label

# In[39]:


label_masks = []
for i in range(len(np.unique(labels))):
    label_masks.append(labels == i)

# In[40]:


fig, axs = plt.subplots(2, 2, figsize=(10, 5))
axs[0, 0].plot(np.mean(pca_matrix[label_masks[0]], axis=0))
axs[0, 0].set_title('Label 1')

axs[0, 1].plot(np.mean(pca_matrix[label_masks[1]], axis=0))
axs[0, 1].set_title('Label 2')

axs[1, 0].plot(np.mean(pca_matrix[label_masks[2]], axis=0))
axs[1, 0].set_title('Label 3')

axs[1, 1].plot(np.mean(pca_matrix[label_masks[3]], axis=0))
axs[1, 1].set_title('Label 4')

fig.tight_layout()
plt.show()


# In[41]:


def bin_matrix(matrix_in, bin_size=100):
    num_bins = round(len(matrix_in[0]) / bin_size)
    matrix_out = np.array_split(matrix_in, num_bins, axis=1)
    matrix_list = []
    for i in range(num_bins):
        matrix_list.append(np.mean(matrix_out[i], axis=1))
    matrix_out = np.asarray(matrix_list).T
    return matrix_out


# ## Block

# In[42]:


fig, axs = plt.subplots(2, 2, figsize=(10, 5))
axs[0, 0].plot(np.mean(bin_matrix(matrix_sorted_b1)[label_masks[0]], axis=0), label='B_1')
axs[0, 0].plot(np.mean(bin_matrix(matrix_sorted_b2)[label_masks[0]], axis=0), label='B_2')
axs[0, 0].set_title('Label 1')
axs[0, 1].plot(np.mean(bin_matrix(matrix_sorted_b1)[label_masks[1]], axis=0), label='B_1')
axs[0, 1].plot(np.mean(bin_matrix(matrix_sorted_b2)[label_masks[1]], axis=0), label='B_2')
axs[0, 1].set_title('Label 2')
axs[1, 0].plot(np.mean(bin_matrix(matrix_sorted_b1)[label_masks[2]], axis=0))
axs[1, 0].plot(np.mean(bin_matrix(matrix_sorted_b2)[label_masks[2]], axis=0))
axs[1, 0].set_title('Label 3')
axs[1, 1].plot(np.mean(bin_matrix(matrix_sorted_b1)[label_masks[3]], axis=0))
axs[1, 1].plot(np.mean(bin_matrix(matrix_sorted_b2)[label_masks[3]], axis=0))
axs[1, 1].set_title('Label 4')
fig.tight_layout()
axs[0, 1].legend()
plt.show()

# In[43]:


figure, axis = plt.subplots(2, figsize=(15, 15))
axis[0].imshow(matrix_sorted_b1[label_masks[1]], vmin=0, vmax=np.quantile(to_imshow, .99), aspect='auto')
axis[0].set_title("strt_reward")
axis[1].imshow(matrix_sorted_b2[label_masks[1]], vmin=0, vmax=np.quantile(to_imshow, .99), aspect='auto')
axis[1].set_title("reward_reward")
plt.show()

# ## Recent Reward Rate

# In[44]:


matrix_rr1, actives_rr1 = get_quantitative_matrix('recent_reward_rate',
                                                  np.quantile(df['recent_reward_rate'].values, 0.0),
                                                  np.quantile(df['recent_reward_rate'].values, 0.25))
matrix_rr2, actives_rr2 = get_quantitative_matrix('recent_reward_rate',
                                                  np.quantile(df['recent_reward_rate'].values, 0.25),
                                                  np.quantile(df['recent_reward_rate'].values, 0.5))
matrix_rr3, actives_rr3 = get_quantitative_matrix('recent_reward_rate',
                                                  np.quantile(df['recent_reward_rate'].values, 0.5),
                                                  np.quantile(df['recent_reward_rate'].values, 0.75))
matrix_rr4, actives_rr4 = get_quantitative_matrix('recent_reward_rate',
                                                  np.quantile(df['recent_reward_rate'].values, 0.75),
                                                  np.quantile(df['recent_reward_rate'].values, 1.0))

# In[45]:


matrix_sorted_rr1 = np.array(matrix_rr1)[inds]
matrix_sorted_rr2 = np.array(matrix_rr2)[inds]
matrix_sorted_rr3 = np.array(matrix_rr3)[inds]
matrix_sorted_rr4 = np.array(matrix_rr4)[inds]

# In[46]:


fig, axs = plt.subplots(2, 2, figsize=(10, 5))
axs[0, 0].plot(np.mean(bin_matrix(matrix_sorted_rr1)[label_masks[0]], axis=0))
axs[0, 0].plot(np.mean(bin_matrix(matrix_sorted_rr2)[label_masks[0]], axis=0))
axs[0, 0].plot(np.mean(bin_matrix(matrix_sorted_rr3)[label_masks[0]], axis=0))
axs[0, 0].plot(np.mean(bin_matrix(matrix_sorted_rr4)[label_masks[0]], axis=0))

axs[0, 0].set_title('Label 1')
axs[0, 1].plot(np.mean(bin_matrix(matrix_sorted_rr1)[label_masks[1]], axis=0), label='rr1')
axs[0, 1].plot(np.mean(bin_matrix(matrix_sorted_rr2)[label_masks[1]], axis=0), label='rr2')
axs[0, 1].plot(np.mean(bin_matrix(matrix_sorted_rr3)[label_masks[1]], axis=0), label='rr3')
axs[0, 1].plot(np.mean(bin_matrix(matrix_sorted_rr4)[label_masks[1]], axis=0), label='rr4')

axs[0, 1].set_title('Label 2')
axs[1, 0].plot(np.mean(bin_matrix(matrix_sorted_rr1)[label_masks[2]], axis=0))
axs[1, 0].plot(np.mean(bin_matrix(matrix_sorted_rr2)[label_masks[2]], axis=0))
axs[1, 0].plot(np.mean(bin_matrix(matrix_sorted_rr3)[label_masks[2]], axis=0))
axs[1, 0].plot(np.mean(bin_matrix(matrix_sorted_rr4)[label_masks[2]], axis=0))

axs[1, 0].set_title('Label 3')
axs[1, 1].plot(np.mean(bin_matrix(matrix_sorted_rr1)[label_masks[3]], axis=0))
axs[1, 1].plot(np.mean(bin_matrix(matrix_sorted_rr2)[label_masks[3]], axis=0))
axs[1, 1].plot(np.mean(bin_matrix(matrix_sorted_rr3)[label_masks[3]], axis=0))
axs[1, 1].plot(np.mean(bin_matrix(matrix_sorted_rr4)[label_masks[3]], axis=0))

axs[1, 1].set_title('Label 4')

fig.tight_layout()
axs[0, 1].legend()
plt.show()

# ## Time in Trial

# In[47]:


matrix_tt1, actives_tt1 = get_quantitative_matrix('trial_time', np.quantile(df['trial_time'].values, 0.0),
                                                  np.quantile(df['trial_time'].values, 0.25))
matrix_tt2, actives_tt2 = get_quantitative_matrix('trial_time', np.quantile(df['trial_time'].values, 0.25),
                                                  np.quantile(df['trial_time'].values, 0.5))
matrix_tt3, actives_tt3 = get_quantitative_matrix('trial_time', np.quantile(df['trial_time'].values, 0.5),
                                                  np.quantile(df['trial_time'].values, 0.75))
matrix_tt4, actives_tt4 = get_quantitative_matrix('trial_time', np.quantile(df['trial_time'].values, 0.75),
                                                  np.quantile(df['trial_time'].values, 1.0))

# In[48]:


matrix_sorted_tt1 = np.array(matrix_tt1)[inds]
matrix_sorted_tt2 = np.array(matrix_tt2)[inds]
matrix_sorted_tt3 = np.array(matrix_tt3)[inds]
matrix_sorted_tt4 = np.array(matrix_tt4)[inds]

# In[49]:


fig, axs = plt.subplots(2, 2, figsize=(10, 5))
axs[0, 0].plot(np.mean(bin_matrix(matrix_sorted_tt1)[label_masks[0]], axis=0))
axs[0, 0].plot(np.mean(bin_matrix(matrix_sorted_tt2)[label_masks[0]], axis=0))
axs[0, 0].plot(np.mean(bin_matrix(matrix_sorted_tt3)[label_masks[0]], axis=0))
axs[0, 0].plot(np.mean(bin_matrix(matrix_sorted_tt4)[label_masks[0]], axis=0))

axs[0, 0].set_title('Label 1')
axs[0, 1].plot(np.mean(bin_matrix(matrix_sorted_tt1)[label_masks[1]], axis=0), label='tt1')
axs[0, 1].plot(np.mean(bin_matrix(matrix_sorted_tt2)[label_masks[1]], axis=0), label='tt2')
axs[0, 1].plot(np.mean(bin_matrix(matrix_sorted_tt3)[label_masks[1]], axis=0), label='tt3')
axs[0, 1].plot(np.mean(bin_matrix(matrix_sorted_tt4)[label_masks[1]], axis=0), label='tt4')

axs[0, 1].set_title('Label 2')
axs[1, 0].plot(np.mean(bin_matrix(matrix_sorted_tt1)[label_masks[2]], axis=0))
axs[1, 0].plot(np.mean(bin_matrix(matrix_sorted_tt2)[label_masks[2]], axis=0))
axs[1, 0].plot(np.mean(bin_matrix(matrix_sorted_tt3)[label_masks[2]], axis=0))
axs[1, 0].plot(np.mean(bin_matrix(matrix_sorted_tt4)[label_masks[2]], axis=0))

axs[1, 0].set_title('Label 3')
axs[1, 1].plot(np.mean(bin_matrix(matrix_sorted_tt1)[label_masks[3]], axis=0))
axs[1, 1].plot(np.mean(bin_matrix(matrix_sorted_tt2)[label_masks[3]], axis=0))
axs[1, 1].plot(np.mean(bin_matrix(matrix_sorted_tt3)[label_masks[3]], axis=0))
axs[1, 1].plot(np.mean(bin_matrix(matrix_sorted_tt4)[label_masks[3]], axis=0))

axs[1, 1].set_title('Label 4')

fig.tight_layout()
axs[0, 1].legend()
plt.show()

# ## Time in Session

# In[50]:


where_change_arr = df['block'].values[:-1] != df['block'].values[1:]
np.where(where_change_arr == 1)[0]

# In[51]:


block_idxs = np.zeros(len(df['block']))

# In[52]:


block_idxs[:121] = 1
block_idxs[121:240] = 2
block_idxs[240:] = 3

# In[53]:


df['block_idxs'] = block_idxs

# In[54]:


matrix_bid1, actives_bid1 = get_categorical_matrix('block_idxs', 1)
matrix_bid2, actives_bid2 = get_categorical_matrix('block_idxs', 2)
matrix_bid3, actives_bid3 = get_categorical_matrix('block_idxs', 3)

# In[55]:


matrix_sorted_bid1 = np.array(matrix_bid1)[inds]
matrix_sorted_bid2 = np.array(matrix_bid2)[inds]
matrix_sorted_bid3 = np.array(matrix_bid3)[inds]

# In[56]:


fig, axs = plt.subplots(2, 2, figsize=(10, 5))
axs[0, 0].plot(np.mean(bin_matrix(matrix_sorted_bid1)[label_masks[0]], axis=0))
axs[0, 0].plot(np.mean(bin_matrix(matrix_sorted_bid2)[label_masks[0]], axis=0))
axs[0, 0].plot(np.mean(bin_matrix(matrix_sorted_bid3)[label_masks[0]], axis=0))

axs[0, 0].set_title('Label 1')
axs[0, 1].plot(np.mean(bin_matrix(matrix_sorted_bid1)[label_masks[1]], axis=0), label='ts1')
axs[0, 1].plot(np.mean(bin_matrix(matrix_sorted_bid2)[label_masks[1]], axis=0), label='ts2')
axs[0, 1].plot(np.mean(bin_matrix(matrix_sorted_bid3)[label_masks[1]], axis=0), label='ts3')

axs[0, 1].set_title('Label 2')
axs[1, 0].plot(np.mean(bin_matrix(matrix_sorted_bid1)[label_masks[2]], axis=0))
axs[1, 0].plot(np.mean(bin_matrix(matrix_sorted_bid2)[label_masks[2]], axis=0))
axs[1, 0].plot(np.mean(bin_matrix(matrix_sorted_bid3)[label_masks[2]], axis=0))

axs[1, 0].set_title('Label 3')
axs[1, 1].plot(np.mean(bin_matrix(matrix_sorted_bid1)[label_masks[3]], axis=0))
axs[1, 1].plot(np.mean(bin_matrix(matrix_sorted_bid2)[label_masks[3]], axis=0))
axs[1, 1].plot(np.mean(bin_matrix(matrix_sorted_bid3)[label_masks[3]], axis=0))

axs[1, 1].set_title('Label 4')

fig.tight_layout()
axs[0, 1].legend()
plt.show()

# ## How fast?

# In[57]:


pca_results, scaled_data, pca_transpose_model = pca(pca_matrix.T)

# In[64]:


weights = np.arange(len(pca_results[:, 0]))
weights = np.flip(weights)
plt.figure(figsize=(10, 5))
plt.scatter(pca_results[:, 0], pca_results[:, 1], c=weights, cmap='Greys', marker='o', label='session')
plt.grid()
plt.legend()
plt.show()

# In[66]:


b1_proj_pca = pca_transpose_model.transform(bin_matrix(matrix_sorted_b1).T)
b2_proj_pca = pca_transpose_model.transform(bin_matrix(matrix_sorted_b2).T)

# In[99]:


plt.figure(figsize=(10, 5))
plt.scatter(b1_proj_pca[:, 0], b1_proj_pca[:, 1], c=weights, cmap='Greys', marker='o')
plt.scatter(b2_proj_pca[:, 0], b2_proj_pca[:, 1], c=weights, cmap='Blues', marker='o')

plt.grid()
weights = np.flip(weights)
plt.show()

# ## Curve-Fitting of Sorts

# In[68]:


session_pc_1 = pca_results[:, 0]
session_pc_2 = pca_results[:, 1]

# In[69]:


fig, axs = plt.subplots(2, figsize=(10, 5))
axs[0].plot(session_pc_1, 'o')
axs[1].plot(session_pc_2, 'o')
plt.show()

# In[70]:


x = np.arange(len(session_pc_1))
z = np.polyfit(x, session_pc_1, 7)
p = np.poly1d(z)

# In[71]:


plt.plot(p(x), '--', session_pc_1, 'o')
plt.show()

# In[72]:


z_2 = np.polyfit(x, session_pc_2, 7)
p_2 = np.poly1d(z_2)

# In[73]:


plt.plot(p_2(x), '--', session_pc_2, 'o')
plt.show()

# In[77]:


to_poly = np.arange(0, 50, 0.1)
x = p(to_poly)
y = p_2(to_poly)

plt.plot(x, y)
plt.scatter(pca_results[1:, 0], pca_results[1:, 1], c=weights[1:], cmap='Greys', marker='o')
plt.show()


def fit_curve(principal_component, poly_degree=7):
    x = np.arange(len(principal_component))
    z = np.polyfit(x, principal_component, poly_degree)
    p = np.poly1d(z)
    return p, x


# In[96]:


pc = 1
p, x = fit_curve(pca_results[:, pc])
plt.plot(p(x), '--', pca_results[:, pc], 'o')
plt.show()

# In[105]:


pc = 1
p, x = fit_curve(b1_proj_pca[:, pc])
plt.plot(p(x), '--', b1_proj_pca[:, pc], 'o')
plt.show()

# In[ ]:
def run_pca_analysis():
    pass

if __name__ == '__main__':
    run_pca_analysis()