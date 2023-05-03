import numpy as np
import itertools
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import backend
import umap


def get_pca_space(normalized_spikes, interval_ids, intervals_df, plot=False):
    group_bools = [intervals_df.group == group for group in np.unique(intervals_df.group)]
    phase_bools = [intervals_df.interval_phase == phase for phase in np.unique(intervals_df.interval_phase)]
    block_bools = [intervals_df.block == block for block in np.unique(intervals_df.block)]
    spike_groups = []
    # for group, phase in itertools.product(group_bools, phase_bools):

    for block in block_bools:
        set_averages = get_set_averages(intervals_df, block, normalized_spikes, interval_ids, min_active=.5)
        # set_averages = get_set_averages(intervals_df, group & phase, normalized_spikes, interval_ids)
        if set_averages is not None:
            spike_groups.append(set_averages)
    pca = fit_pca(np.concatenate(spike_groups, axis=0))
    if plot:
        spike_groups_pca = [pca.transform(spike_group) for spike_group in spike_groups]
        plot_speeds(spike_groups_pca)
    return pca


def single_pca_traj(normalized_spikes, interval_ids, intervals_df, plot=False):
    intervals_df['length'] = intervals_df.interval_ends - intervals_df.interval_starts
    by_length = intervals_df.sort_values('length').index.to_numpy()[::-1]
    spikes_list = []
    for i in range(50):
        interval_spikes = normalized_spikes[:, np.where(by_length[i] == interval_ids)[0]]
        spikes_list.append(interval_spikes)
    stacked_activity = np.stack([spikes[:, :len(spikes_list[-1][0]) - 1] for spikes in spikes_list])
    clusters = backend.NearestLines(stacked_activity).elbow()
    # clusters = backend.NearestLines(stacked_activity).fit()
    mean_activity = np.mean(stacked_activity, axis=0)
    pca = fit_pca(mean_activity.T)
    # pca = fit_pca(np.concatenate(spikes_list, axis=1).T)
    transformed_spikes = [pca.transform(spikes.T) for spikes in spikes_list]
    plot_speeds(transformed_spikes)
    plot_speeds([pca.transform(mean_activity.T)])
    # if plot:
    #     spikes_pca = pca.transform(interval_spikes.T)
    #     plot_speeds([spikes_pca])


def plot_speeds(spike_groups, save_plot=False):
    three_d = True if len(spike_groups[0][0]) > 2 else False
    color_sets = backend.get_color_sets()
    fig = plt.figure()
    if three_d:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()
    ax.set_title('Relative Trajectories for High and Low Rate Blocks')
    for j in range(len(spike_groups)):
        if three_d:
            ax.plot(spike_groups[j][:, 0], spike_groups[j][:, 1], spike_groups[j][:, 2],
                    c=color_sets['set2'][j % len(color_sets['set2'])])
        else:
            ax.plot(spike_groups[j][:, 0], spike_groups[j][:, 1], c=color_sets['set2'][j % len(color_sets['set2'])])
        break
    for i in range(10):
        ind = int(len(spike_groups[-1]) * i / 10)
        for j in range(len(spike_groups)):
            if three_d:
                ax.scatter(spike_groups[j][ind, 0], spike_groups[j][ind, 1], spike_groups[j][ind, 2], c=f'{i / 10:.1f}')
            else:
                ax.scatter(spike_groups[j][ind, 0], spike_groups[j][ind, 1], c=f'{i / 10:.1f}')
            break
    ax.set_aspect('equal', 'box')
    if len(spike_groups) == 2:
        ax.legend(['0.4 rewards/sec', '0.8 rewards/sec'], loc='upper right')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    if three_d:
        ax.set_zlabel('PC 3')
    if save_plot:
        backend.save_fig(fig, f'relative_trajectories.png')
    else:
        plt.show()


def fit_pca(x, pc_count=10):
    pca = PCA(pc_count)
    pca.fit(x)
    return pca


def get_set_averages(intervals_df, bools, spikes, interval_ids, min_active=None):
    if min_active is None:
        min_active = 2
    idx_set = intervals_df[bools].index.tolist()
    if len(idx_set) >= min_active:
        group_spikes = [spikes[:, np.where(i == interval_ids)[0]] for i in idx_set]
        lengths = np.array([len(g[0]) for g in group_spikes])
        if type(min_active) == int:
            max_length = np.sort(lengths)[-min_active]
        elif type(min_active) == float:
            if min_active > 1:
                print('min_active must be and int or a float between 0 and 1, using default of 2 for min_active')
                max_length = np.sort(lengths)[-2]
            else:
                max_length = int(np.quantile(lengths, min_active))
        else:
            print('min_active must be and int or a float between 0 and 1, using default of 2 for min_active')
            max_length = np.sort(lengths)[-2]

        set_averages = np.vstack(
            [np.mean(np.vstack([group_spikes[j][:, i] for j in np.where(i < lengths)[0]]), axis=0) for i in
             range(max_length)])
        return set_averages
    else:
        return None


def get_sets(intervals_df, bools, spikes, interval_ids):
    idx_set = intervals_df[bools].index.tolist()
    return [spikes[:, np.where(i == interval_ids)[0]] for i in idx_set]
