import numpy as np
import itertools
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import backend


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


def plot_speeds(spike_groups, save_plot=False):
    color_sets = backend.get_color_sets()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title('Relative Trajectories for High and Low Rate Blocks')
    ax.plot(spike_groups[0][:, 0], spike_groups[0][:, 1], spike_groups[0][:, 2], c=color_sets['set2'][0])
    ax.plot(spike_groups[1][:, 0], spike_groups[1][:, 1], spike_groups[1][:, 2], c=color_sets['set2'][1])
    for i in range(10):
        ind = int(len(spike_groups[0]) * i / 10)
        ax.scatter(spike_groups[0][ind, 0], spike_groups[0][ind, 1], spike_groups[0][ind, 2], c=f'{i / 10:.1f}')
        ax.scatter(spike_groups[1][ind, 0], spike_groups[1][ind, 1], spike_groups[1][ind, 2], c=f'{i / 10:.1f}')
    ax.set_aspect('equal', 'box')
    ax.legend(['0.4 rewards/sec', '0.8 rewards/sec'], loc='upper right')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
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
