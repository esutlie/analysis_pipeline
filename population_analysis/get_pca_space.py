import numpy as np
import itertools
from sklearn.decomposition import PCA


def get_pca_space(normalized_spikes, interval_ids, intervals_df):
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
        # idx_set = intervals_df[group & phase].index.tolist()
        # if len(idx_set) >= 2:
        #     group_spikes = [normalized_spikes[:, np.where(i == interval_ids)[0]] for i in idx_set]
        #     lengths = np.array([len(g[0]) for g in group_spikes])
        #     max_length = np.sort(lengths)[-2]
        #     set_averages = np.vstack(
        #         [np.mean(np.vstack([group_spikes[j][:, i] for j in np.where(i < lengths)[0]]), axis=0) for i in
        #          range(max_length)])
        #     spike_groups.append(set_averages)
    np.concatenate(spike_groups, axis=0)
    pca = fit_pca(np.concatenate(spike_groups, axis=0))
    return pca


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
                max_length = np.quantile(lengths, min_active)
        else:
            print('min_active must be and int or a float between 0 and 1, using default of 2 for min_active')
            max_length = np.sort(lengths)[-2]

        set_averages = np.vstack(
            [np.mean(np.vstack([group_spikes[j][:, i] for j in np.where(i < lengths)[0]]), axis=0) for i in
             range(max_length)])
        return set_averages
    else:
        return None
