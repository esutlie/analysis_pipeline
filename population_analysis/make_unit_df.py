import backend
import numpy as np
from create_bins_df import create_precision_df
import matplotlib.pyplot as plt
import random


def make_unit_df():
    color_sets = backend.get_color_sets()
    files = backend.get_session_list()
    x = np.arange(10000)
    for session in files:
        info = backend.get_info(session)
        if info['task'] != 'cued_no_forgo_forced':
            continue
        [_, convolved_spikes, _], interval_ids, intervals_df = create_precision_df(session, regenerate=False)
        # max(sum(interval_ids==i) for i in np.unique(interval_ids))  # Find the max interval length
        blocks = intervals_df.block.unique()
        blocks.sort()
        padded = np.stack([np.pad(convolved_spikes[:, np.where(interval_ids == i)[0]],
                                  ((0, 0), (0, len(x) - sum(interval_ids == i))), constant_values=np.nan) for i in
                           np.unique(interval_ids)])
        means = np.zeros([6, len(padded[0]), len(x)])
        stds = np.zeros([6, len(padded[0]), len(x)])
        blocks_labels = []
        for i, group in enumerate(intervals_df.group.unique()):
            group_intervals = padded[intervals_df[intervals_df.group == group].index]
            means[i] = np.nanmean(group_intervals, axis=0)
            stds[i] = np.nanstd(group_intervals, axis=0)
            blocks_labels.append(np.where(intervals_df[intervals_df.group == group].block.values[0] == blocks)[0][0])
        colors = [color_sets['set2'][blocks_labels[0]], color_sets['set2'][blocks_labels[1]],
                  color_sets['set2_med_dark'][blocks_labels[2]], color_sets['set2_med_dark'][blocks_labels[3]],
                  color_sets['set2_dark'][blocks_labels[4]], color_sets['set2_dark'][blocks_labels[5]]]

        for i in range(len(means[0])):
            for j in range(len(means)):
                plt.plot(range(len(means[0, i])), means[j, i].T, c=colors[j])
            plt.title(f'Unit {i} of {len(means[0])-1}')
            plt.legend(blocks[blocks_labels])
            plt.show()

        print('test')


def unit_interval_rates(intervals_df):
    rates = np.vstack(list(intervals_df.rate.values))
    means = np.mean(rates, axis=0)
    std = np.std(rates, axis=0)
    plt.errorbar(range(len(means)), means, std)
    plt.show()
    x = np.ones(np.shape(rates)) * np.array(range(np.shape(rates)[1]))
    x = x.flatten() + np.array([random.random() / 2 - .25 for _ in range(len(x.flatten()))])
    fig, ax = plt.subplots(figsize=[10, 8])
    ax.scatter(x, np.concatenate(list(intervals_df.rate.values)), s=.5)
    plt.show()


if __name__ == '__main__':
    make_unit_df()
