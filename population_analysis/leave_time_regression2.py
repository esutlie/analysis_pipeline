import backend
import numpy as np
from create_bins_df import create_precision_df
from get_pca_space import get_pca_space, get_set_averages
import matplotlib.pyplot as plt
import math


def leave_time_regression():
    color_sets = backend.get_color_sets()
    files = backend.get_session_list()
    for session in files:
        if session != 'ES029_2022-09-14_bot72_0_g0':
            continue
        normalized_spikes, interval_ids, intervals_df = create_precision_df(session)
        pca_space = get_pca_space(normalized_spikes, interval_ids, intervals_df)
        pca_spikes = pca_space.transform(normalized_spikes.T).T
        block_bools = [intervals_df.block == block for block in np.unique(intervals_df.block)]
        block_trajectories = []
        for j, block in enumerate(block_bools):
            set_averages = get_set_averages(intervals_df, block, pca_spikes, interval_ids, min_active=10)
            block_trajectories.append(set_averages)
            idx_set = intervals_df[~block].index.tolist()
            opp_group_spikes = [pca_spikes[:, np.where(i == interval_ids)[0]].T for i in idx_set]
            for i in range(math.floor(len(opp_group_spikes) / 5) - 1):
                plot_singles_to_average(set_averages, opp_group_spikes, range(i * 5, (i + 1) * 5),
                                        color_sets['set2'][j], color_sets['set2'][(j + 1) % 2])
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # colors = [color_sets['set2'][0], color_sets['set2'][1], color_sets['set2'][1], color_sets['set2'][1]]
            # ax.plot(set_averages[:, 0], set_averages[:, 1], set_averages[:, 2], c=colors[0])
            # to_plot = [[opp_group_spikes[i][:, 0], opp_group_spikes[i][:, 1], opp_group_spikes[i][:, 2]] for i in
            #            range(5, 10)]
            # for points in to_plot:
            #     ax.plot(points[0], points[1], points[2], c=colors[1])
            # plt.show()
            # for i in range(5):
            #     plt.plot(opp_group_spikes[i][:, 0], c='b')

        plt.plot(block_trajectories[0][:, 0])
        plt.plot(block_trajectories[1][:, 0])
        plt.show()
        plt.plot(block_trajectories[0][:, 1])
        plt.plot(block_trajectories[1][:, 1])
        plt.show()
        plt.plot(block_trajectories[0][:, 0], block_trajectories[0][:, 1])
        plt.plot(block_trajectories[1][:, 0], block_trajectories[1][:, 1])
        plt.show()
        ax = plt.axes(projection='3d')
        ax.plot3D(block_trajectories[0][:, 0], block_trajectories[0][:, 1], block_trajectories[0][:, 2])
        ax.plot3D(block_trajectories[1][:, 0], block_trajectories[1][:, 1], block_trajectories[1][:, 2])
        plt.show()
        print()


def plot_singles_to_average(average, interval_spikes, intervals, c1, c2):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(average[:, 0], average[:, 1], average[:, 2], c=c1)
    to_plot = [[interval_spikes[i][:, 0], interval_spikes[i][:, 1], interval_spikes[i][:, 2]] for i in
               intervals]
    for points in to_plot:
        ax.plot(points[0], points[1], points[2], c=c2)
    plt.show()


if __name__ == '__main__':
    leave_time_regression()
