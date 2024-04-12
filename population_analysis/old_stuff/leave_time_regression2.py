import backend
import numpy as np
from create_bins_df import create_precision_df
from population_analysis.old_stuff.get_pca_space import get_pca_space, get_set_averages, get_sets, single_pca_traj
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter1d


def leave_time_regression():
    color_sets = backend.get_color_sets()
    files = backend.get_session_list()
    for session in files:
        if session != 'ES029_2022-09-14_bot72_0_g0':
            continue
        [normalized_spikes, _, _], interval_ids, intervals_df = create_precision_df(session)

        pca_space = single_pca_traj(normalized_spikes, interval_ids, intervals_df, plot=True)

        pca_space = get_pca_space(normalized_spikes, interval_ids, intervals_df)
        pca_spikes = pca_space.transform(normalized_spikes.T).T
        blocks = np.unique(intervals_df.block)
        blocks.sort()
        block_bools = [intervals_df.block == block for block in blocks]
        block_trajectories = []
        exit_type = intervals_df.interval_phase == 2
        for j, block in enumerate(block_bools):
            set_averages = get_set_averages(intervals_df, block, pca_spikes, interval_ids, min_active=.5)
            block_trajectories.append(set_averages)

            opp_group_spikes = get_sets(intervals_df, ~block, pca_spikes, interval_ids)
            average_map, std_err_map, opp_group_map = get_relatives(opp_group_spikes, set_averages)
            plot_relative(average_map, std_err_map, c1=color_sets['set2'][j], c2=color_sets['set2'][(j + 1) % 2],
                          name=str((j + 1) % 2))

            set_averages = get_set_averages(intervals_df, block, pca_spikes, interval_ids, min_active=1)
            exit_opp_spikes = get_sets(intervals_df, ~block & exit_type, pca_spikes, interval_ids)
            average_map, std_err_map, exit_opp_map = get_relatives(exit_opp_spikes, set_averages)
            plot_all_relative(exit_opp_map)
            # group_spikes = get_sets(intervals_df, block, pca_spikes, interval_ids)
            # plot_all_2d(group_spikes, opp_group_spikes, color_sets['set2'][j], color_sets['set2'][(j + 1) % 2])
        print()


def get_relatives(opp_group_spikes, set_averages):
    opp_group_map = [np.argmin(np.sum((spikes.T[:, None, :] - set_averages[None, :, :]) ** 2, axis=2), axis=1)
                     for spikes in opp_group_spikes]
    lengths = np.array([len(g) for g in opp_group_map])
    average_map = np.vstack(
        [np.mean(np.vstack([opp_group_map[j][i] for j in np.where(i < lengths)[0]]), axis=0) for i in
         range(max(lengths))])
    std_map = np.vstack(
        [np.std(np.vstack([opp_group_map[j][i] for j in np.where(i < lengths)[0]]), axis=0) for i in
         range(max(lengths))])
    actives = np.array([len(np.where(i < lengths)[0]) for i in range(max(lengths))])
    std_err_map = np.squeeze(std_map) / np.sqrt(actives)
    return average_map, std_err_map, opp_group_map


def plot_relative(average_map, std_err_map, c1, c2, max_ind=1000, name='block'):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.arange(max_ind)/1000, average_map[:max_ind]/1000, c=c2)
    ax.fill_between(np.arange(max_ind)/1000, (np.squeeze(average_map[:max_ind]) - std_err_map[:max_ind])/1000,
                    (np.squeeze(average_map[:max_ind]) + std_err_map[:max_ind])/1000, alpha=.5,
                    edgecolor=c2,
                    facecolor=c2,
                    linewidth=0)

    ax.plot([0, max_ind/1000], [0, max_ind/1000], c=c1)
    ax.set_xlabel('Actual Time (sec)')
    ax.set_ylabel('Projected Time (sec)')
    ax.set_aspect('equal', 'box')
    plt.show()
    backend.save_fig(fig, f'block{name}_relative_trajectories.png')


def plot_all_relative(opp_group_map, max_ind=1000):
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(len(opp_group_map)):
        if len(opp_group_map[i]) > max_ind:
            ax.plot(opp_group_map[i][:max_ind])
        if len(opp_group_map[i]) < max_ind:
            ax.plot(opp_group_map[i])
    ax.set_aspect('equal', 'box')
    plt.show()


def plot_singles_to_average(average, interval_spikes, intervals, c1, c2):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(average[:, 0], average[:, 1], average[:, 2], c=c1)
    to_plot = [[interval_spikes[i][:, 0], interval_spikes[i][:, 1], interval_spikes[i][:, 2]] for i in
               intervals]
    for points in to_plot:
        # ax.plot(points[0], points[1], points[2], c=c2, linewidth=1.5)
        ax.plot(points[0], points[1], points[2], linewidth=1.5)
    plt.show()


def plot_all_3d(group_spikes, opp_group_spikes, c1, c2):
    cmap = matplotlib.cm.get_cmap('gist_rainbow')
    sigma = 50
    points = 10
    colors = [c1, c2]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for j, spikes in enumerate([group_spikes, opp_group_spikes]):
        for i in range(5):
            smoothed_spikes = gaussian_filter1d(spikes[i], sigma, axis=0)
            ax.plot(smoothed_spikes[:, 0], smoothed_spikes[:, 1], smoothed_spikes[:, 2], c=colors[j], linewidth=.5)
    for j, spikes in enumerate([group_spikes, opp_group_spikes]):
        for i in range(5):
            for k in range(points):
                ind = int(k * 1000 / points)
                if len(spikes[i]) > ind:
                    smoothed_spikes = gaussian_filter1d(spikes[i], sigma, axis=0)
                    ax.scatter(smoothed_spikes[ind, 0], smoothed_spikes[ind, 1], smoothed_spikes[ind, 2],
                               c=cmap(k / 12), s=4)

    plt.show()


def plot_all_2d(group_spikes, opp_group_spikes, c1, c2):
    cmap = matplotlib.cm.get_cmap('gist_rainbow')
    sigma = 50
    points = 10
    colors = [c1, c2]
    fig = plt.figure()
    ax = fig.add_subplot()
    for j, spikes in enumerate([group_spikes, opp_group_spikes]):
        for i in range(30):
            smoothed_spikes = gaussian_filter1d(spikes[i], sigma, axis=0)
            ax.plot(smoothed_spikes[:, 0], smoothed_spikes[:, 1], c=colors[j], linewidth=.5)
    for j, spikes in enumerate([group_spikes, opp_group_spikes]):
        for i in range(30):
            for k in range(points):
                ind = int(k * 1000 / points)
                if len(spikes[i]) > ind:
                    smoothed_spikes = gaussian_filter1d(spikes[i], sigma, axis=0)
                    ax.scatter(smoothed_spikes[ind, 0], smoothed_spikes[ind, 1], c=cmap(k / 12), s=4, zorder=10)

    plt.show()


if __name__ == '__main__':
    leave_time_regression()
