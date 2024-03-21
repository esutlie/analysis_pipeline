from big_pca.ca_extract_intervals import extract_intervals
from sklearn.linear_model import LinearRegression
from backend import multi_length_mean
import numpy as np
from wpca import WPCA
import matplotlib.pyplot as plt


def pca_scoring(j, normalized_spikes, intervals_df, leave_out_list):
    if j in leave_out_list:
        return None
    show_plots = np.random.random() < .05
    one_out_spikes = np.delete(normalized_spikes, [j] + leave_out_list, axis=0)
    interval_spikes, intervals_df = extract_intervals(one_out_spikes, intervals_df)
    activity_list = intervals_df.activity.to_list()
    pca, mean_transform = fit_weighted_pca(activity_list, show_plots=show_plots)
    component_weights = pca.explained_variance_ratio_
    scores = []
    for activity in activity_list:
        transformed_activity = pca.transform(activity.T)
        scores.append(
            estimate_velocity(mean_transform, transformed_activity, component_weights, show_plots=False))

    return [np.mean(scores), np.std(scores), j, pca]


def fit_weighted_pca(list_of_arr, show_plots=False):
    time_limit = 4000
    n_components = min(10, list_of_arr[0].shape[0])
    mean_trajectory, counts = multi_length_mean(list_of_arr)
    mean_trajectory, counts = mean_trajectory[:, :time_limit], counts[:, :time_limit]
    print(f'mean_trajectory shape: {mean_trajectory.shape}')
    weights = (counts / counts.max() * .9)
    weighted_pca = WPCA(n_components=n_components).fit(mean_trajectory.T, weights=weights.T)
    transformed = weighted_pca.transform(mean_trajectory.T)

    if show_plots:
        plt.plot(transformed[:, :3], c='black')
        plt.show()
        plt.plot(np.arange(1, len(weighted_pca.explained_variance_ratio_) + 1), weighted_pca.explained_variance_ratio_)
        plt.show()
    return weighted_pca, transformed


def estimate_velocity(mean_trajectories, trajectory, component_weights, show_plots=False):
    window = 1000
    # plt.plot(mean_trajectories[:, 0], mean_trajectories[:, 1])
    # plt.plot(trajectory[:, 0], trajectory[:, 1])
    # plt.scatter([mean_trajectories[0, 0], trajectory[0, 0]], [mean_trajectories[0, 1], trajectory[0, 1]], c='k',
    #             zorder=3)
    # plt.show()

    anchor = 0
    matched_points = np.zeros([trajectory.shape[0]])
    t = np.arange(trajectory.shape[0])
    for i, point in enumerate(trajectory):
        lower_bound = max(0, round(anchor - window / 2))
        upper_bound = min(mean_trajectories.shape[0], round(anchor + window / 2))
        distances = np.linalg.norm(np.multiply(point - mean_trajectories[lower_bound:upper_bound], component_weights),
                                   axis=1)
        matched_points[i] = np.argmin(distances) + lower_bound
        anchor = np.argmin(distances) + lower_bound

    reg = LinearRegression().fit(t.reshape(-1, 1), matched_points)
    score = reg.score(t.reshape(-1, 1), matched_points)
    fitted_line = reg.predict(t.reshape(-1, 1))
    if show_plots:
        plt.scatter(t, matched_points, zorder=3)
        plt.plot(t, fitted_line, c='k')
        plt.show()

        # plt.plot(mean_trajectories[:, 0], mean_trajectories[:, 1])
        # plt.plot(trajectory[:, 0], trajectory[:, 1])
        # plt.scatter([mean_trajectories[0, 0], trajectory[0, 0]], [mean_trajectories[0, 1], trajectory[0, 1]], c='k',
        #             zorder=3)
        # plt.show()

        plt.plot(mean_trajectories[:, 0], mean_trajectories[:, 1])
        plt.plot(trajectory[:, 0], trajectory[:, 1])
        plt.scatter([mean_trajectories[0, 0], trajectory[0, 0]], [mean_trajectories[0, 1], trajectory[0, 1]], c='k',
                    zorder=3)
        for i in range(len(trajectory) // 50):
            plt.plot([mean_trajectories[round(matched_points[i * 50]), 0], trajectory[i * 50, 0]],
                     [mean_trajectories[round(matched_points[i * 50]), 1], trajectory[i * 50, 1]], linewidth=1, c='k')

        plt.show()

    return score
