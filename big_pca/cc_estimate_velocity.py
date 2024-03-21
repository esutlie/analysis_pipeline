import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


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
            plt.plot([mean_trajectories[round(matched_points[i*50]), 0], trajectory[i*50, 0]],
                     [mean_trajectories[round(matched_points[i*50]), 1], trajectory[i*50, 1]], linewidth=1, c='k')

        plt.show()

    return score
