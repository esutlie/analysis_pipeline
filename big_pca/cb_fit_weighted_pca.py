from backend import multi_length_mean
import numpy as np
from wpca import WPCA
import matplotlib.pyplot as plt


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
