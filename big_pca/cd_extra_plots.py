import matplotlib.pyplot as plt
import numpy as np
from backend import multi_length_mean
from scipy.spatial import distance_matrix

def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

def extra_plots(activity_list):
    single_run = np.squeeze(activity_list[0])
    plt.plot(single_run)
    plt.show()

    for run in activity_list[:200:5]:
        plt.plot(np.squeeze(run))
    mean_trajectory, counts = multi_length_mean(activity_list)
    plt.plot(np.squeeze(mean_trajectory), c='k')

    plt.show()
