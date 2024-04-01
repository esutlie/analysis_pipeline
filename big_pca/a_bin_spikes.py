import numpy as np
import math
from scipy.stats import skewnorm


def skewed_gaussian_kernel(size, alpha=5, loc=25, scale=100):
    kernel = np.zeros(size)
    for i in range(size):
        kernel[i] = skewnorm.pdf(i, alpha, loc, scale)
    shift = np.max([kernel[0], kernel[-1]])
    kernel -= shift
    kernel[kernel < 0] = 0
    return kernel


def bin_spikes(spikes, width=400, drop_type='mono'):
    cluster_ids = spikes.cluster.unique()
    cluster_ids.sort()
    max_time = math.ceil(spikes.time.max() * 1000)
    spike_bins = np.zeros([len(cluster_ids), max_time])
    shift = 0
    if drop_type == 'mono':
        drop_off = np.ones([width])
    elif drop_type == 'skew':
        drop_off = skewed_gaussian_kernel(width, alpha=10, loc=int(width * .08), scale=int(width / 3.5))
        drop_off = drop_off / np.max(drop_off)
        shift = int(width * .09)
        # plt.plot(drop_off)
        # plt.vlines(shift, 0, 1)
        # plt.show()
    else:  # 'mono'
        drop_off = np.ones([width])
    for i, row in spikes.iterrows():
        unit_ind = np.where(cluster_ids == row.cluster)[0][0]
        drop_start = 0
        drop_end = len(drop_off)
        time_start = round(row.time * 1000 - shift)
        time_end = time_start + len(drop_off)

        if time_start < 0:
            drop_start = drop_start - time_start
            time_start = 0
        if time_end > max_time:
            drop_end = drop_end - int(time_end - max_time)
            time_end = max_time
        spike_bins[unit_ind, time_start:time_end] += drop_off[drop_start:drop_end]
    return spike_bins
