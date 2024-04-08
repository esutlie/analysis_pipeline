import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from backend import multi_length_mean
from wpca import WPCA
from big_pca.a_bin_spikes import bin_spikes
from big_pca.b_high_pass import high_pass
from big_pca.c_leave_one_out import leave_one_out
import backend
from backend import load_data, make_intervals_df, get_session_list
from sklearn.preprocessing import normalize
from big_pca.ca_extract_intervals import extract_intervals
from sklearn.decomposition import PCA


def pca_space(session):
    '''
    change it to take the average of similar segments
    '''
    master_df = pd.read_pickle(os.path.join(os.path.dirname(os.getcwd()), '_master', 'data', 'master_data.pkl'))
    session_df = master_df[master_df.session == session]
    velocity_df = pd.read_pickle(os.path.join('results', session + '_recurrence.pkl'))
    selected_interval = velocity_df.arr.iloc[0][0]
    spikes, pi_events, cluster_info = backend.load_data(session, photometry=False)
    spike_bins = bin_spikes(spikes, width=3000, drop_type='skew')
    high_pass_spikes = high_pass(spike_bins, .1)
    normalized_spikes = normalize(high_pass_spikes)
    normalized_spikes = normalized_spikes / np.std(normalized_spikes)
    intervals_df = backend.make_intervals_df(pi_events)

    leave_out_list = []
    one_out_spikes = np.delete(normalized_spikes, leave_out_list, axis=0)
    interval_spikes, intervals_df = extract_intervals(one_out_spikes, intervals_df)
    activity_list = intervals_df.activity.to_list()

    interval = activity_list[selected_interval]
    interval_norm = interval - np.mean(interval, axis=1)[:, None]
    interval_norm = interval_norm / np.std(interval_norm, axis=1)[:, None]
    interval_weighted = interval_norm * np.linspace(10, 1, interval_norm.shape[1])

    weights = np.repeat(np.linspace(1, .5, interval.shape[1]), interval.shape[0])
    weighted_pca = WPCA(n_components=10).fit(interval.T, weights=weights.T)
    transformed = weighted_pca.transform(interval.T)

    weighted_pca = PCA(n_components=10).fit(interval.T)
    transformed = weighted_pca.transform(interval.T)

    weighted_pca = PCA(n_components=10).fit(interval_weighted.T)
    transformed = weighted_pca.transform(interval_norm.T)
    transformed_all = []
    for inter in activity_list:
        interval_norm = inter - np.mean(inter, axis=1)[:, None]
        interval_norm = interval_norm / np.std(interval_norm, axis=1)[:, None]
        transformed_all.append(weighted_pca.transform(interval_norm.T))

    # plt.plot(transformed[:, :3], c='black')
    # plt.show()
    # plt.plot(np.arange(1, len(weighted_pca.explained_variance_ratio_) + 1), weighted_pca.explained_variance_ratio_)
    # plt.show()
    plt.plot(transformed[:, 0], transformed[:, 1])
    plt.scatter(transformed[0, 0], transformed[0, 1])
    int_to_plot=0
    plt.plot(transformed_all[int_to_plot][:, 0], transformed_all[int_to_plot][:, 1])
    plt.scatter(transformed_all[int_to_plot][0, 0], transformed_all[int_to_plot][0, 1])
    # plt.scatter(transformed[::10, 0], transformed[::10, 1])
    plt.show()
    print()


if __name__ == '__main__':
    pca_space('ES029_2022-09-12_bot72_0_g0')
