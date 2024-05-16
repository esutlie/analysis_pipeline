import numpy as np
from big_pca.a_bin_spikes import bin_spikes
from big_pca.b_high_pass import high_pass
from big_pca.c_leave_one_out import leave_one_out
import backend
from backend import load_data, make_intervals_df, get_session_list
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def run_pca(session):
    print('started')
    spikes, pi_events, cluster_info = backend.load_data(session, photometry=False)

    spike_bins = bin_spikes(spikes, width=3000, drop_type='skew')
    high_pass_spikes = high_pass(spike_bins, .1)
    normalized_spikes = normalize(high_pass_spikes)
    normalized_spikes = normalized_spikes / np.std(normalized_spikes)
    intervals_df = backend.make_intervals_df(pi_events)

    print('starting leave one out')
    tic = backend.Timer()
    leave_one_out(normalized_spikes, intervals_df, session, process=True, test=False, show_plots=False)
    tic.tic('leave one out full time')


if __name__ == '__main__':
    session_list = backend.get_session_list()
    # run_pca('ES039_2024-02-28_bot144_1_g0')
    # run_pca('ES029_2022-09-12_bot72_0_g0')

    for session in session_list:
        run_pca(session)
