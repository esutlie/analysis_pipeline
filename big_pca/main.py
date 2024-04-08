import numpy as np
from big_pca.a_bin_spikes import bin_spikes
from big_pca.b_high_pass import high_pass
from big_pca.c_leave_one_out import leave_one_out
from big_pca.c_add_one_in import add_one_in
import backend
from backend import load_data, make_intervals_df, get_session_list
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def run_pca(session):
    test = False
    show_plots = False
    process = False
    print('started')
    spikes, pi_events, cluster_info = backend.load_data(session, photometry=False)

    if test:
        spikes = spikes[spikes.cluster < 20]
    spike_bins = bin_spikes(spikes, width=3000, drop_type='skew')
    high_pass_spikes = high_pass(spike_bins, .1)
    normalized_spikes = normalize(high_pass_spikes)
    normalized_spikes = normalized_spikes / np.std(normalized_spikes)
    intervals_df = backend.make_intervals_df(pi_events)

    print('starting add one in')
    tic = backend.Timer()
    add_one_in(normalized_spikes, intervals_df, session, process=process, test=test, show_plots=show_plots)
    tic.tic('add one in full time')


if __name__ == '__main__':
    session_list = backend.get_session_list()
    # run_pca('ES039_2024-02-28_bot144_1_g0')
    # run_pca('ES029_2022-09-12_bot72_0_g0')
    run_pca('ES039_2024-03-08_bot144_1_g0')

    # for session in session_list:
    #     run_pca(session)
