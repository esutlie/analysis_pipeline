import numpy as np
from big_pca.a_bin_spikes import bin_spikes
from big_pca.b_high_pass import high_pass
from big_pca.c_leave_one_out import leave_one_out
from big_pca.c_add_one_in import add_one_in
import backend
from backend import load_data, make_intervals_df, get_session_list
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

'''
df = pi_events
df[(df.key!='camera')&(df.key!='lick')&(df.key!='probability')]
'''


def run_pca(session):
    test = False
    show_plots = False
    process = False
    print('started')
    spikes, pi_events, cluster_info = backend.load_data(session, photometry=False)

    intervals_df, single = backend.make_intervals_df(pi_events, report_single=True)
    if not single:
        return None

    if test:
        spikes = spikes[spikes.cluster < 20]
    spike_bins = bin_spikes(spikes, width=3000, drop_type='skew')
    high_pass_spikes = high_pass(spike_bins, .1)
    normalized_spikes = normalize(high_pass_spikes)
    normalized_spikes = normalized_spikes / np.std(normalized_spikes)

    print('starting add one in')
    tic = backend.Timer()
    add_one_in(normalized_spikes, intervals_df, session, process=process, test=test, show_plots=show_plots)
    tic.tic('add one in full time')
    return True


if __name__ == '__main__':
    run_pca('ES042_2024-02-27_bot168_0_g0')

    session_list = backend.get_session_list()
    rng = np.random.default_rng()
    rng.shuffle(session_list)
    for session in session_list:
        run_pca(session)

    # for session in session_list:
    #     run_pca(session)
