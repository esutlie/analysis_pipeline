import numpy as np
from big_pca.a_bin_spikes import bin_spikes
from big_pca.b_high_pass import high_pass
from big_pca.c_leave_one_out import leave_one_out
from big_pca.c_add_one_in import add_one_in
import backend
from backend import load_data, make_intervals_df, get_session_list
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from cb_method_recurrence import DISTANCE_METRIC

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
    if single:
        intervals_df = intervals_df[(intervals_df.phase == 0) | (intervals_df.phase == 3)]

    if test:
        spikes = spikes[spikes.cluster < 20]
    spike_bins = bin_spikes(spikes, width=3000, drop_type='skew')
    high_pass_spikes = high_pass(spike_bins, .1)
    normalized_spikes = normalize(high_pass_spikes)
    normalized_spikes = normalized_spikes / np.std(normalized_spikes)
    shuffled_spikes = normalized_spikes.copy()
    rng = np.random.default_rng()
    rng.shuffle(shuffled_spikes, axis=1)

    # figure out if it makes sense to shuffle the data before or after dividing into intervals, see if the shuffled
    # data is similarly biased towards lower n

    if DISTANCE_METRIC:
        normalized_spikes /= 4
        normalized_spikes += .5

    print('starting add one in')
    tic = backend.Timer()
    add_one_in(normalized_spikes, intervals_df, session, process=process, test=test, show_plots=show_plots, shuffle=False)
    tic.tic('add one in full time')
    return True


if __name__ == '__main__':
    # run_pca('ES042_2024-02-27_bot168_0_g0')
    # run_pca('ES044_2024-03-08_bot168_1_g0')
    # run_pca('ES041_2024-03-08_bot096_1_g0')
    # run_pca('ES044_2024-03-08_bot168_0_g0')
    # run_pca('ES041_2024-03-08_bot096_0_g0')
    # run_pca('ES042_2024-03-01_bot168_0_g0')
    # run_pca('ES042_2024-02-29_bot168_1_g0')

    session_list = backend.get_session_list()
    # rng = np.random.default_rng()
    # rng.shuffle(session_list)
    for session in session_list:
        run_pca(session)

    # for session in session_list:
    #     run_pca(session)


def hist_normal(normalized_spikes):
    unit = 1
    plt.hist(normalized_spikes[unit], bins=50)
    plt.xlim(-4, 4)
    plt.show()
    np.sum((normalized_spikes[unit] > 0) & (normalized_spikes[unit] < 1)) / len(normalized_spikes[unit])
