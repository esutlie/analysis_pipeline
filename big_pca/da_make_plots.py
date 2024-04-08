import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from cb_method_recurrence import recurrence_known
from big_pca.a_bin_spikes import bin_spikes
from big_pca.b_high_pass import high_pass
from big_pca.c_leave_one_out import leave_one_out
from big_pca.c_add_one_in import add_one_in
import backend
from backend import load_data, make_intervals_df, get_session_list
from sklearn.preprocessing import normalize
from big_pca.ca_extract_intervals import extract_intervals


def make_plots(session):
    result = True
    n = 0
    while result:
        result = velocity_regression(session, n)
        n += 1

    print()


def make_recurrence_plots(session):
    velocity_df = pd.read_pickle(os.path.join('results', session + '_recurrence_add_in.pkl'))
    [selection, velocity, intercept, scores, starts] = velocity_df.iloc[-1].arr

    spikes, pi_events, cluster_info = backend.load_data(session, photometry=False)
    spike_bins = bin_spikes(spikes, width=3000, drop_type='skew')
    high_pass_spikes = high_pass(spike_bins, .1)
    normalized_spikes = normalize(high_pass_spikes)
    normalized_spikes = normalized_spikes / np.std(normalized_spikes)
    intervals_df = backend.make_intervals_df(pi_events)
    one_in_spikes = normalized_spikes[np.array(velocity_df.add_in.iloc[-1])]

    interval_spikes, intervals_df = extract_intervals(one_in_spikes, intervals_df)
    activity_list = intervals_df.activity.to_list()
    start_list = intervals_df.start.to_list()

    recurrence_known(activity_list, selection, known_intercepts=intercept, known_slopes=velocity)


def velocity_regression(session, n):
    """
    change it so you can double check that the segments aren't getting shuffled.
    incorporate the scores as the hue
    """
    master_df = pd.read_pickle(os.path.join(os.path.dirname(os.getcwd()), '_master', 'data', 'master_data.pkl'))
    session_df = master_df[master_df.session == session]
    # velocity_df = pd.read_pickle(os.path.join('results', session + '_recurrence.pkl'))
    velocity_df = pd.read_pickle(os.path.join('results', session + '_recurrence_add_in.pkl'))
    if n >= len(velocity_df):
        return False
    [selection, velocity, intercept, scores, starts] = velocity_df.iloc[n].arr
    # velocity[selection+1:] = 1/velocity[selection+1:]
    session_df = session_df.assign(velocity=velocity)
    session_df = session_df.assign(intercept=intercept)
    session_df = session_df.assign(log_velocity=np.log10(velocity))
    session_df = session_df.assign(score=scores)
    session_df['duration'] = session_df.end - session_df.start
    df_non_start = session_df[(session_df.phase != 0) & (session_df.phase != 3)]
    # df_non_start = session_df[(session_df.phase == 2)]
    # df_best = df_non_start[df_non_start.score > .31]
    df_best = df_non_start[df_non_start.score > np.quantile(df_non_start.score, .5)]
    df_early = df_non_start[df_non_start.group < 4]
    df_late = df_non_start[df_non_start.group >= 4]
    df_non_2 = df_best[(df_best.velocity > 2.2) | (df_best.velocity < 1.8)]
    df_non_short = df_best[df_best.duration > .6]
    options = ['leave_from_reward', 'phase', 'trial_time', 'leave_from_entry', 'block', 'start', 'recent_rate',
               'optimal']

    sns.histplot(df_non_start, x='log_velocity', binwidth=.05)
    plt.show()
    # sns.scatterplot(data=session_df, x="velocity", y="leave_from_reward", hue="trial_time")
    # plt.title('Velocity Plot')
    # plt.show()

    # With Scores in Hue
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    for i, ax in enumerate(axes.flatten()):
        if options[i] in ['phase', 'block', 'group']:
            sns.boxplot(data=df_best, x=options[i], y="log_velocity", ax=ax)
        else:
            sns.scatterplot(data=df_best, x=options[i], y="log_velocity", hue='score', ax=ax)
            ax.set_ylim([-1, 1])
    plt.tight_layout()
    plt.suptitle(f'{velocity_df.iloc[n].num_units} units')
    plt.show()

    # fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    # for i, ax in enumerate(axes.flatten()):
    #     if options[i] in ['phase', 'block', 'group']:
    #         sns.boxplot(data=df_non_start, x=options[i], y="intercept", ax=ax)
    #     else:
    #         sns.scatterplot(data=df_non_start, x=options[i], y="intercept", hue='score', ax=ax)
    #     ax.set_ylim([-100, 100])
    # plt.tight_layout()
    # plt.suptitle(f'{velocity_df.iloc[n].num_units} units')
    # plt.show()

    # # slope intercept relationship
    # fig, axes = plt.subplots(2, 3, figsize=(15, 12))
    # for i, ax in enumerate(axes.flatten()):
    #     if options[i] in ['phase', 'block']:
    #         sns.boxplot(data=df_non_start, x=options[i], y="log_velocity", ax=ax)
    #     else:
    #         sns.scatterplot(data=df_non_start, x='intercept', y='log_velocity', hue='score', ax=ax)
    #         ax.set_xlim([-100, 100])
    #     ax.set_ylim([-1, 1])
    # plt.tight_layout()
    # plt.suptitle(f'{velocity_df.iloc[n].num_units} units')
    # plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    for i, ax in enumerate(axes.flatten()):
        if options[i] in ['phase', 'block', 'group']:
            sns.boxplot(data=df_best, x=options[i], y="log_velocity", ax=ax)
        else:
            sns.regplot(data=df_best, x=options[i], y="log_velocity", ax=ax)
            # sns.scatterplot(data=df_early, x=options[i], y="log_velocity", hue='trial_time', ax=ax)
        ax.set_ylim([-1, 1])
    plt.tight_layout()
    plt.suptitle(f'{velocity_df.iloc[n].num_units} units')
    plt.show()

    # sns.scatterplot(data=session_df, x="intercept", y="leave_from_reward", hue="trial_time")
    # plt.title('Intercept Plot')
    # plt.show()

    # fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    # for i, ax in enumerate(axes.flatten()):
    #     if options[i] in ['phase', 'block']:
    #         sns.boxplot(data=session_df, x=options[i], y="intercept", ax=ax)
    #     else:
    #         sns.scatterplot(data=session_df, x=options[i], y="intercept", ax=ax)
    # plt.tight_layout()
    # plt.show()
    return True


if __name__ == '__main__':
    make_plots('ES039_2024-03-08_bot144_1_g0')
    # make_recurrence_plots('ES039_2024-03-08_bot144_1_g0')
