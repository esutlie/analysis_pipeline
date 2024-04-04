import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns


def make_plots(session):
    result = True
    n = 0
    while result:
        result = velocity_regression(session, n)
        n += 1


    print()


def velocity_regression(session, n):
    master_df = pd.read_pickle(os.path.join(os.path.dirname(os.getcwd()), '_master', 'data', 'master_data.pkl'))
    session_df = master_df[master_df.session == session]
    velocity_df = pd.read_pickle(os.path.join('results', session + '_recurrence.pkl'))
    if n >= len(velocity_df):
        return False
    [velocity, intercept] = velocity_df.iloc[n].arr
    session_df = session_df.assign(velocity=velocity)
    session_df = session_df.assign(intercept=intercept)
    session_df = session_df.assign(log_velocity=np.log10(velocity))
    options = ['leave_from_reward', 'phase', 'trial_time', 'leave_from_entry', 'block', 'start']

    # sns.scatterplot(data=session_df, x="velocity", y="leave_from_reward", hue="trial_time")
    # plt.title('Velocity Plot')
    # plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    for i, ax in enumerate(axes.flatten()):
        if options[i] in ['phase', 'block']:
            sns.boxplot(data=session_df, x=options[i], y="log_velocity", ax=ax)
        else:
            sns.scatterplot(data=session_df, x=options[i], y="log_velocity", ax=ax)
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
    make_plots('ES039_2024-02-28_bot144_1_g0')
