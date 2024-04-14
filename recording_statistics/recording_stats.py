import numpy as np
import pandas as pd
import backend
import seaborn as sns
import matplotlib.pyplot as plt


def recording_stats():
    columns = ['mouse', 'task', 'row', 'num_units', 'date', 'exploratory']
    df = pd.DataFrame(columns=columns)

    main_row = {
        'ES024': 192,
        'ES025': 170,
        'ES026': 170,
        'ES029': 72,
        'ES030': 170,
        'ES031': 170,
        'ES032': 170,
        'ES037': 170,
        'ES039': 144,
        'ES041': 96,
        'ES042': 168,
        'ES044': 168,
    }
    session_list = backend.get_session_list()
    for session in session_list:
        [mouse, date, num_in_day, row] = backend.unpack_session_name(session)
        if mouse in ['ES026']:
            continue
        spikes, pi_events, cluster_info = backend.load_data(session, photometry=False)
        task = pi_events.task.iloc[10]
        num_units = len(cluster_info)
        exploratory = int(row) != main_row[mouse]
        df = pd.concat([df, pd.DataFrame([[mouse, task, row, num_units, date, exploratory]], columns=columns)])
    exploratory_df = df[df.exploratory == True]
    main_df = df[df.exploratory == False]

    # for this_df in [main_df, exploratory_df]:
    for this_df in [main_df]:
        multi_df = this_df[this_df.task == "cued_no_forgo_forced"]
        single_df = this_df[this_df.task == "single_reward"]
        multi_mice = len(multi_df.mouse.unique())
        single_mice = len(single_df.mouse.unique())
        print(f'# of sessions: multi {len(multi_df)} single {len(single_df)}')
        print(f'# of animals on each task {multi_mice} on multi, {single_mice} on single')
        print(f'# units per session: {this_df.num_units.mean()}')
        print(f'# of units per task: {multi_df.num_units.sum()} for multi, {single_df.num_units.sum()} for single')
        print(f'sessions per animal: {this_df.groupby(["mouse"]).count()}')

        fig, ax = plt.subplots(1, 1, figsize=[8, 5])
        sns.boxplot(this_df, x='mouse', y='num_units', ax=ax)
        plt.ylabel('Number of Units per Session')
        plt.xlabel('Mouse')
        plt.title(f"{this_df.num_units.mean():.1f} Units per Session on Average")
        plt.show()

        sess_count_df = this_df.groupby(["mouse"]).count()
        fig, ax = plt.subplots(1, 1, figsize=[8, 5])
        sns.barplot(sess_count_df, y='row', x=sess_count_df.index, ax=ax)
        plt.ylabel('Number of Units per Session')
        plt.xlabel('Mouse')
        plt.title(f"{this_df.num_units.mean():.1f} Units per Session on Average")
        plt.show()
        print('exploratory')


if __name__ == '__main__':
    recording_stats()
