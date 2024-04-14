import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import backend


def task_compare_behavior():
    master_df = pd.read_pickle(os.path.join(os.path.dirname(os.getcwd()), '_master', 'data', 'master_data.pkl'))
    entry_leave_df = master_df[((master_df.single_task == 0) & ((master_df.phase == 0) | (master_df.phase == 3))) |
                               ((master_df.single_task == 1) & (master_df.phase == 3))]
    # sns.histplot(data=entry_leave_df[entry_leave_df.single_task == 1], x='leave_from_entry')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, ax in enumerate(axes):
        # sns.boxplot(data=entry_leave_df, x='single_task', y="leave_from_entry")
        df = entry_leave_df[entry_leave_df.single_task == i]
        sns.scatterplot(data=df, x='optimal_leave', y="leave_from_entry",
                        hue='block', ax=ax, s=2, hue_order=np.sort(df.block.unique()), palette='Set2')
        # x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        ax.plot([-10, 30], [-10, 30], c='grey', zorder=-1)
        ax.set_ylim([0, 22 if i == 0 else 17])
        ax.set_xlim([0, 22 if i == 0 else 17])
        ax.set_aspect(1)
        ax.set_title('Single Reward Task' if i == 1 else 'Multi Reward Task')
        ax.set_ylabel('Actual Leave Time')
        ax.set_xlabel('Optimal Leave Time (by trial)')
    plt.show()

    sns.boxplot(entry_leave_df, x='single_task', y='leave_from_entry', hue='block', palette='Set2')
    plt.xlabel('')
    plt.ylabel('Leave Time (sec)')
    plt.xticks(plt.gca().get_xticks(), ['Multi Reward', 'Single Reward'])
    plt.show()


def examples():
    colors = backend.get_color_sets()['set2'][:2]
    master_df = pd.read_pickle(os.path.join(os.path.dirname(os.getcwd()), '_master', 'data', 'master_data.pkl'))
    fig, axes = plt.subplots(2, 1, figsize=(7, 5))
    example_sessions = ['ES039_2024-02-29_bot144_1_g0', 'ES037_2023-12-15_bot144_0_g0']
    for i, ax in enumerate(axes):
        df = master_df[master_df.session == example_sessions[i]]
        if df.single_task.iloc[0] == 1:
            df = df[df.phase == 3]
        df['block_rate'] = [float(val) for val in df.block.values]
        leave_times = df.groupby('trial').mean().leave_from_entry.to_numpy()
        optimal_times = df.groupby('trial').mean().optimal_leave.to_numpy()
        block = np.round(df.groupby('trial').mean().block_rate.to_numpy(), 1)
        blocks = np.sort(np.unique(block))
        x = np.arange(len(leave_times))
        for j in range(len(leave_times)):
            c = colors[np.where(block[j] == blocks)[0][0]]
            ax.plot([x[j], x[j]], [0, leave_times[j]], c=c, linewidth=2)
        ax.plot(x, optimal_times, c=(.8, .8, .8))
        if df.single_task.iloc[0] == 1:
            ax.set_xlabel('Trial Number (excluding rewarded)')
        else:
            ax.set_xlabel('Trial Number')
        ax.set_ylabel('Leave Time (sec)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    task_compare_behavior()
    # examples()
