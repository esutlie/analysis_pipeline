import numpy as np
import matplotlib.pyplot as plt
import backend
from behavior import *
import pandas as pd
import seaborn as sns

set2 = sns.color_palette('Set2')
set2_dark = [[c * .5 for c in sublist] for sublist in set2]
colorblind = sns.color_palette('colorblind')


def trial_leave_times(file_list, data_list, save=False):
    session_columns = ['entry_times', 'exit_times', 'trial_numbers', 'block_labels', 'day']
    full_columns = session_columns + ['mouse', 'leave time']
    mouse_list = [file[:5] for file in file_list]
    mouse_names = np.unique(mouse_list)
    full_data_frame = pd.DataFrame(columns=full_columns)
    for mouse in mouse_names:
        mouse_inds = np.where(np.array(mouse_list) == mouse)[0]
        mouse_df = pd.DataFrame(columns=full_columns)
        count = 0
        for i, session in enumerate(data_list):
            if i in mouse_inds:
                entry_times, exit_times, trial_numbers = get_entry_exit(session)
                block_labels = [float(session[session.trial == trial].phase.iloc[0]) for trial in trial_numbers]
                day = [count] * len(block_labels)
                mouse_name = [mouse] * len(block_labels)
                count += 1
                session_df = pd.DataFrame(np.array([entry_times, exit_times, trial_numbers, block_labels, day]).T,
                                          columns=session_columns)
                session_df['mouse'] = mouse_name
                session_df['leave time'] = session_df.exit_times - session_df.entry_times
                mouse_df = pd.concat([mouse_df, session_df])
        full_data_frame = pd.concat([full_data_frame, mouse_df])
        blocks = np.unique(mouse_df.block_labels)
        num_days = mouse_df.day.max() + 1
        fig_x = .5 * num_days
        fig, ax = plt.subplots(1, 1, figsize=[fig_x, 5])
        ax.set_title(mouse)
        legend = 'auto'
        sns.lineplot(x="day", y="leave time",
                     hue="block_labels", legend=legend,
                     data=mouse_df, ax=ax, palette=set2[:2])
        ax.legend(loc='lower right')
        ax.set_xlabel('Recording Session Number')
        ax.set_ylabel('Leave Time (sec)')
        ax.set_ylim([0, 16])

        if save:
            backend.save_fig(fig, f'{mouse}_trial_leave_times.png')
        else:
            plt.show()

    fig, ax = plt.subplots(1, 1, figsize=[8, 6])
    ax.set_title('Leave Times')
    legend = 'auto'
    sns.boxplot(x="mouse", y="leave time",
                hue="block_labels",
                data=full_data_frame, ax=ax, palette=set2[:2])
    # sns.violinplot(x="mouse", y="leave time",
    #               hue="block_labels", legend=legend,
    #               data=full_data_frame, ax=ax, palette=set2[:2])
    sns.swarmplot(x="mouse", y="leave time",
                  hue="block_labels", legend=legend, dodge=True,
                  data=full_data_frame, ax=ax, palette=set2_dark[:2], size=2)
    ax.legend(loc='lower right')
    ax.set_xlabel('Mouse')
    ax.set_ylabel('Leave Time (sec)')
    ax.set_ylim([0, 16])

    if save:
        backend.save_fig(fig, f'average_leave_times.png')
    else:
        plt.show()

    return full_data_frame


if __name__ == '__main__':
    files = backend.get_session_list()
    data = [backend.load_data(session)[1] for session in files]
    trial_leave_times(files, data, save=True)