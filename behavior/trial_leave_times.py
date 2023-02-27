import numpy as np
import matplotlib.pyplot as plt
import backend
from behavior.get_trial_events import get_trial_events
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from matplotlib.colors import ListedColormap
from statsmodels.formula.api import ols


def trial_leave_times(file_list, data_list, save=False, data_only=False):
    session_columns = ['entry_times', 'exit_times', 'last_reward_times', 'trial_numbers', 'block_labels', 'day']
    full_columns = session_columns + ['mouse', 'from_entry', 'from_reward']
    mouse_list = [file[:5] for file in file_list]
    mouse_names = np.unique(mouse_list)
    full_data_frame = pd.DataFrame(columns=full_columns)
    for mouse in mouse_names:
        mouse_inds = np.where(np.array(mouse_list) == mouse)[0]
        mouse_df = pd.DataFrame(columns=full_columns)
        count = 0
        for i, session in enumerate(data_list):
            if i in mouse_inds:
                entry_times, exit_times, reward_times, trial_numbers = get_trial_events(session,
                                                                                        include_unrewarded=False)
                last_reward_times = [max([entry_times[i]] + list(trial_rewards)) for i, trial_rewards in
                                     enumerate(reward_times)]
                reward_count = [len(trial_rewards) for trial_rewards in reward_times]

                reward_interval = [trial_rewards[-1] - trial_rewards[-2] if len(trial_rewards) >= 2 else 0 for
                                   trial_rewards in reward_times]
                block_labels = [float(session[session.trial == trial].phase.iloc[0]) for trial in trial_numbers]
                day = [count] * len(block_labels)
                mouse_name = [mouse] * len(block_labels)
                count += 1
                session_df = pd.DataFrame(
                    np.array([entry_times, exit_times, last_reward_times, trial_numbers, block_labels, day]).T,
                    columns=session_columns)
                session_df['mouse'] = mouse_name
                session_df['from_entry'] = session_df.exit_times - session_df.entry_times
                session_df['from_reward'] = session_df.exit_times - session_df.last_reward_times
                session_df['reward_count'] = reward_count
                session_df['reward_interval'] = reward_interval
                mouse_df = pd.concat([mouse_df, session_df])
        full_data_frame = pd.concat([full_data_frame, mouse_df])
        if not data_only:
            # leave_times_per_session_plot(mouse_df, 'from_entry', mouse=mouse, save_plot=save)
            # leave_times_per_session_plot(mouse_df, 'from_reward', mouse=mouse, save_plot=save)
            mouse_df['last_reward_from_entry'] = mouse_df.last_reward_times - mouse_df.entry_times
            mouse_df['reward_rate'] = mouse_df.reward_count / mouse_df.last_reward_from_entry
            mouse_df['reward_rate'] = mouse_df['reward_rate'].fillna(0)
            mouse_df['adjusted_reward_rate'] = mouse_df.reward_count / backend.decay_function_cumulative(
                mouse_df.last_reward_from_entry)
            mouse_df['adjusted_reward_rate'] = mouse_df['adjusted_reward_rate'].fillna(0)

            # compare_plot(mouse_df, 'last_reward_from_entry', 'from_reward', mouse=mouse, save_plot=save,
            #              x_lim=[0, 18], y_lim=[0, 10],
            #              x_label='Time of Final Reward (sec)', y_label='Leave Time after Final Reward (sec)')
            # compare_plot(mouse_df, 'last_reward_times', 'from_reward', mouse=mouse, save_plot=save,
            #              x_lim=[0, mouse_df.exit_times.max()], y_lim=[0, 10],
            #              x_label='Time in Session (sec)', y_label='Leave Time after Final Reward (sec)')
            compare_plot(mouse_df, 'adjusted_reward_rate', 'from_reward', mouse=mouse, save_plot=save,
                         x_lim=[0, 8], y_lim=[0, 10],
                         x_label='Rate of Reward in Trial (reward/sec)', y_label='Leave Time after Final Reward (sec)')
            # compare_plot(mouse_df, 'reward_interval', 'from_reward', mouse=mouse, save_plot=save,
            #              x_lim=[0, mouse_df.reward_interval.max() + 1], y_lim=[0, 10],
            #              x_label='Previous Reward-Reward Interval (sec)', y_label='Leave Time after Final Reward (sec)')
            # compare_plot(mouse_df, 'last_reward_from_entry', 'adjusted_reward_rate', key3='from_reward', mouse=mouse,
            #              save_plot=save,
            #              x_lim=[0, mouse_df.last_reward_from_entry.max() + 1],
            #              y_lim=[0, mouse_df.adjusted_reward_rate.max() + 1], z_lim=[0, 10],
            #              x_label='Time of Final Reward (sec)', y_label='Adjusted Reward Rate (reward/sec)',
            #              z_label='Leave Time after Final Reward (sec)')

    # if not data_only:
    #     leave_times_plot(full_data_frame, 'from_entry', title='Leave Time From Port Entry',
    #                      legend_placement='lower right',
    #                      save_plot=save)
    #     leave_times_plot(full_data_frame, 'from_reward', title='Leave Time From Last Reward',
    #                      legend_placement='upper right', save_plot=save)

    return full_data_frame


def compare_plot(mouse_df, key1, key2, key3=None, mouse=None, title=None, save_plot=False,
                 x_lim=None, y_lim=None, z_lim=None,
                 x_label=None, y_label=None, z_label=None):
    if x_label is None:
        x_label = key1
    if y_label is None:
        y_label = key2
    if z_label is None:
        z_label = key3
    color_sets = backend.get_color_sets()
    if key3 is None:
        fig, ax = plt.subplots(1, 1, figsize=[8, 6])
        sns.scatterplot(data=mouse_df, x=key1, y=key2, hue='block_labels', legend='auto', ax=ax,
                        palette=color_sets['set2'][:2])
        blocks = mouse_df.block_labels.unique()
        blocks.sort()
        for i, block in enumerate(blocks):
            block_df = mouse_df[mouse_df.block_labels == block]
            regression = LinearRegression()
            regression.fit(np.expand_dims(block_df[key1].values, axis=1), np.expand_dims(block_df[key2].values, axis=1))
            x = np.linspace(0, block_df[key1].max())
            y = regression.predict(np.expand_dims(x, axis=1))
            ax.plot(x, y, '-', c=color_sets['set2_med_dark'][i])
        model = ols(f'{key2} ~ block_labels + {key1}', data=mouse_df).fit()
        print(model.summary())
    else:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(mouse_df[key1].values, mouse_df[key2].values, mouse_df[key3].values, c=mouse_df.block_labels,
                   cmap=ListedColormap(color_sets['set2'].as_hex()[:2]))
        model = ols(f'{key3} ~ block_labels + {key1} + {key2}', data=mouse_df).fit()
        print(model.summary())

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if z_label:
        ax.set_zlabel(z_label)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    if z_lim:
        ax.set_zlim(z_lim)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(mouse)

    if save_plot:
        backend.save_fig(fig, f'{mouse}_compare_plot.png')
    else:
        plt.show()


def leave_times_per_session_plot(mouse_df, key, mouse=None, save_plot=False):
    color_sets = backend.get_color_sets()
    num_days = mouse_df.day.max() + 1
    fig_x = .5 * num_days
    fig, ax = plt.subplots(1, 1, figsize=[fig_x, 5])
    ax.set_title(mouse)
    sns.lineplot(x="day", y=key,
                 hue="block_labels", legend='auto',
                 data=mouse_df, ax=ax, palette=color_sets['set2'][:2])
    ax.legend(loc='lower right')
    ax.set_xlabel('Recording Session Number')
    ax.set_ylabel(f'Leave Time (sec)')
    ax.set_ylim([0, 16])

    if save_plot:
        backend.save_fig(fig, f'{mouse}_{key}.png')
    else:
        plt.show()


def leave_times_plot(leave_data, key, title=None, legend_placement=None, save_plot=False):
    color_sets = backend.get_color_sets()
    fig, ax = plt.subplots(1, 1, figsize=[8, 6])
    ax.set_title(key)
    sns.boxplot(x="mouse", y=key,
                hue="block_labels",
                data=leave_data, ax=ax, palette=color_sets['set2'][:2])
    sns.swarmplot(x="mouse", y=key,
                  hue="block_labels", dodge=True,
                  data=leave_data, ax=ax, palette=color_sets['set2_dark'][:2], size=2)
    ax.legend(loc='lower right')
    ax.set_xlabel('Mouse')
    ax.set_ylabel(f'Leave Time (sec)')
    ax.set_ylim([0, 16])
    if title:
        ax.set_title(title)
    else:
        ax.set_title(key)

    if legend_placement:
        ax.legend(loc=legend_placement)

    if save_plot:
        backend.save_fig(fig, f'{key}.png')
    else:
        plt.show()


if __name__ == '__main__':
    files = backend.get_session_list()
    data = [backend.load_data(session)[1] for session in files]
    leave_time_df = trial_leave_times(files, data, save=False)
