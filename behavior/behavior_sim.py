import pandas as pd
import backend
from trial_leave_times import leave_times_plot, trial_leave_times


def behavior_sim(save=False):
    files = backend.get_session_list()
    data = [backend.load_data(session)[1] for session in files]
    leave_time_df = trial_leave_times(files, data, save=save, data_only=False)
    mice = leave_time_df.mouse.unique()
    mean_times = pd.DataFrame(columns=['entry', 'reward', 'mouse'])
    for mouse in mice:
        leave_from_entry = leave_time_df[leave_time_df.mouse == mouse].groupby('block_labels')[
            'leave_time_from_entry'].mean()
        leave_from_reward = leave_time_df[leave_time_df.mouse == mouse].groupby('block_labels')[
            'leave_time_from_reward'].mean()
        temp_df = pd.DataFrame()
        temp_df['entry'] = leave_from_entry
        temp_df['reward'] = leave_from_reward
        temp_df['mouse'] = len(leave_from_reward) * [mouse]
        mean_times = pd.concat([mean_times, temp_df])
    mean_from_reward = []
    mean_from_entry = []
    leave_time_df = leave_time_df.reset_index()
    for i in leave_time_df.index:
        mouse = leave_time_df.loc[i].mouse
        block = leave_time_df.loc[i].block_labels
        mean_from_entry.append(mean_times[mean_times.mouse == mouse].entry.loc[block])
        mean_from_reward.append(mean_times[mean_times.mouse == mouse].reward.loc[block])

    leave_time_df['mean_from_entry'] = mean_from_entry
    leave_time_df['mean_from_reward'] = mean_from_reward
    leave_time_df['sim_from_reward'] = leave_time_df.last_reward_times + leave_time_df.mean_from_reward - leave_time_df.entry_times
    leave_time_df['sim_from_entry'] = leave_time_df.entry_times + leave_time_df.mean_from_entry - leave_time_df.last_reward_times
    leave_times_plot(leave_time_df, 'sim_from_reward', title='Leave Time From Port Entry - Simulated',
                     legend_placement='lower right', save_plot=save)
    leave_times_plot(leave_time_df, 'sim_from_entry', title='Leave Time From Last Reward - Simulated',
                     legend_placement='upper right', save_plot=save)


if __name__ == '__main__':
    behavior_sim(save=True)
