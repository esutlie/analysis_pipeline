import numpy as np
import pandas as pd
from backend.get_trial_events import get_trial_events


def get_trial_group(events):
    pi_events_shortened = events[(events.phase != 'setup') & ~np.isnan(events.trial)]
    phases = pi_events_shortened.phase.values
    transition_trials = pi_events_shortened.iloc[np.where(phases != np.roll(phases, 1))[0]].trial.values
    block_idx = [np.sum(trial >= transition_trials) for trial in np.unique(pi_events_shortened.trial)]
    return pd.Series(block_idx, index=np.unique(pi_events_shortened.trial))


def recent_rate_kernel(x):
    return np.exp(-x / 30)


def optimal_kernel(x, single=False):
    if not single:
        return -8 * np.log(x)
    else:
        reward_level = 0.5994974874371859  # cumulative for an 8 reward version
        starting_prob = 0.1301005025125628
        return np.log(((1 / x) - reward_level) / (1 - reward_level)) * (reward_level / starting_prob)


def make_intervals_df(pi_events, report_single=False):
    all_reward_times = pi_events[(pi_events.key == 'reward') & (pi_events.value == 1)].time.values
    single = bool(len(pi_events[pi_events.key == 'reward_time']))
    recent_rewards = []
    recent_rates = []
    recent_rate_t = np.sum(recent_rate_kernel(np.linspace(.5, 59.5, 60)))
    interval_columns = ['start', 'end', 'trial', 'phase', 'trial_time', 'leave_from_reward', 'leave_from_entry',
                        'optimal_leave', 'single_task']
    intervals_df = pd.DataFrame(columns=interval_columns)
    for entry_time, exit_time, reward_list, trial, _ in zip(*get_trial_events(pi_events)):
        reward_list = reward_list[reward_list < exit_time]
        all_interval_starts = np.concatenate([[entry_time], reward_list])
        interval_starts = [entry_time] + reward_list.tolist()
        interval_ends = reward_list.tolist() + [exit_time]
        interval_trial = [trial] * len(interval_starts)
        interval_phase = [0] + [1] * (len(reward_list) - 1) + [2] if len(reward_list) else [3]
        trial_time = all_interval_starts - entry_time
        leave_from_reward = [exit_time - reward_list[-1]] * len(interval_starts) if len(reward_list) else [np.nan]
        leave_from_entry = [exit_time - entry_time] * len(interval_starts)
        rewards = entry_time - all_reward_times
        recent_rewards_arr = np.array([r for r in rewards if 0 < r < 60])
        recent_rate = np.sum(recent_rate_kernel(recent_rewards_arr)) / recent_rate_t
        optimal_leave = [optimal_kernel(recent_rate, single=single)] * len(interval_starts)
        single_true = [single] * len(interval_starts)
        for t in all_interval_starts:
            rewards = t - all_reward_times
            recent_rewards_arr = np.array([r for r in rewards if 0 < r < 60])
            recent_rewards.append(recent_rewards_arr)
            recent_rates.append(np.sum(recent_rate_kernel(recent_rewards_arr)) / recent_rate_t)
        interval_data = np.array(
            [interval_starts, interval_ends, interval_trial, interval_phase, trial_time,
             leave_from_reward, leave_from_entry, optimal_leave, single_true]).T
        intervals_df = pd.concat([intervals_df, pd.DataFrame(interval_data, columns=interval_columns)])
    intervals_df.reset_index(inplace=True, drop=True)

    trial_blocks = pi_events.groupby('trial').phase.max()
    intervals_df['block'] = trial_blocks.loc[intervals_df.trial.values].values
    trial_group = get_trial_group(pi_events)
    intervals_df['group'] = trial_group.loc[intervals_df.trial.values].values
    intervals_df['recent_rewards'] = recent_rewards
    intervals_df['recent_rate'] = recent_rates
    if report_single:
        single = bool(len(pi_events[pi_events.key == 'reward_time']))
        return intervals_df, single
    return intervals_df
