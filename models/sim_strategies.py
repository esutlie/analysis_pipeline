import numpy as np
import backend
from constants import constants_dict as constants
import matplotlib.pyplot as plt

colors = backend.get_color_sets()


def sim_multi(num):
    x = np.linspace(0, 50, 501)
    c = 8
    s = 1
    return [x[np.where((backend.decay_function(x, c, s) / 10 - np.random.random(len(x))) > 0)[0]] for _ in range(num)]


def sim_strategy(strategy=None):
    if strategy not in ['from_entry', 'from_reward', 'combo']:
        strategy = 'from_entry'
    optimal_times = []
    optimal_rewards = []
    for block in range(2):
        rewards = sim_multi(500)
        if strategy == 'from_entry':
            pi = mean_time = np.linspace(0, 20, 201)
            last_rewards = np.array([np.sum(np.subtract.outer(mean_time, trial) > 0, axis=1) for trial in rewards])
            mean_reward = np.average(last_rewards, axis=0)
            leave_times = np.array([mean_time] * len(rewards))
            total_rewards = last_rewards
        else:
            if strategy == 'from_reward':
                pi = np.linspace(0, 10, 101)
                bounds = [np.concatenate([np.array([0]), times, np.array([100])]) for times in rewards]
                intervals = [times[1:] - times[:-1] for times in bounds]
                where = [[np.where(policy < interval)[0] for policy in pi] for interval in intervals]
                last_rewards = np.array([[min(value) if len(value) else np.nan for value in trial] for trial in where])
                leave_times = np.array([trial[last_rewards[i]] + pi for i, trial in enumerate(bounds)])
                mean_reward = np.average(last_rewards, axis=0)
                mean_time = np.average(leave_times, axis=0)
                total_rewards = last_rewards
            elif strategy == 'combo':
                pi_reward = np.linspace(.1, 10, 100)
                pi_entry = np.linspace(.1, 10, 100)
                bounds = [[np.concatenate([np.array([policy]), times[np.where(times > policy)[0]], np.array([100])])
                           for policy in pi_entry] for times in rewards]
                intervals = [[times[1:] - times[:-1] for times in trial] for trial in bounds]
                where = [[[np.where(policy < interval)[0] for policy in pi_reward] for interval in trial] for trial in
                         intervals]
                last_rewards = np.array(
                    [[[min(value) if len(value) else np.nan for value in group] for group in trial] for trial in where])
                leave_times = np.array(
                    [[group[last_rewards[i, j]] + pi_reward for j, group in enumerate(trial)] for i, trial in
                     enumerate(bounds)])
                total_rewards = np.array(
                    [[np.sum(np.subtract.outer(group, rewards[i]) >= 0, axis=1) for j, group in enumerate(trial)] for
                     i, trial
                     in enumerate(leave_times)])
                mean_reward = np.average(total_rewards, axis=0)
                mean_time = np.average(leave_times, axis=0)
            else:
                raise Exception

        total_time = (mean_time + constants['background_time'][block]
                      + constants['travel_time'] * 2 + constants['consumption_time'])
        total_reward = mean_reward + constants['background_reward'][block]
        rate = total_reward / total_time
        max_x = np.argmax(rate)
        if strategy == 'combo':
            max_pi_entry, max_pi_reward = np.unravel_index(max_x, np.shape(rate))
            plt.imshow(rate.T, extent=[min(pi_entry), max(pi_entry), min(pi_reward), max(pi_reward)], origin='lower')
            plt.ylabel('From Reward Policy (seconds)')
            plt.xlabel('From Entry Policy (seconds)')
            plt.title(f'Multi Reward Paradigm b{block} ({strategy})')
            plt.show()
            inds = [0, 19, 39, 59, 79, 99]
            for i in inds:
                plt.plot(rate[i])
            plt.legend([f'Entry Policy: {pi_entry[i]} sec' for i in inds])
            plt.ylabel('Rate')
            plt.xlabel('From Reward Policy (seconds)')
            plt.title(f'Multi Reward Paradigm ({strategy})')
            plt.show()
        else:
            optimal_rewards.append(total_rewards[:, max_x])
            optimal_times.append(leave_times[:, max_x])
            plt.vlines(pi[max_x], 0, rate[max_x], color=colors['set2'][block])
            plt.plot(pi, rate, c=colors['set2'][block])
    if strategy == 'combo':
        pass
    else:
        plt.ylim([0, 1])
        plt.xlim([min(pi), max(pi)])
        plt.ylabel('Reward Rate (rewards per second)')
        plt.xlabel('Leave Time Policy (seconds)')
        plt.title(f'Multi Reward Paradigm ({strategy})')
        plt.show()
    return optimal_times, optimal_rewards


if __name__ == '__main__':
    sim_strategy('from_reward')
    sim_strategy('from_entry')
    # sim_strategy('combo')
