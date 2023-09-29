import random

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


def sim_strategy(strategy=None, accuracy=None, plot=False):
    if strategy not in ['from_entry', 'from_reward', 'combo']:
        strategy = 'from_entry'
    optimal_times = []
    optimal_rewards = []
    max_rates = []
    policies = []
    for block in range(2):
        rewards = sim_multi(500000)
        if strategy == 'from_entry':
            pi = mean_time = np.linspace(0, 20, 201)
            if accuracy:
                var = np.random.normal(0, accuracy * mean_time / 10, size=(len(rewards), len(mean_time)))
                last_rewards = np.array(
                    [np.sum(np.subtract.outer(mean_time + var[i], trial) > 0, axis=1) for i, trial in
                     enumerate(rewards)])
                mean_reward = np.average(last_rewards, axis=0)
                leave_times = np.array([mean_time] * len(rewards)) + var
                total_rewards = last_rewards
            else:
                last_rewards = np.array([np.sum(np.subtract.outer(mean_time, trial) > 0, axis=1) for trial in rewards])
                mean_reward = np.average(last_rewards, axis=0)
                leave_times = np.array([mean_time] * len(rewards))
                total_rewards = last_rewards
        else:
            if strategy == 'from_reward':
                pi = np.linspace(0, 10, 101)
                bounds = [np.concatenate([np.array([0]), times, np.array([100])]) for times in rewards]
                intervals = [times[1:] - times[:-1] for times in bounds]

                if accuracy:
                    var = np.random.normal(0, accuracy * pi / 10, size=(len(intervals), len(pi)))
                    where = [[np.where(policy + var[i, j] < interval)[0] for j, policy in enumerate(pi)] for
                             i, interval in enumerate(intervals)]
                    last_rewards = np.array(
                        [[min(value) if len(value) else np.nan for value in trial] for trial in where])
                    leave_times = np.array([trial[last_rewards[i]] + pi + var[i] for i, trial in enumerate(bounds)])
                else:
                    where = [[np.where(policy < interval)[0] for policy in pi] for interval in intervals]
                    last_rewards = np.array(
                        [[min(value) if len(value) else np.nan for value in trial] for trial in where])
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
        max_rate = np.max(rate)
        policies.append(pi[max_x])
        max_rates.append(max_rate)
        print(f'Rate max ({strategy}): {max_rate}  block {block}')
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
        if plot:
            plt.show()
        else:
            plt.close()
    return optimal_times, optimal_rewards, max_rates, policies


def webers_effect():
    accuracy_range = np.linspace(.1, 9, 10)
    reward_accuracies = []
    entry_accuracies = []
    for a in accuracy_range:
        _, _, max_rate, policy = sim_strategy('from_reward', accuracy=a)
        reward_accuracies.append(max_rate)
        _, _, max_rate, policy = sim_strategy('from_entry', accuracy=a)
        entry_accuracies.append(max_rate)
    reward_accuracies = np.array(reward_accuracies)
    entry_accuracies = np.array(entry_accuracies)

    plt.plot(accuracy_range, reward_accuracies[:, 0])
    plt.plot(accuracy_range, entry_accuracies[:, 0])
    plt.ylim([0, plt.gca().get_ylim()[1] + .1])
    plt.xlabel('timing error coefficient')
    plt.ylabel('maximum reward rate')
    plt.title('Low Rate Block')
    backend.save_fig(plt.gcf(), 'low_block.png')
    plt.show()

    plt.plot(accuracy_range, reward_accuracies[:, 1])
    plt.plot(accuracy_range, entry_accuracies[:, 1])
    plt.ylim([0, plt.gca().get_ylim()[1]])
    plt.ylim([0, plt.gca().get_ylim()[1] + .1])
    plt.xlabel('timing error coefficient')
    plt.ylabel('maximum reward rate')
    plt.title('High Rate Block')
    backend.save_fig(plt.gcf(), 'high_block.png')
    plt.show()


if __name__ == '__main__':
    webers_effect()
    # sim_strategy('from_reward', accuracy=None)
    # sim_strategy('from_entry', accuracy=None)
    # print()
    # sim_strategy('from_reward', accuracy=1)
    # sim_strategy('from_entry', accuracy=1)
    # print()
    # sim_strategy('from_reward', accuracy=2)
    # sim_strategy('from_entry', accuracy=2)
    # print()
    # sim_strategy('from_reward', accuracy=3)
    # sim_strategy('from_entry', accuracy=3)
    # sim_strategy('combo')
