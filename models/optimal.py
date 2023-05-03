import backend
import numpy as np
import matplotlib.pyplot as plt
from constants import constants_dict
from sim_strategies import sim_strategy


def multi_reward(constants):
    colors = backend.get_color_sets()
    optimal_times, optimal_rewards = sim_strategy('from_reward')
    pi = np.linspace(0, 20, 200)
    c = 8
    s = 1
    F = backend.decay_function_cumulative
    best = []
    max_rate = []
    for i in range(2):
        total_time = constants['background_time'][i] + pi + constants['travel_time'] * 2 + constants['consumption_time']
        total_reward = constants['background_reward'][i] + F(pi, c, s)
        rate = total_reward / total_time
        max_x = np.argmax(rate)
        plt.vlines(pi[max_x], 0, rate[max_x], color=colors['set2'][i])
        plt.plot(pi, rate, c=colors['set2'][i])
        # (counts, bins) = np.histogram(optimal_times[i], bins=np.linspace(0, 20, 50))
        # plt.hist(bins[:-1], bins, weights=counts / len(optimal_times) * 2, color=colors['set2'][i])
        best.append(pi[max_x])
        max_rate.append(rate[max_x])

    plt.ylim([0, 1])
    plt.xlim([min(pi), max(pi)])
    plt.ylabel('Reward Rate (rewards per second)')
    plt.xlabel('Leave Time Policy (seconds)')
    plt.title('Multi Reward Paradigm')
    plt.show()
    return best, max_rate


def single_reward(constants, c=.602, s=.1329, r=8, plot=True):
    colors = backend.get_color_sets()
    pi = np.linspace(0, 20, 200)  # Policies
    # c = .6  # to the right a little up
    # s = .13  # up and together
    # reward_value = 7  # up and to the right
    # c = .55
    # s = .12
    # reward_value = 8
    F = backend.decay_function_cumulative
    Ft = backend.weighted_time_function
    best = []
    max_rate = []
    for i in range(2):
        total_reward = constants['background_reward'][i] + F(pi, c, s) * r
        inside_time = Ft(pi, c, s) + (1 - F(pi, c, s)) * pi + F(pi, c, s) * (
                constants['consumption_time'] + r * .1 - .1)
        outside_time = constants['background_time'][i] + constants['travel_time'] * 2 + constants['consumption_time']
        total_time = inside_time + outside_time
        rate = total_reward / total_time
        max_x = np.argmax(rate)
        if plot:
            plt.vlines(pi[max_x], 0, rate[max_x], color=colors['set2'][i])
            plt.plot(pi, rate, c=colors['set2'][i])
        best.append(pi[max_x])
        max_rate.append(rate[max_x])
    if not plot:
        return best, max_rate
    plt.ylim([0, 1])
    plt.xlim([min(pi), max(pi)])
    plt.ylabel('Reward Rate (rewards per second)')
    plt.xlabel('Leave Time Policy (seconds)')
    plt.title(f'Single Reward Paradigm c:{c:.4f} s:{s:.4f} r:{r}')
    plt.show()
    plt.plot(pi, backend.decay_function_cumulative(pi, c, s))
    plt.show()


def solve_fix(constants, fix='policy'):
    starts = np.linspace(.09, .3, 200)
    cums = np.linspace(.5, .8, 200)
    rewards = [5, 6, 7, 8]
    bests = np.zeros([len(starts), len(cums), len(rewards)])
    multi_best, multi_max_rate = multi_reward(constants)
    for i, s in enumerate(starts):
        for j, c in enumerate(cums):
            for k, r in enumerate(rewards):
                best, max_rate = single_reward(constants, c, s, r, plot=False)
                if fix == 'rate':
                    if (best[0] > multi_best[0]) or (best[1] > multi_best[1]) or best[0] == 0 or best[1] == 0:
                        dif = 1
                    else:
                        dif = (max_rate[0] - multi_max_rate[0]) ** 2 + (max_rate[1] - multi_max_rate[1]) ** 2
                else:  # fix == 'policy'
                    dif = (best[0] - multi_best[0]) ** 2 + (best[1] - multi_best[1]) ** 2
                bests[i, j, k] = dif
    coords = []
    for i in range(len(bests[0, 0])):
        coords.append(list(np.unravel_index(np.argmin(bests[:, :, i]), np.shape(bests[:, :, i]))) + [i])
    inds = np.array(coords)
    # inds = np.array(np.where(bests < .00001))
    for [i, j, k] in inds:
        print([i, j, k])
        print(f'start = {starts[i]}')
        print(f'cumulative = {cums[j]}')
        print(f'reward = {rewards[k]}')
        single_reward(constants, cums[j], starts[i], rewards[k])
    print('done')


if __name__ == '__main__':
    multi_reward(constants_dict)
    # single_reward(constants_dict)
    # solve_fix(constants_dict)
