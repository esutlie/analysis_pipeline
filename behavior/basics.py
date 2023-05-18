import backend
from behavior.trial_leave_times import trial_leave_times
from datetime import date
import os
import matplotlib.pyplot as plt
import datetime

tasks = ['cued_no_forgo_forced', 'single_reward']


def basics():
    mouse_list = backend.get_directories(backend.get_pi_path())
    for mouse in mouse_list:
        total_rewards(mouse)


def total_rewards(mouse, ax=None):
    sessions = backend.get_directories(os.path.join(backend.get_pi_path(), mouse))
    reward_nums = []
    dates = []
    for session in sessions:
        pi_events, info = backend.load_pi_data(os.path.join(backend.get_pi_path(), mouse, session))
        if info['task'] in tasks and pi_events.time.max() > 1000:
            [head, trial, cue, reward, lick, leave, start, port1, port2] = backend.get_bools(pi_events)
            reward_nums.append(len(pi_events[reward & start]))
            date_time = backend.read_datetime(info['date'] + '_' + info['time'])
            dates.append(date_time)
    if not len(reward_nums):
        return None
    if ax is None:
        fig, axes = plt.subplots(1, 1)
    else:
        axes = ax

    axes.scatter(dates, reward_nums)
    backend.set_labels(axes, xlabel='date', ylabel='reward number', title=f'{mouse} Reward Counts')

    if ax is None:
        plt.show()


if __name__ == '__main__':
    basics()
