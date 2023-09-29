from datetime import date
import os
from tkinter import *
import time
from os import walk
import pandas as pd
from csv import DictReader, reader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import backend
from get_entry_exit import get_entry_exit


def session_summary_axis_settings(axes, max_trial):
    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.get_yaxis().set_visible(False)
        ax.set_ylim([-1, max_trial + 1])
        ax.set_xlim([0, 20])
        ax.invert_yaxis()
        ax.set_ylabel('Trial')
        ax.set_xlabel('Time (sec)')


def session_summary(data, title):
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=[10, 10])
    port_palette = sns.color_palette('Set1')
    block_palette = sns.color_palette('Set2')
    start = data.value == 1
    end = data.value == 0
    head = data.key == 'head'
    lick = data.key == 'lick'
    reward = data.key == 'reward'
    port1 = data.port == 1
    port2 = data.port == 2
    max_trial = data.trial.max()

    bg_rectangles = []
    exp_rectangles_in_bg = []
    exp_rectangles = []
    block1_rectangles = []
    block2_rectangles = []
    bg_reward_events = []
    exp_reward_events = []
    bg_lick_events = []
    exp_lick_events = []
    bg_lengths = []
    exp_lengths = []
    trial_blocks = data.groupby(['trial'])['phase'].agg(pd.Series.mode)
    blocks = data.phase.unique()
    blocks.sort()
    for trial in data.trial.unique():
        if np.isnan(trial):
            continue
        is_trial = data.trial == trial
        try:
            trial_start = data[is_trial & start & (data.key == 'trial')].session_time.values[0]
            trial_middle = data[is_trial & end & (data.key == 'LED') & port2].session_time.values[0]
            trial_end = data[is_trial & end & (data.key == 'trial')].session_time.values[0]
        except IndexError:
            continue

        bg_rewards = data[is_trial & start & port2 & reward].session_time.values
        exp_rewards = data[is_trial & start & port1 & reward].session_time.values
        bg_licks = data[is_trial & start & lick & (data.session_time < trial_middle)].session_time.values
        exp_licks = data[is_trial & start & lick & (data.session_time > trial_middle)].session_time.values

        bg_lengths.append(trial_middle - trial_start)
        exp_lengths.append(trial_end - trial_middle)

        bg_entries, bg_exits, exp_entries, exp_exits, early_exp_entries, early_exp_exits = get_entry_exit(data, trial)
        bg_intervals = list(zip(bg_entries, bg_exits))
        exp_intervals = list(zip(exp_entries, exp_exits))
        early_exp_intervals = list(zip(early_exp_entries, early_exp_exits))
        for [s, e] in bg_intervals:
            bg_rectangles.append(Rectangle((s - trial_start, trial), e - s, .7))
        for [s, e] in early_exp_intervals:
            exp_rectangles_in_bg.append(Rectangle((s - trial_start, trial), e - s, .7))
        for [s, e] in exp_intervals:
            exp_rectangles.append(Rectangle((s - trial_middle, trial), e - s, .7))
        if np.where(blocks == trial_blocks.loc[trial])[0][0] == 0:
            block1_rectangles.append(Rectangle((0, trial), 100, 1))
        else:
            block2_rectangles.append(Rectangle((0, trial), 100, 1))
        bg_reward_events.append(bg_rewards - trial_start)
        exp_reward_events.append(exp_rewards - trial_middle)
        bg_lick_events.append(bg_licks - trial_start)
        exp_lick_events.append(exp_licks - trial_middle)

    alpha = .5
    pc_b1 = PatchCollection(block1_rectangles, facecolors=block_palette[0], alpha=alpha)
    pc_b2 = PatchCollection(block2_rectangles, facecolors=block_palette[1], alpha=alpha)
    ax1.add_collection(pc_b1)
    ax1.add_collection(pc_b2)
    pc_b12 = PatchCollection(block1_rectangles, facecolors=block_palette[0], alpha=alpha)
    pc_b22 = PatchCollection(block2_rectangles, facecolors=block_palette[1], alpha=alpha)
    ax2.add_collection(pc_b12)
    ax2.add_collection(pc_b22)

    pc_bg = PatchCollection(bg_rectangles, edgecolor=port_palette[0], facecolor='w', alpha=1)
    ax1.add_collection(pc_bg)

    pc_exp_bg = PatchCollection(exp_rectangles_in_bg, edgecolor=port_palette[1], facecolor='w', alpha=1)
    ax1.add_collection(pc_exp_bg)

    pc_exp = PatchCollection(exp_rectangles, edgecolor=port_palette[1], facecolor='w', alpha=1)
    ax2.add_collection(pc_exp)

    offsets = np.array(list(range(len(bg_reward_events)))) + 1.4
    ax1.eventplot(bg_reward_events, color='purple', linelengths=.62, lineoffsets=offsets)
    offsets = np.array(list(range(len(exp_reward_events)))) + 1.4
    ax2.eventplot(exp_reward_events, color='purple', linelengths=.62, lineoffsets=offsets)

    light = [.8, .7, .8]
    dark = [.2, .2, .2]
    offsets = np.array(list(range(len(bg_lick_events)))) + 1.4
    ax1.eventplot(bg_lick_events, color=light, linelengths=.25, lineoffsets=offsets)
    offsets = np.array(list(range(len(exp_lick_events)))) + 1.4
    ax2.eventplot(exp_lick_events, color=light, linelengths=.25, lineoffsets=offsets)

    session_summary_axis_settings([ax1, ax2], max_trial)
    plt.suptitle(title)
    backend.save_fig(fig, title, sub_folder=os.path.join('session_summary', title[:5]))

    plt.show()


def make_session_summaries(photometry=False):
    files = backend.get_session_list(photometry=photometry)
    for session in files:
        _, pi_events, _ = backend.load_data(session, photometry=photometry)
        session_summary(pi_events, session)


if __name__ == '__main__':
    make_session_summaries()
