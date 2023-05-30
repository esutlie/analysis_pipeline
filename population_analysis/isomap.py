from sklearn.manifold import Isomap
from create_bins_df import create_precision_df
import backend
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression, PoissonRegressor, GammaRegressor
import json
import os
from backend import get_data_path
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from thread_function import thread_function, model, get_isomap, get_x

plasma = matplotlib.cm.get_cmap('plasma')


def isomap():
    color_sets = backend.get_color_sets()
    files = backend.get_session_list()
    x = np.arange(10000)
    for session in files:
        [normalized_spikes, convolved_spikes, _, original_spikes], interval_ids, intervals_df = create_precision_df(
            session, regenerate=False)
        if convolved_spikes is None:
            continue
        intervals_df['trial_starts'] = [intervals_df[(intervals_df.interval_trial == row[1].interval_trial) & (
                (intervals_df.interval_phase == 0) | (intervals_df.interval_phase == 3.))].interval_starts.iloc[0]
                                        for row in intervals_df.iterrows()]
        intervals_df['trial_time'] = intervals_df.interval_starts - intervals_df.trial_starts

        blocks = intervals_df.block.unique()
        blocks.sort()
        down_sampled = convolved_spikes.T[::300]
        embedding = Isomap(n_components=2)
        embedding.fit(down_sampled)
        transformed_spikes = embedding.transform(convolved_spikes.T)
        entry_fig, entry_axes = plt.subplots(2, 2, figsize=(16, 9))
        reward_fig, reward_axes = plt.subplots(2, 2, figsize=(16, 9))
        for i in np.unique(interval_ids):
            trial = intervals_df.loc[i].interval_trial
            phase = intervals_df.loc[i].interval_phase
            block = intervals_df.loc[i].block
            group = intervals_df.loc[i].group
            trial_time = intervals_df.loc[i].trial_time
            block_color = color_sets['set2'][np.where(blocks == block)[0][0]]
            group_color = plasma((group - 1) / 5)
            trial_time_color = plasma((trial_time // 3) / 5)
            print(f'trial {trial}')
            if phase == 0:
                axes = entry_axes.flatten()
            else:
                axes = reward_axes.flatten()
            interval_spikes = transformed_spikes[np.where(interval_ids == i)[0]]
            size = 1
            axes[0].scatter(interval_spikes[:, 0], interval_spikes[:, 1],
                            c=range(len(interval_spikes)),
                            cmap="plasma", vmin=0, vmax=3000, s=size)
            axes[1].scatter(interval_spikes[:, 0], interval_spikes[:, 1],
                            color=block_color, s=size)
            axes[2].scatter(interval_spikes[:, 0], interval_spikes[:, 1],
                            color=group_color, s=size)
            axes[3].scatter(interval_spikes[:, 0], interval_spikes[:, 1],
                            color=trial_time_color, s=size)
            titles = ['Time', 'Block', 'Group', 'Trial Time']
            for j, ax in enumerate(axes):
                ax.set_xlim([transformed_spikes[:, 0].min(), transformed_spikes[:, 0].max()])
                ax.set_ylim([transformed_spikes[:, 1].min(), transformed_spikes[:, 1].max()])
                ax.set_title(titles[j])
        entry_fig.suptitle(f'session {session} entry')
        reward_fig.suptitle(f'session {session} reward')
        plt.show()
        backend.save_fig(entry_fig, f'{session}_entry_isomap', 'isomaps')
        backend.save_fig(reward_fig, f'{session}_reward_isomap', 'isomaps')


def add_one_in():
    json_path = os.path.join(os.getcwd(), 'isomap_units.json')
    if os.path.exists(json_path):
        unit_order_record = backend.load_json(json_path)
    else:
        unit_order_record = {}
    print('Starting add_one_in')
    files = backend.get_session_list()
    for session in files:
        [_, convolved_spikes, _, _], interval_ids, intervals_df = create_precision_df(
            session, regenerate=False)
        if convolved_spikes is None:
            return None
        phases = get_phase(interval_ids, intervals_df)
        phase_filter = np.where((phases == 1) | (phases == 2))[0]
        convolved_spikes = convolved_spikes[:, phase_filter]
        interval_ids = interval_ids[phase_filter]

        all_unit_list = np.arange(len(convolved_spikes))
        unit_order = []
        best_scores = []
        for num in range(len(convolved_spikes)):
            start = time.time()
            units_to_try = np.where([i not in unit_order for i in all_unit_list])[0]
            with ProcessPoolExecutor(max_workers=5) as execute:
                master_list = list(execute.map(thread_function, repeat(unit_order), units_to_try,
                                               repeat(convolved_spikes), repeat(interval_ids), repeat(intervals_df)))
            [spike_list, score_list] = list(zip(*master_list))
            print(f'multi core time: {time.time() - start:.2f} seconds')
            best = np.argmax(score_list)
            unit_order.append(int(best))
            plot_isomap(spike_list[best], interval_ids, intervals_df, unit_order)
            score = model(interval_ids, spike_list[best], 'linear', plot=True)
            best_scores.append(score)
            print(f'{unit_order}: {score} ')
            unit_order_record[session + '_in'] = [unit_order, best_scores]
            backend.save_json(json_path, unit_order_record)
            print('isomap_units.json saved')


def leave_one_out(session, multi_core=False, plot=False):
    unit_order_record = get_saved_isomap_units()

    print('Starting leave_one_out')
    [_, convolved_spikes, _, _], interval_ids, intervals_df = create_precision_df(
        session, regenerate=False)
    if convolved_spikes is None:
        return False
    phases = get_phase(interval_ids, intervals_df)
    phase_filter = np.where((phases == 1) | (phases == 2))[0]
    convolved_spikes = convolved_spikes[:, phase_filter]
    interval_ids = interval_ids[phase_filter]

    key = f'{session}_out'
    if key in unit_order_record:
        [unit_order, best_scores] = unit_order_record[key]
    else:
        unit_order = []
        best_scores = []

    for num in range(len(convolved_spikes) - len(unit_order) - 1):
        start = time.time()
        current_unit_list = [unit for unit in np.arange(len(convolved_spikes)) if unit not in unit_order]
        print(f'{len(current_unit_list)} units in unit_list')
        if multi_core:
            with ProcessPoolExecutor(max_workers=5) as execute:
                master_list = list(execute.map(thread_function, repeat(current_unit_list), current_unit_list,
                                               repeat(convolved_spikes), repeat(interval_ids),
                                               repeat(intervals_df)))
                [spike_list, score_list] = list(zip(*master_list))
                print(f'multi core time: {time.time() - start:.2f} seconds')
        else:
            master_list = list(map(thread_function, repeat(current_unit_list), current_unit_list,
                                   repeat(convolved_spikes), repeat(interval_ids), repeat(intervals_df)))
            [spike_list, score_list] = list(zip(*master_list))
            print(f'single core time: {time.time() - start:.2f} seconds')
        best = np.argmax(score_list)
        try:
            unit_order.append(int(current_unit_list[best]))
        except:
            print('test')
        if plot:
            plot_isomap(spike_list[best], interval_ids, intervals_df, unit_order)
        score = model(interval_ids, spike_list[best], 'linear', plot=plot)
        best_scores.append(score)
        print(f'{unit_order}: {score} ')
        unit_order_record[session + '_out'] = [unit_order, best_scores]
        save_isomap_units(unit_order_record)


def plot_isomap(transformed_spikes, interval_ids, intervals_df, session):
    color_sets = backend.get_color_sets()
    blocks = intervals_df.block.unique()
    blocks.sort()
    # entry_fig, entry_axes = plt.subplots(2, 2, figsize=(16, 9))
    reward_fig, reward_axes = plt.subplots(2, 2, figsize=(16, 9))
    for i in np.unique(interval_ids):
        phase = intervals_df.loc[i].interval_phase
        block = intervals_df.loc[i].block
        block_color = color_sets['set2'][np.where(blocks == block)[0][0]]
        # if phase == 0:
        #     axes = entry_axes.flatten()
        # else:
        axes = reward_axes.flatten()
        interval_spikes = transformed_spikes[np.where(interval_ids == i)[0]]
        size = 1
        x = np.linspace(0, len(interval_spikes) / 1000, len(interval_spikes))
        titles = ['Dim 1', 'Dim 2', 'Dim 3', 'Dim 4']
        for j, ax in enumerate(axes):
            if len(interval_spikes[0]) <= j:
                break
            axes[j].scatter(x, interval_spikes[:, j],
                            color=block_color, s=size)
            ax.set_ylim([transformed_spikes[:, j].min(), transformed_spikes[:, j].max()])
            ax.set_title(titles[j])
    # entry_fig.suptitle(f'session {session} entry')
    # reward_fig.suptitle(f'session {session} reward')
    reward_fig.suptitle(f'{session}')
    plt.show()


def get_phase(interval_ids, intervals_df):
    x_total = []
    for i in np.unique(interval_ids):
        num = len(np.where(interval_ids == i)[0])
        phase = intervals_df.loc[i].interval_phase
        x_total.append([phase] * num)
    phase = np.concatenate(x_total)
    return phase


def get_block(interval_ids, intervals_df):
    x_total = []
    for i in np.unique(interval_ids):
        num = len(np.where(interval_ids == i)[0])
        block = intervals_df.loc[i].block
        x_total.append([block] * num)
    block = np.concatenate(x_total)
    return block


def isomap_single():
    color_sets = backend.get_color_sets()
    files = backend.get_session_list()
    for session in files:
        [_, convolved_spikes, _, _], interval_ids, intervals_df = create_precision_df(
            session, regenerate=False)
        if convolved_spikes is None:
            continue
        intervals_df['trial_starts'] = [intervals_df[(intervals_df.interval_trial == row[1].interval_trial) & (
                (intervals_df.interval_phase == 0) | (intervals_df.interval_phase == 3.))].interval_starts.iloc[0]
                                        for row in intervals_df.iterrows()]
        intervals_df['trial_time'] = intervals_df.interval_starts - intervals_df.trial_starts

        blocks = intervals_df.block.unique()
        blocks.sort()
        down_sampled = convolved_spikes.T[::300]
        embedding = Isomap(n_components=4)
        embedding.fit(down_sampled)
        transformed_spikes = embedding.transform(convolved_spikes.T)
        entry_fig, entry_axes = plt.subplots(2, 2, figsize=(16, 9))
        reward_fig, reward_axes = plt.subplots(2, 2, figsize=(16, 9))
        x_total = []

        for i in np.unique(interval_ids):
            trial = intervals_df.loc[i].interval_trial
            phase = intervals_df.loc[i].interval_phase
            block = intervals_df.loc[i].block
            group = intervals_df.loc[i].group
            trial_time = intervals_df.loc[i].trial_time
            block_color = color_sets['set2'][np.where(blocks == block)[0][0]]
            group_color = plasma((group - 1) / 5)
            trial_time_color = plasma((trial_time // 3) / 5)
            print(f'trial {trial}')
            if phase == 0:
                axes = entry_axes.flatten()
            else:
                axes = reward_axes.flatten()
            interval_spikes = transformed_spikes[np.where(interval_ids == i)[0]]
            size = 1
            x = np.linspace(0, len(interval_spikes) / 1000, len(interval_spikes))
            x_total.append(x)
            titles = ['Dim 1', 'Dim 2', 'Dim 3', 'Dim 4']
            for j, ax in enumerate(axes):
                axes[j].scatter(x, interval_spikes[:, j],
                                color=block_color, s=size)
                ax.set_ylim([transformed_spikes[:, j].min(), transformed_spikes[:, j].max()])
                ax.set_title(titles[j])
        longest = np.argmax(np.array([len(x_i) for x_i in x_total]))
        model = PoissonRegressor()
        model.fit(transformed_spikes, np.concatenate(x_total))
        score = model.score(transformed_spikes, np.concatenate(x_total))

        """stopped here"""
        fit_data = model.predict(np.concatenate(x_total).reshape((-1, 1)))
        residuals = fit_data - transformed_spikes
        fit_line = model.predict(x_total[longest].reshape((-1, 1)))
        entry_fig.suptitle(f'session {session} entry')
        reward_fig.suptitle(f'session {session} reward')
        plt.show()
        # backend.save_fig(entry_fig, f'{session}_entry_isomap', 'isomaps')
        # backend.save_fig(reward_fig, f'{session}_reward_isomap', 'isomaps')


def pretty_plot(session):
    color_sets = backend.get_color_sets()

    unit_order_record = get_saved_isomap_units()

    [_, convolved_spikes, _, _], interval_ids, intervals_df = create_precision_df(
        session, regenerate=False)
    if convolved_spikes is None:
        return None
    blocks = intervals_df.block.unique()
    blocks.sort()
    phases = get_phase(interval_ids, intervals_df)
    x_blocks = get_block(interval_ids, intervals_df)
    phase_filter = np.where((phases == 2))[0]
    # phase_filter = np.where((phases == 1) | (phases == 2))[0]
    convolved_spikes = convolved_spikes[:, phase_filter]
    interval_ids = interval_ids[phase_filter]
    x_blocks = x_blocks[phase_filter]
    [unit_order, scores] = unit_order_record[f'{session}_out']
    unit_order = unit_order[np.argmax(scores) + 1:]
    transformed_spikes, score = get_isomap(convolved_spikes, interval_ids, intervals_df, unit_order, components=2)
    fit_model, x = model(interval_ids, transformed_spikes, 'linear', return_model=True)
    fit_model_b1, x_b1 = model(interval_ids[x_blocks == blocks[0]], transformed_spikes[x_blocks == blocks[0]], 'linear',
                               return_model=True)
    fit_model_b2, x_b2 = model(interval_ids[x_blocks == blocks[1]], transformed_spikes[x_blocks == blocks[1]], 'linear',
                               return_model=True)
    prediction = fit_model.predict(transformed_spikes)
    prediction_fit_to1 = fit_model_b1.predict(transformed_spikes)
    prediction_fit_to2 = fit_model_b2.predict(transformed_spikes)
    titles = ['fit to average', f'fit to block {blocks[0]}', f'fit to block {blocks[1]}']
    for k, pred in enumerate([prediction, prediction_fit_to1, prediction_fit_to2]):
        for c, block in enumerate(blocks):
            i = x_blocks == block
            plt.scatter(x[i], pred[i], s=1, c=color_sets['set2'][c])
        plt.xlabel('real time')
        plt.ylabel('predicted time')
        plt.title(titles[k])
        plt.show()
        plt.plot()
        y_max = []
        for c, block in enumerate(blocks):
            i = x_blocks == block
            block_predictions = pred[i]
            block_x = x[i]
            times = np.unique(block_x)
            times.sort()
            plt.plot(times, times, c=[.8, .8, .8])
            mean_neural_time = np.zeros(len(times))
            std_neural_time = np.zeros(len(times))
            for j, x_i in enumerate(times):
                mean_neural_time[j] = np.mean(block_predictions[np.where(block_x == x_i)[0]])
                std_neural_time[j] = np.std(block_predictions[np.where(block_x == x_i)[0]])
            plt.plot(times, mean_neural_time, c=color_sets['set2'][c])
            y_max.append(max(mean_neural_time))
        plt.xlabel('real time')
        plt.ylabel('predicted time')
        # plt.legend(['true'] + list(blocks))
        plt.ylim([-.3, max(y_max) + .3])
        plt.title(titles[k])
        plt.show()
    plot_isomap(transformed_spikes, interval_ids, intervals_df, session)
    plot_multicolor(transformed_spikes, interval_ids, intervals_df, session, dims=2)
    # plot_multicolor(transformed_spikes, interval_ids, intervals_df, session, dims=3)


def plot_multicolor(transformed_spikes, interval_ids, intervals_df, session, dims=2):
    color_sets = backend.get_color_sets()
    fit_model, x = model(interval_ids, transformed_spikes, 'linear', return_model=True)
    if dims == 2:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    phases = get_phase(interval_ids, intervals_df)
    phase_filter = np.where((phases == 1) | (phases == 2))[0]
    spikes = transformed_spikes[phase_filter, :]
    x = x[phase_filter]
    if dims == 2:
        sc = ax.scatter(spikes[:, 0], spikes[:, 1], c=x, cmap="plasma", vmin=0, vmax=max(x) * .7, s=1)
    else:
        sc = ax.scatter3D(spikes[:, 0], spikes[:, 1], spikes[:, 2], c=x, cmap="plasma", vmin=0, vmax=max(x), s=1)

    ax.set_title(session)
    color_bar = plt.colorbar(sc)
    color_bar.set_label('time (seconds)')
    ax.set_xlabel('Isomap Component 1')
    ax.set_ylabel('Isomap Component 2')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    blocks = intervals_df.block.unique()
    blocks.sort()
    x_blocks = get_block(interval_ids, intervals_df)
    x_blocks = x_blocks[phase_filter]
    for b in range(len(blocks)):
        ax.scatter(spikes[x_blocks == blocks[b], 0], spikes[x_blocks == blocks[b], 1], color=color_sets['set2'][b], s=1)
    ax.set_title(session)
    ax.set_xlabel('Isomap Component 1')
    ax.set_ylabel('Isomap Component 2')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    intervals_df['length'] = intervals_df['interval_ends'] - intervals_df['interval_starts']
    lengths = intervals_df.length.loc[np.unique(interval_ids)]
    num_groups = 3
    time_groups = np.array_split(np.array(lengths.index[np.argsort(lengths.values)]), num_groups)
    colors = [color_sets['set2'], color_sets['set2_med_dark'], color_sets['set2_dark']]

    for b in range(len(blocks)):
        for j in range(num_groups):
            time_cutoff = np.array([val in time_groups[j] for val in interval_ids[phase_filter]])
            block_spikes = spikes[(x_blocks == blocks[b]) & time_cutoff, :]
            t = x[(x_blocks == blocks[b]) & time_cutoff]
            sc = ax.scatter(block_spikes[:, 0], block_spikes[:, 1],  c=list(colors[j][b]), s=1)
        ax.set_title(session + ' block ' + blocks[b])
        ax.set_xlabel('Isomap Component 1')
        ax.set_ylabel('Isomap Component 2')
        plt.show()


def plot_scores(session):
    unit_order_record = get_saved_isomap_units()
    unit_order = unit_order_record[f'{session}_out']
    fig, ax = plt.subplots(1, 1, figsize=[6, 8])
    ax.plot(unit_order[1])
    ax.set_xlabel('number of units')
    ax.set_ylabel('model score')
    plt.show()


def get_saved_isomap_units():
    json_path = os.path.join(os.getcwd(), 'isomap_units.json')
    if os.path.exists(json_path):
        unit_order_record = backend.load_json(json_path)
    else:
        unit_order_record = {}
    return unit_order_record


def save_isomap_units(unit_order_record):
    json_path = os.path.join(os.getcwd(), 'isomap_units.json')
    backend.save_json(json_path, unit_order_record)
    print('isomap_units.json saved')


if __name__ == '__main__':
    # add_one_in()
    files = backend.get_session_list()
    for sess in files:
        if sess != 'ES029_2022-09-14_bot72_0_g0':
            continue
        # leave_one_out(sess, multi_core=False)
        pretty_plot(sess)
        # plot_scores(sess)
