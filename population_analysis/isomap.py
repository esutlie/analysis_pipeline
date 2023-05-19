from sklearn.manifold import Isomap
from create_bins_df import create_precision_df
import backend
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression, PoissonRegressor, GammaRegressor

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

        unit_order = [6, 19, 3, 20, 11]
        best_scores = []
        for num in range(len(convolved_spikes)):
            spike_list = []
            score_list = []
            for unit in range(len(convolved_spikes)):
                unit_list = unit_order + [unit]
                transformed_spikes, score = get_isomap(convolved_spikes, interval_ids, intervals_df, unit_list)
                spike_list.append(transformed_spikes)
                score_list.append(score)
            best = np.argmax(score_list)
            unit_order.append(int(best))
            plot_isomap(spike_list[best], interval_ids, intervals_df, unit_order)
            score = model(interval_ids, spike_list[best], 'linear', plot=True)
            best_scores.append(score)
            print(f'{unit_order}: {score} ')
        print('test')


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


def get_isomap(convolved_spikes, interval_ids, intervals_df, unit_list):
    spikes = convolved_spikes[unit_list]
    intervals_df['trial_starts'] = [intervals_df[(intervals_df.interval_trial == row[1].interval_trial) & (
            (intervals_df.interval_phase == 0) | (intervals_df.interval_phase == 3.))].interval_starts.iloc[0]
                                    for row in intervals_df.iterrows()]
    intervals_df['trial_time'] = intervals_df.interval_starts - intervals_df.trial_starts

    blocks = intervals_df.block.unique()
    blocks.sort()
    down_sampled = spikes.T[::300]
    n_components = len(unit_list) if len(unit_list) < 4 else 4
    embedding = Isomap(n_components=n_components)
    embedding.fit(down_sampled)
    transformed_spikes = embedding.transform(spikes.T)
    score = model(interval_ids, transformed_spikes, 'linear')
    # print(f'{unit_list}: {score}')
    return transformed_spikes, score


def model(interval_ids, transformed_spikes, type='linear', plot=False):
    types = {'linear': LinearRegression,
             'poisson': PoissonRegressor,
             'gamma': GammaRegressor}
    x, longest = get_x(interval_ids)
    model = types[type]()
    model.fit(transformed_spikes, x)
    score = model.score(transformed_spikes, x)
    prediction = model.predict(transformed_spikes)
    if plot:
        plt.scatter(x, prediction)
        plt.xlabel('real time')
        plt.ylabel('predicted time')
        plt.show()
    return score


def get_phase(interval_ids, intervals_df):
    x_total = []
    for i in np.unique(interval_ids):
        num = len(np.where(interval_ids == i)[0])
        phase = intervals_df.loc[i].interval_phase
        x_total.append([phase] * num)
    phase = np.concatenate(x_total)
    return phase


def get_x(interval_ids):
    x_total = []
    for i in np.unique(interval_ids):
        num = len(np.where(interval_ids == i)[0])
        x_total.append(np.linspace(0, num / 1000, num))
    longest = np.argmax(np.array([len(x_i) for x_i in x_total]))
    longest = x_total[longest]
    x = np.concatenate(x_total)
    return x, longest


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


if __name__ == '__main__':
    add_one_in()
