from sklearn.manifold import Isomap
from create_bins_df import create_precision_df
import backend
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression, PoissonRegressor

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


def isomap_single():
    color_sets = backend.get_color_sets()
    files = backend.get_session_list()
    for session in files:
        [normalized_spikes, convolved_spikes, _, _], interval_ids, intervals_df = create_precision_df(
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
    isomap_single()
