"""
Techniques based off of:
 MacDonald, Christopher J., Stephen Carrow, Ryan Place, and Howard Eichenbaum. 2013. “Distinct Hippocampal Time Cell
 Sequences Represent Odor Memories in Immobilized Rats.” The Journal of Neuroscience: The Official Journal of the
 Society for Neuroscience 33 (36): 14607–16.

Found: https://www.jneurosci.org/content/33/36/14607.full
"""

from sklearn.manifold import Isomap
from population_analysis.create_bins_df import create_precision_df
import backend
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression, PoissonRegressor

plasma = matplotlib.cm.get_cmap('plasma')


def glm():
    model = PoissonRegressor()

    pass


def anova():
    pass


def shift():
    pass


def even_odd():
    pass


def time_field():
    pass


def time_cells():
    pass


def classify():
    color_sets = backend.get_color_sets()
    files = backend.get_session_list()
    for session in files:
        [normalized_spikes, convolved_spikes, boxcar_spikes, _], interval_ids, intervals_df = create_precision_df(
            session, regenerate=True)
        if convolved_spikes is None:
            continue
        intervals_df['trial_starts'] = [intervals_df[(intervals_df.interval_trial == row[1].interval_trial) & (
                (intervals_df.interval_phase == 0) | (intervals_df.interval_phase == 3.))].interval_starts.iloc[0]
                                        for row in intervals_df.iterrows()]
        intervals_df['trial_time'] = intervals_df.interval_starts - intervals_df.trial_starts

        blocks = intervals_df.block.unique()
        blocks.sort()
        fig, ax = plt.subplots(1, 1, figsize=[8, 6])
        size = 1
        for unit in range(len(convolved_spikes)):
            spikes = boxcar_spikes
            rate = sum(spikes[unit]) / len(spikes[unit]) * 10
            x_total = []
            for i in np.unique(interval_ids):
                interval_spikes = spikes[unit, np.where(interval_ids == i)[0]]
                x = np.linspace(0, len(interval_spikes) / 1000, len(interval_spikes))
                x_total.append(x)
                plt.scatter(x, interval_spikes, s=size)
            longest = np.argmax(np.array([len(x_i) for x_i in x_total]))
            model = PoissonRegressor()
            model.fit(np.concatenate(x_total).reshape((-1, 1)), spikes[unit])
            score = model.score(np.concatenate(x_total).reshape((-1, 1)), spikes[unit])
            fit_data = model.predict(np.concatenate(x_total).reshape((-1, 1)))
            residuals = fit_data - spikes[unit]
            fit_line = model.predict(x_total[longest].reshape((-1, 1)))
            plt.scatter(x_total[longest], fit_line, s=size, color='k')
            plt.title(f'unit {unit}, score: {score:.3f} D^2, {rate:.2f} spikes per second')
            plt.show()
            # plt.hist(residuals)
            # plt.show()

        break


if __name__ == '__main__':
    classify()
