from sklearn.manifold import Isomap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, PoissonRegressor, GammaRegressor
import warnings
import time

def thread_function(unit_order, unit, convolved_spikes, interval_ids, intervals_df):
    unit_list = unit_order.copy()
    if unit in unit_order:
        unit_list.remove(unit)
    else:
        unit_list = unit_order + [unit]
    start = time.time()
    transformed_spikes, score = get_isomap(convolved_spikes, interval_ids, intervals_df, unit_list)
    print(f'unit {unit}: {time.time()-start:.1f} seconds')
    return transformed_spikes, score


def get_isomap(convolved_spikes, interval_ids, intervals_df, unit_list, time_limit=5, components=4):
    spikes = convolved_spikes[unit_list]
    x, longest = get_x(interval_ids)
    length_filter = np.where(x < time_limit)[0]

    blocks = intervals_df.block.unique()
    blocks.sort()
    down_sampled = spikes[:, length_filter].T[::300]
    n_components = len(unit_list) if len(unit_list) < components else components
    embedding = Isomap(n_components=n_components)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedding.fit(down_sampled)
    transformed_spikes = embedding.transform(spikes.T)
    score = model(interval_ids, transformed_spikes, 'linear')
    return transformed_spikes, score


def model(interval_ids, transformed_spikes, type='linear', plot=False, return_model=False):
    types = {'linear': LinearRegression,
             'poisson': PoissonRegressor,
             'gamma': GammaRegressor}
    x, longest = get_x(interval_ids)
    model1 = types[type]()
    model1.fit(transformed_spikes, x+.0001)
    if return_model:
        return model1, x
    score = model1.score(transformed_spikes, x)
    prediction = model1.predict(transformed_spikes)
    if plot:
        plt.plot(x, prediction)
        plt.xlabel('real time')
        plt.ylabel('predicted time')
        plt.show()
    return score


def get_x(interval_ids, min_longest=1):
    x_total = []
    for i in np.unique(interval_ids):
        num = len(np.where(interval_ids == i)[0])
        x_total.append(np.linspace(0, (num - 1) / 1000, num))
    longest = np.argsort(np.array([len(x_i) for x_i in x_total]), axis=0)[-min_longest]
    # longest = np.argmax(np.array([len(x_i) for x_i in x_total]))
    longest = x_total[longest]
    x = np.concatenate(x_total)
    x = np.around(x, 3)
    return x, longest
