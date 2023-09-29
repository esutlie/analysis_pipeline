from backend.min_dif import min_dif
import matplotlib.pyplot as plt
import numpy as np


def extract_event(df, key, value=1, tolerance=None, plot=False):
    off_value = 0 if value == 1 else 1
    if tolerance is None:
        if key == 'lick':
            tolerance = .03
        elif key == 'head':
            tolerance = .2
        else:
            tolerance = .1

    on_times = df[(df.key == key) & (df.value == value)].time.to_numpy()
    off_times = df[(df.key == key) & (df.value == off_value)].time.to_numpy()

    forward = min_dif(on_times, off_times)
    forward_off = min_dif(on_times, off_times, rev=True)

    if plot:
        h = plt.hist(forward[forward < .5], bins=100)
        plt.vlines(.03, 0, max(h[0]), 'r')
        plt.show()

    on_times = on_times[forward > tolerance]
    off_times = off_times[forward_off > tolerance]

    back = min_dif(off_times, on_times, rev=True)

    if plot:
        h = plt.hist(back[back < .5], bins=100)
        plt.vlines(.03, 0, max(h[0]), 'r')
        plt.show()

    return on_times[back > tolerance]
