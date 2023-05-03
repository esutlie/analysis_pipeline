import numpy as np
import matplotlib.pyplot as plt


def decay_function(x, cumulative=8., starting=1.):  # f(t)
    a = starting
    b = a / cumulative
    density = a / np.exp(b * x)
    return density


def decay_function_cumulative(x, cumulative=8., starting=1.):  # F(x)
    a = starting
    b = a / cumulative
    cum = cumulative * (1 - np.exp(-b * x))
    return cum


def weighted_time_function_step(x, cumulative=8., starting=1., steps=1000):  # Ft(x)
    s = np.linspace(0, x, steps)
    values = decay_function(s, cumulative=cumulative, starting=starting) * s
    integral = np.sum(values * x / steps, axis=0)
    return integral


def weighted_time_function(x, cumulative=8., starting=1.):  # Ft(x)
    a = starting
    b = a / cumulative
    integral = a / (b ** 2) * (1 - np.exp(-b * x) * (b * x + 1))
    return integral


if __name__ == '__main__':
    t = np.linspace(0, 20, 50)
    c = .6
    start = .1
    plt.plot(t, decay_function(t, cumulative=c, starting=start))
    plt.show()
    plt.plot(t, decay_function_cumulative(t, cumulative=c, starting=start))
    plt.show()
    plt.plot(t, weighted_time_function(t, cumulative=c, starting=start))
    plt.show()
