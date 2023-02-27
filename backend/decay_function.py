import numpy as np
import matplotlib.pyplot as plt


def decay_function(x, cumulative=8, starting=1):
    a = starting
    b = a / cumulative
    density = a / np.exp(b * x)
    return density


def decay_function_cumulative(x, cumulative=8, starting=1):
    a = starting
    b = a / cumulative
    cum = cumulative * (1 - np.exp(-b * x))
    return cum


if __name__ == '__main__':
    t = np.linspace(0, 20, 50)
    plt.plot(t, decay_function(t))
    plt.show()
    plt.plot(t, decay_function_cumulative(t))
    plt.show()
