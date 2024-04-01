import numpy as np


def make_color_gradient(c1, c2, num):
    return [val/255 for val in np.vstack([np.linspace(a, b, num) for a, b in zip(c1, c2)]).T]
