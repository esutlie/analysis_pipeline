import numpy as np


def center_of_mass(x, axis=0):
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)
    if axis == 1:
        x = x.T

    return np.average(np.vstack([list(range(x.shape[0]))] * x.shape[1]), axis=1, weights=x.T)


if __name__ == '__main__':
    test_x = np.array([0, 0, 0, 0, 1, 0, 0])
    center_of_mass(test_x)

    test_x = np.array([[0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 0, 1, 0, 0]])
    center_of_mass(test_x, axis=1)
