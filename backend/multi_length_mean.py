import numpy as np


def multi_length_mean(list_of_arr, uneven_axis=1):
    even_axis = (uneven_axis + 1) % 2
    for arr in list_of_arr:
        assert len(arr.shape) < 3, 'array should be 2 dimensional'
        assert arr.shape[even_axis] == list_of_arr[0].shape[
            even_axis], 'the opposite axis should have the same size on all arrays'

    max_length = np.max(np.array([arr.shape[uneven_axis] for arr in list_of_arr]))
    height = list_of_arr[0].shape[even_axis]
    counts = np.zeros([height, max_length])
    values = np.zeros([height, max_length])
    for arr in list_of_arr:
        length = arr.shape[uneven_axis]
        counts[:, :length] += 1
        if uneven_axis == 1:
            values[:, :length] += arr
        else:
            values[:, :length] += arr.T
    if uneven_axis == 0:
        values = values.T
    return np.divide(values, counts), counts
