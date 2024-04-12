import matplotlib.pyplot as plt
import numpy as np

import backend
from backend import make_color_gradient

DISTANCE_METRIC = False
arg_func = np.argmin if DISTANCE_METRIC else np.argmax
num_func = np.min if DISTANCE_METRIC else np.max


def make_relu(activity_list):
    max_length = np.max([arr.shape[1] for arr in activity_list])
    x = np.arange(max_length)
    scalar = 1 - x / np.quantile([arr.shape[1] for arr in activity_list], .9)
    scalar[scalar < .1] = .1

    # scalar_arr = .5 - x / (np.quantile([arr.shape[1] for arr in activity_list], .9) * 4)
    # scalar_arr = scalar_arr[:, None] + scalar_arr[None, :]
    # scalar_arr[scalar_arr < .1] = .1

    scalar_arr = 1 - x / (np.quantile([arr.shape[1] for arr in activity_list], .9) * 2)
    scalar_arr = scalar_arr[:, None] * scalar_arr[None, :]
    scalar_arr[scalar_arr < .1] = .1
    scalar_arr[scalar_arr > 1] = 1

    # plt.imshow(scalar_arr, origin='lower')
    # plt.colorbar()
    # plt.show()

    return scalar, scalar_arr


def sigmoid(x, plots=False):
    if plots:
        plt.plot(x, 1 / (1 + np.exp(-(x - .7) * 10)))
        plt.show()

    return 1 / (1 + np.exp(-(x - .7) * 10))


def manhattan_distance_matrix(a, b, bounds=False, clip_long=False, distribute=False):
    if clip_long:
        multiplier = 1
        if len(a.T) > len(b.T) * multiplier:
            a = a[:, :len(b.T) * multiplier].copy()
        elif len(b.T) > len(a.T) * multiplier:
            b = b[:, :len(a.T) * multiplier].copy()
    try:
        if bounds:
            a -= a.min(axis=1)[:, None]
            a /= a.max(axis=1)[:, None]
            b -= b.min(axis=1)[:, None]
            b /= b.max(axis=1)[:, None]
    except Exception as e:
        print(e)
        print()

    res = np.mean(np.abs(a[:, :, np.newaxis] - b[:, np.newaxis, :]), axis=0)

    if distribute:
        flat = res.flatten()
        flat = np.argsort(np.argsort(flat)) / (len(flat) - 1)
        res = np.reshape(flat, res.shape)
    return res


def line_maker(m, b, x):
    return m * x + b


def score_line(m, b, arr, scalar, show_plot=False, swath=True):
    x = np.arange(arr.shape[0])
    y = line_maker(m, b, x)
    y_ind = np.floor(y).astype(int)
    on_line = (y_ind < arr.shape[1]) & (y_ind > 0)
    coverage_req = .15
    if not np.any(on_line) or np.sum(on_line) < coverage_req * arr.shape[0] \
            or (np.max(y_ind[on_line]) - np.min(y_ind[on_line])) < coverage_req * arr.shape[1]:
        return arr.max() if DISTANCE_METRIC else 0

    result = None
    swath_width = np.max(arr.shape) / 10
    scaled_arr = scalar[:arr.shape[0], :arr.shape[1]] * arr

    try:
        if swath:
            shift = swath_width / 2 / np.cos(np.arctan(m))
            y_upper = line_maker(m, b + shift, x)
            y_lower = line_maker(m, b - shift, x)
            lower_arr = np.greater(np.arange(scaled_arr.shape[1])[None, :], y_lower[:, None])
            upper_arr = np.less(np.arange(scaled_arr.shape[1])[None, :], y_upper[:, None])
            valid = upper_arr & lower_arr
            result = np.mean(scaled_arr[valid])
            # trying to correct for diagonal preference
            # mean_inside = np.mean(scaled_arr[valid])
            # mean_outside = np.mean(scaled_arr[~valid])
            # result = mean_inside / (mean_inside + mean_outside)
            # result = np.sum(scaled_arr[valid]) / np.sum(scaled_arr)

            # print(f'{np.sum(arr[valid]):.2f} inside out of {np.sum(arr):.2f} total. Score = {result}')
            if show_plot:
                plt.imshow(arr.T, origin='lower')
                plt.colorbar()
                y_lim = plt.gca().get_ylim()
                x_lim = plt.gca().get_xlim()
                plt.plot(y, c='k')
                plt.plot(y_lower, c='r')
                plt.plot(y_upper, c='r')
                # plt.title(
                #     f'Slope: {m:.3f}  Intercept {b:.1f}  Score = {result:.3f}')
                # f'{np.sum(scaled_arr[valid]):.0f} inside out of {np.sum(scaled_arr):.0f} total. Score = {result:.3f}')
                plt.xlabel('Interval 1 (sec)')
                plt.ylabel('Interval 2 (sec)')
                x_ticks = plt.gca().get_xticks()
                y_ticks = plt.gca().get_yticks()
                plt.xticks(x_ticks,x_ticks/100)
                plt.yticks(y_ticks,y_ticks/100)
                plt.ylim(y_lim)
                plt.xlim(x_lim)
                plt.show()
                # print()
        else:
            y_lower = np.floor(y).astype(int)
            y_upper = np.ceil(y).astype(int)
            y_weights = y - y_lower

            valid = (y_lower < scaled_arr.shape[1]) & (y_upper < scaled_arr.shape[1]) & (y_lower > 0) & (y_upper > 0)
            if not np.any(valid):
                return scaled_arr.max() if DISTANCE_METRIC else 0

            y_lower = y_lower[valid]
            y_upper = y_upper[valid]
            y_weights = y_weights[valid]
            x = x[valid]

            result = np.mean(scaled_arr[x, y_lower] * (1 - y_weights) + scaled_arr[x, y_upper] * y_weights)
    except Exception as e:
        print(e)
        print()

    return result


def starter_line(arr, scalar, show_plot=False):
    # slopes = [4, 3, 2, 1.3, .8, .5, .25]
    slopes = np.sort(np.concatenate([np.logspace(-1, 1, 5), np.logspace(-.3, .3, 6)]))
    b = 0
    if show_plot:
        colors = make_color_gradient([222, 15, 0], [64, 4, 0], len(slopes))
        plt.imshow(arr.T, origin='lower')
        y_lim = plt.gca().get_ylim()
        for j in range(len(slopes)):
            plt.plot(line_maker(slopes[j], b, np.arange(arr.shape[0])), c=colors[j])
        plt.ylim(y_lim)
        plt.show()
    scores = [score_line(val, b, arr, scalar) for val in slopes]
    try:
        res = slopes[arg_func(scores)]
    except Exception as e:
        print(e)
        print()
        res = None
    return res


def gradient_descent(m, b, arr, scalar, show_plots=False, show_progress=False, show_steps=False, free_intercept=False):
    # tic = backend.Timer()
    colors = make_color_gradient([222, 15, 0], [64, 4, 0], 5)
    intercept_delta = arr.shape[0] / 20
    slope_multiplier = 1.3
    b_delta_list = [-4 * intercept_delta, -intercept_delta / 2, 0, intercept_delta / 2, 4 * intercept_delta]
    m_multiplier_list = [1 / slope_multiplier, .95, .99, 1, 1.01, 1.05, slope_multiplier]
    # b_delta_list = [-4 * intercept_delta, -2 * intercept_delta, -intercept_delta / 2, 0, intercept_delta / 2,
    #                 2 * intercept_delta, 4 * intercept_delta]
    # m_multiplier_list = [1 / (slope_multiplier ** 2), 1 / slope_multiplier, .99, 1, 1.01, slope_multiplier,
    #                      slope_multiplier ** 2]
    m_history = [m]
    b_history = [b]
    m_cutoff = .1
    # tic.tic('setup')
    for i in range(100):
        if show_steps:
            if free_intercept:
                plt.imshow(arr.T, origin='lower')
                y_lim = plt.gca().get_ylim()
                plt.plot(line_maker(m, b, np.arange(arr.shape[0])), c='k')
                [plt.plot(line_maker(m, b + val, np.arange(arr.shape[0])), zorder=0, c='w') for val in b_delta_list]
                scores = [score_line(m, b + val, arr, scalar) for val in b_delta_list]
                b += b_delta_list[arg_func(scores)]
                plt.plot(line_maker(m, b, np.arange(arr.shape[0])), c='r')
                plt.ylim(y_lim)
                plt.show()

                plt.imshow(arr.T, origin='lower')
                y_lim = plt.gca().get_ylim()
                plt.plot(line_maker(m, b, np.arange(arr.shape[0])), c='k')
                [plt.plot(line_maker(m * val, b - (m * (val - 1) * arr.shape[0] / 2), np.arange(arr.shape[0])),
                          zorder=0,
                          c='w') for val in m_multiplier_list]
                scores = [score_line(m * val, b - (m * (val - 1) * arr.shape[0] / 2), arr, scalar) for val in
                          m_multiplier_list]
                mult_to_use = m_multiplier_list[arg_func(scores)]
                m *= mult_to_use
                m = m if m > m_cutoff else m_cutoff
                m = m if m < 1 / m_cutoff else 1 / m_cutoff
                if m != m_cutoff:
                    b -= (m * (mult_to_use - 1) * arr.shape[0] / 2)
                plt.plot(line_maker(m, b, np.arange(arr.shape[0])), c='r')
                plt.ylim(y_lim)
                plt.show()

            plt.imshow(arr.T, origin='lower')
            y_lim = plt.gca().get_ylim()
            plt.plot(line_maker(m, b, np.arange(arr.shape[0])), c='k')
            [plt.plot(line_maker(m * val, b, np.arange(arr.shape[0])), zorder=0, c='w') for val in m_multiplier_list]
            scores = [score_line(m * val, b, arr, scalar) for val in m_multiplier_list]
            mult_to_use = m_multiplier_list[arg_func(scores)]
            m *= mult_to_use
            m = m if m > m_cutoff else m_cutoff
            m = m if m < 1 / m_cutoff else 1 / m_cutoff
            plt.plot(line_maker(m, b, np.arange(arr.shape[0])), c='r')
            plt.ylim(y_lim)
            plt.show()
        else:
            if free_intercept:
                scores = [score_line(m, b + val, arr, scalar) for val in b_delta_list]
                b += b_delta_list[arg_func(scores)]
                scores = [score_line(m * val, b - (m * (val - 1) * arr.shape[0] / 2), arr, scalar) for val in
                          m_multiplier_list]
                mult_to_use = m_multiplier_list[arg_func(scores)]
                b -= (m * (mult_to_use - 1) * arr.shape[0] / 2)
                m *= mult_to_use
                m = m if m > m_cutoff else m_cutoff
                m = m if m < 1 / m_cutoff else 1 / m_cutoff
            scores = [score_line(m * val, b, arr, scalar) for val in m_multiplier_list]
            mult_to_use = m_multiplier_list[arg_func(scores)]
            m *= mult_to_use
            m = m if m > m_cutoff else m_cutoff
            m = m if m < 1 / m_cutoff else 1 / m_cutoff

        m = m if m > m_cutoff else m_cutoff
        m = m if m < 1 / m_cutoff else 1 / m_cutoff
        m_history.append(m)
        b_history.append(b)

        if len(m_history) > 3:
            m_history = m_history[-3:]
            b_history = b_history[-3:]

        if show_progress:
            plt.imshow(arr.T, origin='lower')
            y_lim = plt.gca().get_ylim()
            for j in range(len(m_history)):
                plt.plot(line_maker(m_history[j], b_history[j], np.arange(arr.shape[0])), c=colors[j])
            plt.ylim(y_lim)
            plt.show()
        if np.all(np.array(m_history) == m_history[0]) and np.all(np.array(b_history) == b_history[0]):
            score = score_line(m, b, arr, scalar)
            # print(f'converged in {i} steps')
            if show_plots:
                plt.imshow(arr.T, origin='lower')
                y_lim = plt.gca().get_ylim()
                for j in range(len(m_history)):
                    plt.plot(line_maker(m_history[j], b_history[j], np.arange(arr.shape[0])), c=colors[j])
                plt.ylim(y_lim)
                plt.title(f'converged at i = {i} with score = {score}')
                plt.show()
            return m, b
    # print('gradient descent never converged')
    if show_plots:
        plt.imshow(arr.T, origin='lower')
        y_lim = plt.gca().get_ylim()
        for j in range(len(m_history)):
            plt.plot(line_maker(m_history[j], b_history[j], np.arange(arr.shape[0])), c=colors[j])
        plt.ylim(y_lim)
        plt.title('never converged')
        plt.show()
    return m, b


def regression(arr_original, scalar, show_plots=False, show_final=False, slope=None, intercept=None):
    # tic = backend.Timer()
    if DISTANCE_METRIC:
        arr = arr_original
    else:
        arr_original -= np.min(arr_original)
        arr_original /= np.max(arr_original)
        arr_original = 1 - arr_original
        arr = sigmoid(arr_original)
    # tic.tic('setup')
    do_gradient_descent = slope is None or intercept is None
    if slope is None:
        slope = starter_line(arr, scalar, show_plot=show_plots)
    if intercept is None:
        intercept = 0
    # tic.tic('starter line')

    if do_gradient_descent:
        m, b = gradient_descent(slope, intercept, arr, scalar, show_plots=show_plots)
    else:
        m, b = slope, intercept
    # tic.tic('gradient descent')
    score = score_line(m, b, arr, scalar, show_plot=show_plots)
    # tic.tic('score final')
    if show_final:
        plt.imshow(arr.T, origin='lower')
        y_lim = plt.gca().get_ylim()
        plt.plot(line_maker(m, b, np.arange(arr.shape[0])), 'k')
        plt.ylim(y_lim)
        plt.show()
    return score, m, b


def recurrence(activity_list, show_plots=False):
    """
    Main function of this script, everything else is called from here
    """
    # tic = backend.Timer()
    n = len(activity_list)
    _, scalar = make_relu(activity_list)
    n_units = activity_list[0].shape[0]
    mean_DM = np.zeros([n, n])
    scores = np.zeros([n, n])
    slopes = np.zeros([n, n])
    intercepts = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if j < i or activity_list[i].shape[1] == 0 or activity_list[j].shape[1] == 0:
                continue

            # tic.tic('setup')
            distance_matrix = manhattan_distance_matrix(activity_list[i], activity_list[j], bounds=True,
                                                        clip_long=True, distribute=True)
            # tic.tic('distance matrix')
            score, m, b = regression(distance_matrix, scalar, show_plots=show_plots)
            # tic.tic('regression')

            scores[i, j] = score
            slopes[i, j] = m
            intercepts[i, j] = b
            mean_DM[i, j] = np.mean(distance_matrix)

            scores[j, i] = score
            slopes[j, i] = 1 / m
            intercepts[j, i] = -b / m
            mean_DM[j, i] = np.mean(distance_matrix)
            # score_line(m, b, distance_matrix, scalar, show_plot=True)
        # tic.tic(f'i: {i}')
    # tic.tic('recurrence loop finished')
    # print(f'{n_units} units, mean distance = {np.mean(mean_DM)}')
    scores[np.where(np.isnan(scores))] = np.max(scores) if DISTANCE_METRIC else 0
    mean_select = np.argsort(np.mean(scores, axis=0))[::1 if DISTANCE_METRIC else -1]
    slope_select = np.argsort(abs(np.mean(slopes > 1, axis=0) - .5))
    multi_select = np.argsort(np.argsort(slope_select) + np.argsort(mean_select))

    selection = multi_select[np.argmax([activity_list[idx].shape[1] for idx in multi_select[:20]])]
    for idx in multi_select[:20]:
        if activity_list[idx].shape[1] > 100:
            selection = idx
            break

    return [np.mean(scores[:, selection]), np.std(scores[:, selection]),
            [selection, slopes[:, selection], intercepts[:, selection], scores[:, selection]]]


def recurrence_known(activity_list, selection, known_intercepts=None, known_slopes=None, show_plots=True):
    """
    Main function of this script, everything else is called from here
    """
    # tic.tic('start recurrence function')
    n = len(activity_list)
    _, scalar = make_relu(activity_list)
    n_units = activity_list[0].shape[0]
    mean_DM = np.zeros([n])
    scores = np.zeros([n])
    slopes = np.zeros([n])
    intercepts = np.zeros([n])
    for i in range(n):
        b = known_intercepts[i] if known_intercepts is not None else None
        m = known_slopes[i] if known_slopes is not None else None
        distance_matrix = manhattan_distance_matrix(activity_list[i], activity_list[selection], bounds=True,
                                                    clip_long=True, distribute=True)

        # plt.imshow(distance_matrix.T, origin='lower')
        # plt.ylabel(f'{selection} (reference)')
        # plt.xlabel(i)
        # plt.show()

        score, m, b = regression(distance_matrix, scalar, show_plots=show_plots, show_final=False,
                                 intercept=b, slope=m)
        if score>.55:
            print()
        scores[i] = score
        slopes[i] = m
        intercepts[i] = b
        mean_DM[i] = np.mean(distance_matrix)

        # tic.tic(f'i: {i}')
    # tic.tic('recurrence loop finished')

    return [np.mean(scores), np.std(scores), [selection, slopes, intercepts, scores]]
