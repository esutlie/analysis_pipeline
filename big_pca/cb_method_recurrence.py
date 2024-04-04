import matplotlib.pyplot as plt
import numpy as np
from backend import make_color_gradient

DISTANCE_METRIC = False
arg_func = np.argmin if DISTANCE_METRIC else np.argmax
num_func = np.min if DISTANCE_METRIC else np.max


def make_relu(activity_list):
    max_length = np.max([arr.shape[1] for arr in activity_list])
    x = np.arange(max_length)
    scalar = 1 - x / np.quantile([arr.shape[1] for arr in activity_list], .9)
    scalar[scalar < .1] = .1
    return scalar


def sigmoid(x, plots=False):
    if plots:
        plt.plot(x, 1 / (1 + np.exp(-(x - .7) * 10)))
        plt.show()

    return 1 / (1 + np.exp(-(x - .7) * 10))


def manhattan_distance_matrix(a, b, bounds=False, clip_long=False, distribute=False):
    if clip_long:
        if len(a.T) > len(b.T) * 3:
            a = a[:, :len(b.T) * 3].copy()
        elif len(b.T) > len(a.T) * 3:
            b = b[:, :len(a.T) * 3].copy()
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


def score_line(m, b, arr, scalar, show_plot=False):
    result = None
    try:
        x = np.arange(arr.shape[0])
        y = line_maker(m, b, x)
        y_lower = np.floor(y).astype(int)
        y_upper = np.ceil(y).astype(int)
        y_weights = y - y_lower

        valid = (y_lower < arr.shape[1]) & (y_upper < arr.shape[1]) & (y_lower > 0) & (y_upper > 0)
        if not np.any(valid):
            return arr.max() if DISTANCE_METRIC else 0
        y_lower = y_lower[valid]
        y_upper = y_upper[valid]
        y_weights = y_weights[valid]
        x = x[valid]

        if show_plot:
            plt.plot(arr[x, y_lower])
            plt.plot(arr[x, y_upper])
            plt.show()
        result = np.sum((arr[x, y_lower] * (1 - y_weights) + arr[x, y_upper] * y_weights)
                        * scalar[np.where(valid)]) / np.sum(scalar[np.where(valid)])
    except Exception as e:
        print(e)
        print()

    return result


def starter_line(arr, scalar, show_plot=False):
    slopes = [4, 3, 2, 1.3, .8, .5, .25]
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


def gradient_descent(m, b, arr, scalar, show_plots=False, show_progress=False, show_steps=False):
    colors = make_color_gradient([222, 15, 0], [64, 4, 0], 5)
    intercept_delta = arr.shape[0] / 20
    slope_multiplier = 1.1
    learning_rate = 1

    b_delta_list = [-4 * intercept_delta, -2 * intercept_delta, -intercept_delta / 2, 0, intercept_delta / 2,
                    2 * intercept_delta, 4 * intercept_delta]
    m_multiplier_list = [1 / (slope_multiplier ** 2), 1 / slope_multiplier, .99, 1, 1.01, slope_multiplier,
                         slope_multiplier ** 2]
    m_history = [m]
    b_history = [b]
    for i in range(100):
        if show_steps:
            plt.imshow(arr.T, origin='lower')
            y_lim = plt.gca().get_ylim()
            plt.plot(line_maker(m, b, np.arange(arr.shape[0])), c='k')
            [plt.plot(line_maker(m, b + val, np.arange(arr.shape[0])), zorder=0) for val in b_delta_list]
            scores = [score_line(m, b + val, arr, scalar) for val in b_delta_list]
            b += learning_rate * b_delta_list[arg_func(scores)]
            plt.plot(line_maker(m, b, np.arange(arr.shape[0])), c='r')
            plt.ylim(y_lim)
            plt.show()

            plt.imshow(arr.T, origin='lower')
            y_lim = plt.gca().get_ylim()
            plt.plot(line_maker(m, b, np.arange(arr.shape[0])), c='k')
            [plt.plot(line_maker(m * val, b - (m * (val - 1) * arr.shape[0] / 2), np.arange(arr.shape[0])), zorder=0)
             for val in m_multiplier_list]
            scores = [score_line(m * val, b - (m * (val - 1) * arr.shape[0] / 2), arr, scalar) for val in
                      m_multiplier_list]
            mult_to_use = m_multiplier_list[arg_func(scores)]
            b -= (m * (mult_to_use - 1) * arr.shape[0] / 2)
            m *= mult_to_use
            plt.plot(line_maker(m, b, np.arange(arr.shape[0])), c='r')
            plt.ylim(y_lim)
            plt.show()

            plt.imshow(arr.T, origin='lower')
            y_lim = plt.gca().get_ylim()
            plt.plot(line_maker(m, b, np.arange(arr.shape[0])), c='k')
            [plt.plot(line_maker(m * val, b, np.arange(arr.shape[0])), zorder=0) for val in m_multiplier_list]
            scores = [score_line(m * val, b, arr, scalar) for val in m_multiplier_list]
            mult_to_use = m_multiplier_list[arg_func(scores)]
            m *= mult_to_use
            plt.plot(line_maker(m, b, np.arange(arr.shape[0])), c='r')
            plt.ylim(y_lim)
            plt.show()
        else:
            scores = [score_line(m, b + val, arr, scalar) for val in b_delta_list]
            b += learning_rate * b_delta_list[arg_func(scores)]
            scores = [score_line(m * val, b - (m * (val - 1) * arr.shape[0] / 2), arr, scalar) for val in
                      m_multiplier_list]
            mult_to_use = m_multiplier_list[arg_func(scores)]
            b -= (m * (mult_to_use - 1) * arr.shape[0] / 2)
            m *= mult_to_use
            scores = [score_line(m * val, b, arr, scalar) for val in m_multiplier_list]
            mult_to_use = m_multiplier_list[arg_func(scores)]
            m *= mult_to_use

        m = m if m > .2 else .2
        m_history.append(m)
        b_history.append(b)

        if len(m_history) > 5:
            m_history = m_history[-5:]
            b_history = b_history[-5:]

        if show_progress:
            plt.imshow(arr.T, origin='lower')
            y_lim = plt.gca().get_ylim()
            for j in range(len(m_history)):
                plt.plot(line_maker(m_history[j], b_history[j], np.arange(arr.shape[0])), c=colors[j])
            plt.ylim(y_lim)
            plt.show()

        if np.all(np.array(m_history) == m_history[0]) and np.all(np.array(b_history) == b_history[0]):
            score = score_line(m, b, arr, scalar)
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


def regression(arr_original, scalar, show_plots=False, show_final=False):
    if DISTANCE_METRIC:
        arr = arr_original
    else:
        arr_original -= np.min(arr_original)
        arr_original /= np.max(arr_original)
        arr_original = 1 - arr_original
        arr = sigmoid(arr_original)
    # xv, yv = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]), indexing='ij')

    # reg = LinearRegression(positive=True).fit(xv.flatten().reshape(-1, 1), yv.flatten(), arr.flatten())
    # slope = reg.coef_[0]
    # intercept = reg.intercept_

    slope = starter_line(arr, scalar, show_plot=show_plots)
    intercept = 0
    # plt.imshow(arr.T, origin='lower')
    # plt.plot(line_maker(slope, intercept, np.arange(arr.shape[0])), 'k')
    # plt.show()

    m, b = gradient_descent(slope, intercept, arr, scalar, show_plots=show_plots)
    score = score_line(m, b, arr, scalar, show_plot=False)
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
    # tic.tic('start recurrence function')
    n = len(activity_list)
    scalar = make_relu(activity_list)
    n_units = activity_list[0].shape[0]
    mean_DM = np.zeros([n, n])
    scores = np.zeros([n, n])
    slopes = np.zeros([n, n])
    intercepts = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if j <= i or activity_list[i].shape[1] == 0 or activity_list[j].shape[1] == 0:
                continue
            # print(f'i: {i}')
            # print(f'j: {j}')
            distance_matrix = manhattan_distance_matrix(activity_list[i], activity_list[j], bounds=True,
                                                        clip_long=True, distribute=True)
            # if distance_matrix.shape[0] < distance_matrix.shape[1]:
            #     distance_matrix = distance_matrix.T

            # plt.imshow(distance_matrix.T, origin='lower')
            # plt.show()

            score, m, b = regression(distance_matrix, scalar, show_plots=show_plots)

            scores[i, j] = score
            slopes[i, j] = m
            intercepts[i, j] = b
            mean_DM[i, j] = np.mean(distance_matrix)

            scores[j, i] = score
            slopes[j, i] = m
            intercepts[j, i] = b
            mean_DM[j, i] = np.mean(distance_matrix)
        # tic.tic(f'i: {i}')
    # tic.tic('recurrence loop finished')
    print(f'{n_units} units, mean distance = {np.mean(mean_DM)}')
    scores[np.where(np.isnan(scores))] = np.max(scores) if DISTANCE_METRIC else 0
    mean_select = np.argsort(np.mean(scores, axis=0))[::1 if DISTANCE_METRIC else -1]
    slope_select = np.argsort(abs(np.median(slopes, axis=0) - 1), axis=0)
    multi_select = np.argsort(np.argsort(slope_select) + np.argsort(mean_select))

    # for select in multi_select[:3]:
    #     select_slopes = slopes[select]
    #     select_intercepts = intercepts[select]
    #     fig, axes = plt.subplots(3, 1, figsize=(6, 8))
    #     log_bins = np.logspace(np.log10(.001), np.log10(5), 50)
    #     axes[0].hist(select_slopes, bins=log_bins)
    #     axes[0].set_title('Slopes Histogram')
    #     axes[0].set_xlim([.1, 10.1])
    #     axes[0].set_ylim([0, 50])
    #     axes[0].set_xscale('log')
    #     # axes[0].set_xticks([.1, 1, 10])
    #     fig.canvas.draw()
    #     axes[0].set_xticklabels([.001, .1, 1, 10])
    #
    #     axes[1].hist(select_intercepts, bins=50, range=[-100, 100])
    #     axes[1].set_title('Intercepts Histogram')
    #     axes[1].set_xlim([-100, 100])
    #     axes[1].set_ylim([0, 100])
    #     axes[2].scatter(select_intercepts, select_slopes)
    #     axes[2].set_title(f'Mean Score: {np.mean(scores[select])}')
    #     axes[2].set_ylabel('Slopes')
    #     axes[2].set_xlabel('Intercepts')
    #     axes[2].set_xlim([-100, 100])
    #     axes[2].set_ylim([-.5, 5])
    #     plt.tight_layout()
    #     plt.show()
    selection = multi_select[np.argmax([activity_list[idx].shape[1] for idx in multi_select[:20]])]
    for idx in multi_select[:20]:
        if activity_list[idx].shape[1] > 100:
            selection = idx
            break
    #
    # for i in range(n):
    #     distance_matrix = manhattan_distance_matrix(activity_list[multi_select[0]], activity_list[i], bounds=True,
    #                                                 clip_long=True)
    #     score, m, b = regression(distance_matrix, show_plots=True)

    return [np.mean(scores[selection]), np.std(scores[selection]),
            [selection, slopes[selection], intercepts[selection], scores[selection]]]

