import concurrent
import numpy as np
import backend
from big_pca.cb_method_pca import pca_scoring
from big_pca.cb_method_recurrence import recurrence, arg_func, num_func
from big_pca.ca_extract_intervals import extract_intervals
from functools import partial

import matplotlib.pyplot as plt


def leave_one_out(normalized_spikes, intervals_df, process=False, test=False):
    func = recurrence
    if test:
        normalized_spikes = normalized_spikes[:4]
    if process:
        executor = concurrent.futures.ProcessPoolExecutor(10)
    initial = one_out_set_up(-1, func, normalized_spikes, intervals_df, [], test=test)
    mean_progress = [initial[0]]
    std_progress = [initial[1]]
    arr_best = [initial[2]]
    leave_out_list = [initial[3]]

    for i in range(normalized_spikes.shape[0] - 1):
        print(f'round {i} of leave one out')
        mean_scores = []
        std_scores = []
        index_list = []
        arr_list = []
        tic = backend.Timer()

        # func = pca_scoring
        func_partial = partial(one_out_set_up, func=func, normalized_spikes=normalized_spikes,
                               intervals_df=intervals_df, leave_out_list=leave_out_list, test=test)
        if process:
            futures = [executor.submit(func_partial, j) for j in range(normalized_spikes.shape[0])]
            concurrent.futures.wait(futures)
            tic.tic('concurrent processing')
            for future in futures:
                if future.result() is None:
                    continue
                result = future.result()
                mean_scores.append(result[0])
                std_scores.append(result[1])
                arr_list.append(result[2])
                index_list.append(result[3])
        else:
            results = [func_partial(j) for j in range(normalized_spikes.shape[0])]
            for result in results:
                mean_scores.append(result[0])
                std_scores.append(result[1])
                arr_list.append(result[2])
                index_list.append(result[3])

        leave_out_list.append(index_list[arg_func(mean_scores)])
        mean_progress.append(num_func(mean_scores))
        std_progress.append(std_scores[arg_func(mean_scores)])
        arr_best.append(arr_list[arg_func(mean_scores)])
        print(leave_out_list)
        print(mean_progress)
        print(std_progress)
    num_units = np.linspace(len(mean_progress), 1, len(mean_progress))
    plt.plot(num_units, np.array(mean_progress))
    plt.ylabel('Mean Trajectory Distance')
    # plt.ylabel('Mean Trajectory Overlap (percent)')
    plt.xlabel('Num Units Included')
    plt.show()
    print()


def one_out_set_up(j, func, normalized_spikes, intervals_df, leave_out_list, test=False):
    if j in leave_out_list:
        return None
    if test:
        intervals_df = intervals_df.loc[:50]
    show_plots = np.random.random() < .05
    if j >= 0:
        one_out_spikes = np.delete(normalized_spikes, [j] + leave_out_list, axis=0)
    else:
        one_out_spikes = normalized_spikes
    interval_spikes, intervals_df = extract_intervals(one_out_spikes, intervals_df)
    activity_list = intervals_df.activity.to_list()
    res = func(activity_list, show_plots=False)
    return res + [j]
