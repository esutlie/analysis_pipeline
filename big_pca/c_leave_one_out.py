import concurrent
import numpy as np
import pandas as pd
import backend
from big_pca.cb_method_pca import pca_scoring
from big_pca.cb_method_recurrence import recurrence, arg_func, num_func, DISTANCE_METRIC
from big_pca.ca_extract_intervals import extract_intervals
from functools import partial
import os
import matplotlib.pyplot as plt


def leave_one_out(normalized_spikes, intervals_df, session, process=False, test=False, show_plots=False):
    func = recurrence

    n_units = normalized_spikes.shape[0]
    columns = ['num_units', 'mean_score', 'std_score', 'arr', 'leave_out']
    save_path = os.path.join('results', session + '_recurrence_leave_out.pkl')
    if os.path.exists(save_path) and not test:
        res_df = pd.read_pickle(save_path)
        # '''temp section'''
        # arr_list = calc_scores(res_df, normalized_spikes, intervals_df)
        # res_df = res_df.assign(arr2=arr_list)
        # res_df.to_pickle(save_path)
        # '''end of temp section'''
        leave_out_list = res_df.leave_out.iloc[-1]
    else:
        tic = backend.Timer()
        initial = one_out_set_up(None, func, normalized_spikes, intervals_df, [], test=test, show_plots=show_plots)
        leave_out_list = []
        res_df = pd.DataFrame([[n_units, initial[0], initial[1], initial[2], []]], columns=columns)
        if not test:
            res_df.to_pickle(save_path)
        tic.tic('finished initial')
    if test:
        normalized_spikes = normalized_spikes[:5]
    if process:
        executor = concurrent.futures.ProcessPoolExecutor(10)
    i = 1
    while len(leave_out_list) < n_units - 1:
        print(f'round {i} of leave one out')
        mean_scores = []
        std_scores = []
        index_list = []
        arr_list = []
        tic = backend.Timer()

        # func = pca_scoring
        func_partial = partial(one_out_set_up, func=func, normalized_spikes=normalized_spikes,
                               intervals_df=intervals_df, leave_out_list=leave_out_list, test=test,
                               show_plots=show_plots)
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
                arr_list.append(result[2] + [np.array(result[4])])
                index_list.append(result[3])
        else:
            results = [func_partial(j) for j in range(n_units)]
            for result in results:
                mean_scores.append(result[0])
                std_scores.append(result[1])
                arr_list.append(result[2] + [np.array(result[4])])
                index_list.append(result[3])
        index_list = np.array(index_list)
        try:
            res_df = pd.concat([res_df, pd.DataFrame([[n_units - len(leave_out_list) - 1,
                                                       num_func(mean_scores),
                                                       std_scores[arg_func(mean_scores)],
                                                       arr_list[arg_func(mean_scores)],
                                                       leave_out_list + [index_list[arg_func(mean_scores)]]]],
                                                     columns=columns)], axis=0)
        except Exception as e:
            print(e)
            print()
        if not test:
            res_df.to_pickle(save_path)

        leave_out_order = np.argsort(mean_scores)[::1 if DISTANCE_METRIC else -1]
        if len(leave_out_order) > 100:
            leave_out_list += index_list[leave_out_order[:15]].tolist()
        elif len(leave_out_order) > 40:
            leave_out_list += index_list[leave_out_order[:8]].tolist()
        elif len(leave_out_order) > 15:
            leave_out_list += index_list[leave_out_order[:3]].tolist()
        elif len(leave_out_order) > 8:
            leave_out_list += index_list[leave_out_order[:2]].tolist()
        else:
            leave_out_list.append(index_list[leave_out_order[0]])

        print(leave_out_list)
        print(res_df.mean_score)
        print(res_df.std_score)
        i += 1
    num_units = [n_units - len(val) for val in res_df.leave_out.values]
    plt.plot(num_units, res_df.mean_score.values * 100)
    # plt.ylabel('Mean Trajectory Distance')
    plt.ylabel('Mean Trajectory Overlap (percent)')
    plt.xlabel('Num Units Included')
    plt.show()
    print()


def one_out_set_up(j, func, normalized_spikes, intervals_df, leave_out_list, test=False, show_plots=False):
    tic = backend.Timer()
    if j in leave_out_list:
        return None
    if test:
        intervals_df = intervals_df.loc[:50]
    if j is None:
        one_out_spikes = np.delete(normalized_spikes, leave_out_list, axis=0)
    elif j >= 0:
        one_out_spikes = np.delete(normalized_spikes, [j] + leave_out_list, axis=0)
    else:
        one_out_spikes = normalized_spikes

    interval_spikes, intervals_df = extract_intervals(one_out_spikes, intervals_df)
    activity_list = intervals_df.activity.to_list()
    start_list = intervals_df.start.to_list()
    res = func(activity_list, show_plots=show_plots)
    tic.tic(f'finished one out for unit {j}')
    return res + [j] + [start_list]