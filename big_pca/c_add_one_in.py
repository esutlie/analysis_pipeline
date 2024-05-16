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


def add_one_in(normalized_spikes, intervals_df, session, process=False, test=False, show_plots=False, shuffle=False):
    func = recurrence
    n_units = normalized_spikes.shape[0]
    columns = ['num_units', 'mean_score', 'std_score', 'arr', 'add_in']
    save_path = os.path.join('results', session + '_recurrence_add_in.pkl')
    if os.path.exists(save_path) and not test:
        res_df = pd.read_pickle(save_path)
        add_in_list = res_df.add_in.iloc[-1]
    else:
        tic = backend.Timer()
        add_in_list = [0,1]
        res_df = pd.DataFrame(columns=columns)

    if test:
        normalized_spikes = normalized_spikes[:5]
    if process:
        executor = concurrent.futures.ProcessPoolExecutor(os.cpu_count()//2)
    i = 1
    while len(add_in_list) < n_units - 1:
        print(f'round {i} of add one in, {n_units} units')
        mean_scores = []
        std_scores = []
        index_list = []
        arr_list = []
        interval_start_list = []
        tic = backend.Timer()

        # func = pca_scoring
        func_partial = partial(one_in_set_up, func=func, normalized_spikes=normalized_spikes,
                               intervals_df=intervals_df, add_in_list=add_in_list, test=test,
                               show_plots=show_plots)
        if process:
            futures = [executor.submit(func_partial, j) for j in range(n_units)]
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
                if result is None:
                    continue
                mean_scores.append(result[0])
                std_scores.append(result[1])
                arr_list.append(result[2] + [np.array(result[4])])
                index_list.append(result[3])

        index_list = np.array(index_list)
        try:
            res_df = pd.concat([res_df, pd.DataFrame([[len(add_in_list) + 1,
                                                       num_func(mean_scores),
                                                       std_scores[arg_func(mean_scores)],
                                                       arr_list[arg_func(mean_scores)],
                                                       add_in_list + [index_list[arg_func(mean_scores)]]]],
                                                     columns=columns)], axis=0)
        except Exception as e:
            print(e)
            print()
        if not test:
            res_df.to_pickle(save_path)
        one_in_order = np.argsort(mean_scores)[::1 if DISTANCE_METRIC else -1]
        if len(one_in_order) > 100:
            add_in_list += index_list[one_in_order[:15]].tolist()
        elif len(one_in_order) > 40:
            add_in_list += index_list[one_in_order[:8]].tolist()
        elif len(one_in_order) > 15:
            add_in_list += index_list[one_in_order[:3]].tolist()
        elif len(one_in_order) > 8:
            add_in_list += index_list[one_in_order[:2]].tolist()
        else:
            add_in_list.append(index_list[one_in_order[0]])

        print(add_in_list)
        print(res_df.mean_score)
        # print(res_df.std_score)
        i += 1
    num_units = [len(val) for val in res_df.add_in.values]
    plt.plot(num_units, res_df.mean_score.values * 100)
    # plt.ylabel('Mean Trajectory Distance')
    plt.ylabel('Mean Trajectory Overlap (percent)')
    plt.xlabel('Num Units Included')
    plt.show()
    print()


def one_in_set_up(j, func, normalized_spikes, intervals_df, add_in_list, test=False, show_plots=False):
    tic = backend.Timer()
    tic.tic(f'starting add one in for unit {j}')
    if j in add_in_list:
        tic.tic(f'finished one in for unit {j}')
        return None
    if test:
        intervals_df = intervals_df.loc[:50]
    if j is None:
        one_in_spikes = normalized_spikes[np.array(add_in_list)]
    elif j >= 0:
        one_in_spikes = normalized_spikes[np.array([j] + add_in_list)]
    else:
        one_in_spikes = normalized_spikes[0]

    interval_spikes, intervals_df = extract_intervals(one_in_spikes, intervals_df)
    activity_list = intervals_df.activity.to_list()
    start_list = intervals_df.start.to_list()
    res = func(activity_list, show_plots=show_plots)
    tic.tic(f'finished one in for unit {j}')
    return res + [j] + [start_list]
