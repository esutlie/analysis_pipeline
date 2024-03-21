import concurrent
import numpy as np
import backend
from big_pca.cb_method_pca import pca_scoring
from functools import partial


def leave_one_out(normalized_spikes, intervals_df):
    executor = concurrent.futures.ProcessPoolExecutor(10)
    mean_progress = []
    std_progress = []
    leave_out_list = []
    best_pca = []
    for i in range(normalized_spikes.shape[0] - 1):
        print(f'round {i} of leave one out')
        mean_scores = []
        std_scores = []
        index_list = []
        pca_list = []
        tic = backend.Timer()
        func = partial(pca_scoring, normalized_spikes=normalized_spikes, intervals_df=intervals_df,
                       leave_out_list=leave_out_list)
        futures = [executor.submit(func, j) for j in range(normalized_spikes.shape[0])]
        concurrent.futures.wait(futures)
        tic.tic('concurrent processing')
        for future in futures:
            if future.result() is None:
                continue
            result = future.result()
            mean_scores.append(result[0])
            std_scores.append(result[1])
            index_list.append(result[2])
            pca_list.append(result[3])
        leave_out_list.append(index_list[np.argmax(mean_scores)])
        mean_progress.append(np.max(mean_scores))
        std_progress.append(std_scores[np.argmax(mean_scores)])
        best_pca.append(pca_list[np.argmax(mean_scores)])
        print(leave_out_list)
        print(mean_progress)
        print(std_progress)
    print()
