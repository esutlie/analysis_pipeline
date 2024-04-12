import pandas as pd
import backend
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import datetime
from itertools import combinations

THRESHOLD = 1.5


# matplotlib.use('GTK3Agg')


def get_unit_book(regen=False):
    save_path = os.path.join(os.getcwd(), 'unit_book.pkl')
    if os.path.exists(save_path) and not regen:
        unit_book = pd.read_pickle(save_path)
        return unit_book

    columns = ['global_id', 'consolidated_id', 'mouse', 'session', 'recording_block', 'date', 'num_in_day', 'structure']
    columns2 = ['id', 'amp', 'ch', 'depth', 'fr', 'group', 'n_spikes', 'shank', 'row']
    unit_book = pd.DataFrame(columns=columns + columns2)
    files = backend.get_session_list()
    global_id = 0
    consolidated_id = -1
    structure = 'None'
    for i, session in enumerate(files):
        [mouse, date, num_in_day, recording_block] = backend.unpack_session_name(session)
        spikes, pi_events, cluster_info = backend.load_data(session)
        for _, row in cluster_info.iterrows():
            global_id += 1
            data = [global_id, consolidated_id, mouse, session, recording_block, date, num_in_day, structure] + list(
                row[columns2].values)
            unit_book = pd.concat([unit_book, pd.DataFrame(data=[data], columns=columns + columns2)])
        print(f'processed session: {session} {i}/{len(files)}')
    unit_book.to_pickle(save_path)
    return unit_book


def get_session_groups():
    session_groups = {}
    files = backend.get_session_list()
    for session in files:
        [mouse, date, num_in_day, recording_block] = backend.unpack_session_name(session)
        group = f'{mouse}_{recording_block}'
        if group in session_groups.keys():
            session_groups[group].append(session)
        else:
            session_groups[group] = [session]
    return session_groups


def consolidate():
    t = backend.Timer()
    unit_book = get_unit_book()
    session_groups = get_session_groups()
    consolidated_id = 1
    for key, sessions in session_groups.items():
        for [session_1, session_2] in combinations(sessions, 2):
            print(f'{session_1}  {session_2}')
            full_template_1, template_ids_1 = backend.gen_full_template(session_1)
            full_template_2, template_ids_2 = backend.gen_full_template(session_2)
            pairs = compare(full_template_1, full_template_2)
            print(len(pairs))
            for [x, y] in pairs:
                unit_1 = template_ids_1[x][0]
                unit_2 = template_ids_2[y][0]

                cu_id_1 = unit_book[(unit_book.session == session_1) &
                                    (unit_book.id == unit_1)].consolidated_id.values[0]
                cu_id_2 = unit_book[(unit_book.session == session_2) &
                                    (unit_book.id == unit_2)].consolidated_id.values[0]

                if cu_id_1 == -1 and cu_id_2 == -1:
                    cu_id_1 = cu_id_2 = consolidated_id
                    consolidated_id += 1
                elif cu_id_1 == -1:
                    cu_id_1 = cu_id_2
                elif cu_id_2 == -1:
                    cu_id_2 = cu_id_1
                else:
                    print()
                unit_book.loc[(unit_book.session == session_1) & (unit_book.id == unit_1), 'consolidated_id'] = cu_id_1
                unit_book.loc[(unit_book.session == session_2) & (unit_book.id == unit_2), 'consolidated_id'] = cu_id_2


def compare(t1, t2):
    compare_matrix = np.zeros([len(t1), len(t2)])
    # t = backend.Timer()
    # t.tic()
    for i in range(len(t1)):
        for j in range(len(t2)):
            rolled = np.array([np.mean(abs(np.roll(t1[i], k, axis=0) - t2[j]) ** 2) for k in range(-15, 15, 1)])
            compare_matrix[i, j] = np.min(rolled) / np.mean(abs(t1[i]) + abs(t2[j]))
    pairs_list = []
    while np.min(compare_matrix) < THRESHOLD:
        x, y = np.where(compare_matrix == np.min(compare_matrix))
        x, y = x[0], y[0]
        plot_templates(t1[x], t2[y],
                       title=f'unit {x} and unit {y}   dif={compare_matrix[x, y]}')
        # ans = input(f'are {x} and {y} the same? (y/n)')
        ans = 'yes'
        if ans in ['y', 'Y', 'yes', 'Yes']:
            pairs_list.append([x, y])
            compare_matrix[x] = 10
            compare_matrix[:, y] = 10
        else:
            compare_matrix[x, y] = 10
    return pairs_list


def plot_templates(m1, m2, title='templates'):
    num_rows = 10
    rows_1 = np.sort(np.argpartition(np.mean(abs(m1), axis=0), -num_rows)[-num_rows:])
    rows_2 = np.sort(np.argpartition(np.mean(abs(m2), axis=0), -num_rows)[-num_rows:])
    rows = np.intersect1d(rows_1, rows_2)
    fig, ax = plt.subplots(1, 1, figsize=[7, 5])
    for i in range(len(rows)):
        ax.plot(m1[:, rows[i]] + i * 100, c='blue')
        ax.plot(m2[:, rows[i]] + i * 100, c='red')
    ax.set_title(title)
    plt.show()


if __name__ == '__main__':
    # unit_book_df = get_unit_book(regen=True)
    consolidate()
