import pickle
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import backend

mpl.use('Qt5Agg')

sides = {
    'ES024': 'left',
    'ES025': 'right',
    'ES029': 'right',
    'ES030': 'right',
    'ES031': 'left',
    'ES032': 'left',
}
surface_row = {
    'ES024': 370,
    'ES025': 325,
    'ES029': 1,
    'ES030': 320,
    'ES031': 350,
    'ES032': 345,
}

bot_row = {
    'ES024': 192,
    'ES025': 170,
    'ES029': 72,
    'ES030': 170,
    'ES031': 170,
    'ES032': 170,
}


def analyze_probe():
    save_dir = os.path.join(os.getcwd(), 'probe_info')
    for mouse in sides.keys():
        columns = ['row', 'shank', 'coords', 'area', 'area_code']
        probe_df = pd.DataFrame(columns=columns)
        for shank in range(4):
            path = os.path.join(os.getcwd(), 'reg_pics', mouse, 'probes', f'probe {shank}.pkl')
            if os.path.exists(path):
                with open(path, 'rb') as fp:
                    dic = pickle.load(fp)

                insertion_coords = dic['data']['insertion_coords']
                termination_coords = dic['data']['terminus_coords']
                in_brain_length = dic['data']['probe_length']
                region_lengths = dic['data']['region_length']
                row_labels = dic['data']['sites_label'][0]  # area code for every site
                brain_areas = dic['data']['label_name']  # area names, get codes from area_codes
                area_codes = dic['data']['region_label']  # area code for each area in brain_areas
                row_coords = dic['data']['sites_loc_b'][0]
                implant_side = sides[mouse]
                if np.mean(row_coords[:, 0]) > 0 and implant_side == 'left':
                    row_coords[:, 0] *= -1
                if np.mean(row_coords[:, 0]) < 0 and implant_side == 'right':
                    row_coords[:, 0] *= -1
                for row in range(len(row_labels)):
                    area = brain_areas[np.where(row_labels[row] == area_codes)[0][0]]
                    probe_df.loc[len(probe_df.index)] = [row, shank, row_coords[row], area, row_labels[row]]

                # total_rows = len(row_labels)
                # measured_rows = surface_row[mouse]
                # print(f'{mouse} {total_rows}/{measured_rows}={total_rows / measured_rows}')

                # fig = plt.figure()
                # ax = fig.add_subplot(projection='3d')
                # for i in range(4):
                #     ax.scatter(site_coords[i][:5, 0], site_coords[i][:5, 1], site_coords[i][:5, 2])
                #     plt.show()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        probe_df.to_pickle(os.path.join(save_dir, f'{mouse}.pkl'))


def add_to_data():
    files = backend.get_session_list()
    for session in files:
        mouse = session[:5]
        probe_df_path = os.path.join(os.getcwd(), 'probe_info', f'{mouse}.pkl')
        local_path = os.path.join(backend.get_data_path(), session, 'cluster_info.pkl')
        if not os.path.exists(probe_df_path):
            continue

        probe_df = pd.read_pickle(probe_df_path)
        spikes, pi_events, cluster_info = backend.load_data(session)
        if not len(cluster_info) or not len(probe_df):
            print(f'{mouse} not done')
            continue
        print(f'{mouse} done')

        cluster_info['area'] = cluster_info.apply(
            lambda x: probe_df[(probe_df['shank'] == x['shank']) &
                               (probe_df['row'] == x['row'])]['area'].values[0], axis=1)
        cluster_info['area_code'] = cluster_info.apply(
            lambda x: probe_df[(probe_df['shank'] == x['shank']) &
                               (probe_df['row'] == x['row'])]['area_code'].values[0], axis=1)
        cluster_info['coords'] = cluster_info.apply(
            lambda x: probe_df[(probe_df['shank'] == x['shank']) &
                               (probe_df['row'] == x['row'])]['coords'].values[0], axis=1)
        cluster_info.to_pickle(local_path)


if __name__ == '__main__':
    # analyze_probe()
    add_to_data()
