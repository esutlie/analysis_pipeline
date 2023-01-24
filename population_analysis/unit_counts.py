import numpy as np

import backend
import os
import numpy


def unit_counts():
    session_list = backend.get_session_list()
    num_units = []
    mouse_id = []
    for session in session_list:
        spikes, pi_events, cluster_info = backend.load_data(session)
        mouse = session[:5]
        print(f'{session} has {len(cluster_info)} units')
        num_units.append(len(cluster_info))
        mouse_id.append(mouse)
    num_units = np.array(num_units)
    mouse_id = np.array(mouse_id)
    for mouse in np.unique(mouse_id):
        total = sum(num_units[np.where(mouse_id == mouse)])
        print(f'{mouse} has {total} total units')


if __name__ == '__main__':
    unit_counts()
