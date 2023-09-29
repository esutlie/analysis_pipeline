import os
import backend
# from get_data_path import get_data_path
# from get_session_list import get_session_list
import pandas as pd


def load_data(session, photometry=False):
    local_path = os.path.join(backend.get_data_path(photometry=photometry), session)
    pi_events = pd.read_pickle(os.path.join(local_path, 'pi_events.pkl'))

    if photometry:
        neural_events = pd.read_pickle(os.path.join(local_path, 'neural_events.pkl'))
        return neural_events, pi_events, None

    spikes = pd.read_pickle(os.path.join(local_path, 'spikes.pkl'))
    cluster_info = pd.read_pickle(os.path.join(local_path, 'cluster_info.pkl'))
    return spikes, pi_events, cluster_info


if __name__ == '__main__':
    session = backend.get_session_list()[0]
    spikes, pi_events, cluster_info = load_data(session)
