import pandas as pd
import os
import json


def load_pi_data(path):
    pi_events = pd.read_pickle(os.path.join(path, 'pi_events.pkl'))
    with open(os.path.join(path, 'info.json'), "r") as info_file:
        info = json.load(info_file)
    return pi_events, info
