import json
import os
from backend import get_data_path


def get_info(session):
    with open(os.path.join(get_data_path(), session, 'info.json'), 'r') as f:
        info = json.load(f)
    return info
