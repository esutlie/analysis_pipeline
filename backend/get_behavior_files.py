import os
from os import walk
from .get_data_path import get_behavior_path


def get_behavior_files():
    file_paths = []
    data_path = get_behavior_path()
    for root, dirs, filenames in walk(os.path.join(data_path, 'data')):
        if len(dirs) == 0 and os.path.basename(root)[:2] == 'ES':
            mouse = os.path.basename(root)
            for f in filenames:
                file_paths.append(os.path.join(mouse, f))
    return file_paths
