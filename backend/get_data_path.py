import os


def get_data_path():
    return os.path.join(os.path.dirname(os.getcwd()), 'processed_data')
