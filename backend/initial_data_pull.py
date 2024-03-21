import os
import shutil
from get_data_path import get_data_path
from get_file_paths import get_directories


def initial_data_pull():
    dest = get_data_path()
    source = os.path.join('C:\\', 'processed_data')
    shutil.copytree(source, dest)


def pull_file(file_name):
    dest = get_data_path()
    source = os.path.join('C:\\', 'processed_data')
    dirs = get_directories(dest)
    for dir in dirs:
        file_dest = os.path.join(dest, dir, file_name)
        file_source = os.path.join(source, dir, file_name)
        if os.path.exists(file_source):
            shutil.copy(file_source, file_dest)
        else:
            print(f'{file_source} doesnt exist')


if __name__ == '__main__':
    initial_data_pull()
    # pull_file('cluster_info.pkl')
