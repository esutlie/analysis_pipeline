import os
import backend

def get_session_list(photometry=False):
    """
    get list of locally stored sessions for iterating through file paths
    :return: list
    """
    directory_list = []
    local_path = backend.get_data_path(photometry=photometry)
    # local_path = os.path.join(os.path.dirname(os.getcwd()), 'z_sessions')

    # Walk through directory
    for root, directories, files in os.walk(local_path):
        for d in directories:
            directory_list.append(d)
        break
    return directory_list


if __name__ == '__main__':
    file_list = get_session_list()
