import os


def get_session_list():
    """
    get list of locally stored sessions for iterating through file paths
    :return: list
    """
    directory_list = []
    local_path = os.path.join(os.path.dirname(os.getcwd()), 'processed_data')

    # Walk through directory
    for root, directories, files in os.walk(local_path):
        for d in directories:
            directory_list.append(d)
        break
    return directory_list


if __name__ == '__main__':
    file_list = get_session_list()
