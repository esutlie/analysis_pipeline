import os


def get_file_paths(directory):
    # List to store paths
    file_paths = []
    file_names = []

    # Walk through directory
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Add filename to list
            file_paths.append(os.path.join(root, filename))
            file_names.append(filename)

            # Return all paths
    return file_paths, file_names


def get_directories(directory, top_level_only=True):
    # List to store paths
    directory_list = []

    # Walk through directory
    for root, directories, files in os.walk(directory):
        for d in directories:
            # Add filename to list
            directory_list.append(d)
        if top_level_only:
            break

            # Return all paths
    return directory_list
