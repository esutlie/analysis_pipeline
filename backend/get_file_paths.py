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
