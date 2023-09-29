import os


def get_data_path(photometry=False):
    if photometry:
        return os.path.join(os.path.dirname(os.getcwd()), 'z_photometry')
    return os.path.join(os.path.dirname(os.getcwd()), 'z_sessions')


def get_pi_path():
    return os.path.join(os.path.dirname(os.getcwd()), 'z_behavior')


def get_behavior_path():
    return os.path.join('C:\\', 'Users', 'Elissa', 'GoogleDrive', 'Code', 'Python', 'behavior_code')


def get_probe_info_path():
    return os.path.join(os.path.dirname(os.getcwd()), 'image_editing', 'probe_info')
