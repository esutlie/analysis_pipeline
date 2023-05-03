import backend
from behavior.trial_leave_times import trial_leave_times
from datetime import date
import os


def multi_vs_single():
    mouse_list = ['ES024', 'ES025']
    start_date = date.fromisoformat('2023-04-04')
    files = backend.get_session_list()
    files = [f for f in files if os.path.basename(f)[:4] == 'data']
    mouse_files = [
        [f for f in files if (f[:5] == mouse) and (date.fromisoformat(os.path.basename(f)[6:16]) >= start_date)] for
        mouse in mouse_list]
    data = [[backend.load_data(session)[1] for session in mouse] for mouse in mouse_files]
    leave_time_df = [trial_leave_times(files, mouse_data, save=False, data_only=True) for mouse_data in data]
    '''
    continue working on this once you have sorted the sessions for it
    '''


if __name__ == '__main__':
    multi_vs_single()
