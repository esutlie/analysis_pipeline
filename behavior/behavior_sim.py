from behavior import trial_leave_times
import backend

def behavior_sim():
    files = backend.get_session_list()
    data = [backend.load_data(session)[1] for session in files]
    leave_time_df = trial_leave_times(files, data, save=False, data_only=True)
    print('test')



if __name__ == '__main__':
    behavior_sim()
