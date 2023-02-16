from trial_leave_times import trial_leave_times
import backend


def leave_time_stats(exit_data):
    exit_data['block_categorical'] = [str(val) for val in exit_data.block_labels]
    backend.t_test(exit_data, 'leave time', 'block_labels')
    backend.t_test(exit_data, 'leave time from last reward', 'block_labels')
    # backend.anova(exit_data, 'leave time', 'block_categorical', 'mouse')
    # backend.anova(exit_data, 'leave time from last reward', 'block_categorical', 'mouse')


if __name__ == '__main__':
    files = backend.get_session_list()
    data = [backend.load_data(session)[1] for session in files]
    full_exit_data = trial_leave_times(files, data, save=True)
    leave_time_stats(full_exit_data)
