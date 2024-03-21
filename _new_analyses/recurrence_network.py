import pandas as pd
import numpy as np
import os


def main():
    data_path = os.path.abspath(os.path.join(os.getcwd(), '..', '_master', 'data', 'master_data.pkl'))
    df = pd.read_pickle(data_path)
    sessions = np.unique(df.session)
    session = sessions[0]

    # Just pull a single session to save and use as an example
    session_df = df[df.session == session]
    time_series = np.concatenate(session_df.bins500ms, axis=1)
    interval_ids = np.concatenate([[i] * len(row.bins500ms[0]) for i, row in session_df.iterrows()])
    interval_time = np.concatenate(
        [np.linspace(0, (len(row.bins500ms[0]) - 1) * .05, len(row.bins500ms[0])) for i, row in
         session_df.iterrows()])

    np.save('time_series', time_series)
    np.save('interval_ids', interval_ids)
    np.save('interval_time', interval_time)


if __name__ == '__main__':
    main()
