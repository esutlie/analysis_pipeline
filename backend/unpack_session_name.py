from datetime import datetime
import pandas as pd


def unpack_session_name(session):
    mouse = session[:5]
    date = datetime.strptime(session[6:16], '%Y-%m-%d')

    recording_block = session[20:23]
    if recording_block[1] == '_':
        recording_block = '00' + recording_block[0]
    elif recording_block[2] == '_':
        recording_block = '0' + recording_block[0:2]

    num_in_day = session[24]
    return [mouse, date, num_in_day, recording_block]
