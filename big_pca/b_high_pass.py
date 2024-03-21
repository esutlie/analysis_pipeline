from scipy.signal import butter, sosfiltfilt


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sos


def butter_highpass_filter(data, cutoff, fs, order=5):
    sos = butter_highpass(cutoff, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def high_pass(bin_spikes, cutoff, sampling_frequency=1000):
    return butter_highpass_filter(bin_spikes, cutoff, sampling_frequency, order=5)
