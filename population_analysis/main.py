"""
This file should make up a complete set of population analysis plots for each session.
"""

from sklearn.manifold import Isomap
from create_bins_df import create_precision_df, get_gaussian_kernel
import backend
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression, PoissonRegressor, GammaRegressor
import json
import os
from backend import get_data_path
import time
from isomap import get_phase, get_block
from thread_function import get_x
from sklearn.preprocessing import minmax_scale
from scipy.ndimage import convolve1d
import math
from center_of_mass import get_mean

from isomap import leave_one_out, plot_scores, pretty_plot

def main():
    color_sets = backend.get_color_sets()
    files = backend.get_session_list()
    for session in files:
        if session != 'ES029_2022-09-14_bot72_0_g0':
            continue
        leave_one_out(session, multi_core=True)
        plot_scores(session)
        pretty_plot(session)





if __name__ == '__main__':
    main()
