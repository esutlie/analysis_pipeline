"""
This file should make up a complete set of population analysis plots for each session.
"""

import backend

from population_analysis.old_stuff.isomap import leave_one_out, plot_scores, pretty_plot

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
