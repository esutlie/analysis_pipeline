import matplotlib.pyplot as plt
from matplotlib import lines
import seaborn as sns
import backend

set2 = sns.color_palette(palette='Set2')


def trial_raster():
    return {
        'Reward': 'purple',
        'Spikes (.8r/s block)': set2[0],
        'Spikes (.4r/s block)': set2[1],
        'Exponential Port Entry': 'r',
        'Exponential Port Exit': 'k',
        # 'Background Port Entry': 'g',
        # 'Background Port Exit': 'b',
    }


def reward_raster():
    return {
        'Spikes (.8r/s block)': set2[0],
        'Spikes (.4r/s block)': set2[1],
        'Rewarded Lick': 'red',
        'Exponential Port Exit': 'k',
    }


def separate_legend(entries, file_name=None):
    properties = {
        'marker': '|',
        'linestyle': 'None',
        'markersize': 15,
        'markeredgewidth': 2.5,
    }
    width = 1 + .12 * max([len(key) for key in entries.keys()])
    height = .3 + .35 * len(entries.keys())
    fig, ax = plt.subplots(1, 1, figsize=[width, height])
    lines_for_legend = []
    for key, value in entries.items():
        lines_for_legend.append(lines.Line2D([], [], color=value, marker=properties['marker'],
                                             linestyle=properties['linestyle'],
                                             markersize=properties['markersize'],
                                             markeredgewidth=properties['markeredgewidth'], label=key))
    ax.legend(handles=lines_for_legend, loc='center', prop={'size': 15})
    plt.axis('off')
    if file_name:
        backend.save_fig(fig, file_name)
    else:
        plt.show()


if __name__ == '__main__':
    save_legend = True
    separate_legend(trial_raster(), file_name='legend_trial_raster.png')
    separate_legend(reward_raster(), file_name='legend_reward_raster.png')
