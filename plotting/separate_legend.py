import matplotlib.pyplot as plt
from matplotlib import lines
import seaborn as sns
import backend

set2 = sns.color_palette(palette='Set2')
color_sets = backend.get_color_sets()


def trial_raster():
    return {
        'Reward': 'purple',
        'Spikes (.4r/s block)': set2[0],
        'Spikes (.8r/s block)': set2[1],
        'Exponential Port Entry': 'r',
        'Exponential Port Exit': 'k',
        # 'Background Port Entry': 'g',
        # 'Background Port Exit': 'b',
    }


def reward_raster():
    return {
        'Spikes (.4r/s block)': set2[0],
        'Spikes (.8r/s block)': set2[1],
        'Rewarded Lick': 'red',
        'Exponential Port Exit': 'k',
    }


def blocks():
    return {
        '0.4 r/s': set2[0],
        '0.8 r/s': set2[1],
    }


def short_med_long(block):
    return {
        'Short': color_sets['set2'][block],
        'Medium': color_sets['set2_med_dark'][block],
        'Long': color_sets['set2_dark'][block],
    }


def pca_cluster_activity():
    return {
        'Cluster 1 - Low': set2[0],
        'Cluster 1 - High': color_sets['set2_dark'][0],
        'Cluster 2 - Low': set2[1],
        'Cluster 2 - High': color_sets['set2_dark'][1],
        'Cluster 3 - Low': set2[2],
        'Cluster 3 - High': color_sets['set2_dark'][2],
    }


def separate_legend(entries, file_name=None, marker='|'):
    properties = {
        'marker': marker,
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
    separate_legend(blocks(), file_name='block_dots.png', marker='.')
    separate_legend(blocks(), file_name='block_lines.png', marker='_')
    separate_legend(pca_cluster_activity(), file_name='pca_cluster_activity.png', marker='_')
    separate_legend(short_med_long(0), file_name='short_med_long_b1.png', marker='_')
    separate_legend(short_med_long(1), file_name='short_med_long_b2.png', marker='_')
