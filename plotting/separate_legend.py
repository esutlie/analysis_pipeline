import matplotlib.pyplot as plt
from matplotlib import lines
from matplotlib.patches import Patch
import seaborn as sns
import backend

set2 = sns.color_palette(palette='Set2')
color_sets = backend.get_color_sets()
set1 = sns.color_palette('Set1')


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

def session_summary_example():
    return {
        'Optimal Leave Time': (.8,.8,.8),
        '.4 r/s Block': set2[0],
        '.8 r/s Block': set2[1],
    }
def session_summary_example_markers():
    return {
        'Optimal Leave Time': '_',
        '.4 r/s Block': '|',
        '.8 r/s Block': '|',
    }


def session_summary():
    return {
        'Reward': 'purple',
        'Lick': [.8, .7, .8],
        'Background Port': set1[0],
        'Exponential Port': set1[1],
        '0.4 r/s block': set2[0],
        '0.8 r/s block': set2[1],
    }


def session_summary_markers():
    return {
        'Reward': '|',
        'Lick': '|',
        'Background Port': 'empty_box',
        'Exponential Port': 'empty_box',
        '0.4 r/s block': 'box',
        '0.8 r/s block': 'box',
    }


def dopamine_rewards():
    return {
        'Low Rate Block': set2[0],
        'High Rate Block': set2[1],
        '': [1, 1, 1],
        'First Reward': [.7, .7, .7],
        'Later Rewards': [.7, .7, .7],
    }


def dopamine_rewards_markers():
    return {
        'Low Rate Block': '.',
        'High Rate Block': '.',
        '': '',
        'First Reward': 'x',
        'Later Rewards': '.',
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
        '0: .4 r/s': set2[0],
        '0: .8 r/s': color_sets['set2_dark'][0],
        '1: .4 r/s': set2[1],
        '1: .8 r/s': color_sets['set2_dark'][1],
        '2: .4 r/s': set2[2],
        '2: .8 r/s': color_sets['set2_dark'][2],
    }


def separate_legend(entries, file_name=None, marker=None):
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
        marker = properties['marker']
        marker = marker if isinstance(marker, str) else marker[key]
        if marker == 'box':
            lines_for_legend.append(Patch(facecolor=value, label=key))
        elif marker == 'empty_box':
            lines_for_legend.append(Patch(edgecolor=value, label=key, fill=False))
        else:
            lines_for_legend.append(lines.Line2D([], [], color=value, marker=marker,
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
    # separate_legend(trial_raster(), file_name='legend_trial_raster.png')
    # separate_legend(reward_raster(), file_name='legend_reward_raster.png')
    # separate_legend(blocks(), file_name='block_dots.png', marker='.')
    # separate_legend(blocks(), file_name='block_lines.png', marker='_')
    separate_legend(pca_cluster_activity(), file_name='pca_cluster_activity.png', marker='_')
    # separate_legend(short_med_long(0), file_name='short_med_long_b1.png', marker='_')
    # separate_legend(short_med_long(1), file_name='short_med_long_b2.png', marker='_')
    # separate_legend(session_summary(), file_name='session_summary.png', marker=session_summary_markers())
    # separate_legend(dopamine_rewards(), file_name='dopamine_rewards.png', marker=dopamine_rewards_markers())
    # separate_legend(session_summary_example(), file_name='session_summary_example.png', marker=session_summary_example_markers())
