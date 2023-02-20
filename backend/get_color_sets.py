import seaborn as sns


def get_color_sets():
    color_sets = {
        'set2': sns.color_palette('Set2'),
        'set2_dark': [[c * .5 for c in sublist] for sublist in sns.color_palette('Set2')],
        'set2_med_dark': [[c * .8 for c in sublist] for sublist in sns.color_palette('Set2')],
        'colorblind': sns.color_palette('colorblind'),
    }
    return color_sets
