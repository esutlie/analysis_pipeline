import seaborn as sns


def get_color_sets():
    color_sets = {
        'set2': sns.color_palette('Set2'),
        'set2_med_dark': [list(c * .8 for c in sublist) for sublist in sns.color_palette('Set2')],
        'set2_dark': [list(c * .5 for c in sublist) for sublist in sns.color_palette('Set2')],
        'colorblind': sns.color_palette('colorblind'),
        'grays': sns.color_palette('Greys'),
        'set1': sns.color_palette('Set1'),
        'set1_med_dark': [list(c * .8 for c in sublist) for sublist in sns.color_palette('Set1')],
        'set1_dark': [list(c * .5 for c in sublist) for sublist in sns.color_palette('Set1')],
    }
    return color_sets
