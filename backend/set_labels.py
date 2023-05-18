import matplotlib.pyplot as plt


def set_labels(ax, xlabel=None, ylabel=None, xlim=None, ylim=None, legend=None, xticks=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if legend:
        ax.legend(legend)
    if xticks:
        plt.sca(ax)
        locs, labels = plt.xticks()
        plt.xticks(locs, xticks, rotation=45)
    if title:
        ax.set_title(title)
