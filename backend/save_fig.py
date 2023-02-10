import os


def save_fig(figure, filename, sub_folder=None):
    if sub_folder:
        path = os.path.join(os.getcwd(), 'figures', sub_folder)
    else:
        path = os.path.join(os.getcwd(), 'figures')
    if not os.path.exists(path):
        os.makedirs(path)
    figure.savefig(os.path.join(path, filename))
