import pickle
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import backend
import nrrd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# def atlas_colormap():
#     atlas_path = os.path.join(os.getcwd(), 'atlas_files', 'atlas_labels.txt')
#     atlas_cmap = pd.read_csv(atlas_path, sep=' ', names=['q', 'area_code', 'r', 'g', 'b', 'a', 's', 'd', 'area'])
#     return ListedColormap(atlas_cmap[['r', 'g', 'b']].values / 256)


def atlas_colormap():
    atlas_path = os.path.join(os.getcwd(), 'atlas_files', 'atlas_labels.txt')
    atlas_cmap = pd.read_csv(atlas_path, sep=' ', names=['q', 'area_code', 'r', 'g', 'b', 'a', 's', 'd', 'area'])
    # tuples = list(zip(atlas_cmap[['r', 'g', 'b']].values / 256, atlas_cmap['area_code'].values.astype(str)))
    colors = np.row_stack([np.ones([50,3]), atlas_cmap[['r', 'g', 'b']].values / 256])
    return LinearSegmentedColormap.from_list('atlas_cmap', colors)


def atlas_graymap():
    atlas_path = os.path.join(os.getcwd(), 'atlas_files', 'allen_atlas.nrrd')
    voxels, header = nrrd.read(atlas_path)
    voxels = np.max(voxels)
    colors = np.array([np.random.random(np.max(voxels)) / 3 + .5] * 3)
    colors = np.row_stack([np.zeros([3]), colors.T])
    # colors = np.row_stack([np.zeros([3]), atlas_cmap[['r', 'g', 'b']].values / 256])
    return LinearSegmentedColormap.from_list('atlas_cmap', colors)


def make_atlas_pic(mice):
    atlas_cmap = atlas_graymap()
    atlas_path = os.path.join(os.getcwd(), 'atlas_files', 'allen_atlas.nrrd')
    voxels, header = nrrd.read(atlas_path)

    files = backend.get_session_list()
    unit_coords = []
    unit_colors = []
    # colors = np.array([(255, 209, 220), (150, 206, 193), (176, 212, 255), (216, 191, 255)])/255.
    # colors = np.array([(255, 127, 80), (0, 128, 128), (0, 0, 255), (128, 0, 128)])/255.
    colors = np.array([(31,120,180), (51,160,44), (227,26,28), (255,127,0)])/255.

    mouse_list = []
    for session in files:
        if session[:5] not in mice:
            continue
        mouse = session[:5]
        probe_df_path = os.path.join(os.getcwd(), 'probe_info', f'{mouse}.pkl')
        local_path = os.path.join(backend.get_data_path(), session, 'cluster_info.pkl')
        probe_df = pd.read_pickle(probe_df_path)
        cluster_info = pd.read_pickle(local_path)
        if 'coords' in cluster_info.keys():
            coords = np.stack(cluster_info.coords.values)
            coords = coords[:, [1, 2, 0]]
            x = coords[:, 2] / 2.5 + voxels.shape[2] / 2
            y = -1 * coords[:, 1] / 2.5
            z = -1 * coords[:, 0] / 2.5 + voxels.shape[0] / 2
            unit_coords.append(np.array([x, y, z]))
            unit_colors.append(colors[cluster_info.shank.values.astype(int)])
            # plt.imshow(voxels[int(np.round(np.mean(z)))], cmap='gray', vmin=-1000, vmax=2000)
            # plt.scatter(x, y)
            # plt.show()
            mouse_list.append(mouse)
    mouse_list = np.unique(mouse_list)
    unit_coords = np.column_stack(unit_coords)
    unit_colors = np.row_stack(unit_colors).T
    x, y, z = unit_coords[0], unit_coords[1], unit_coords[2]
    plt.figure(figsize=(12, 8))
    [plt.scatter([], [], c=colors[i]) for i in range(len(colors))]
    plt.legend(['Shank 0', 'Shank 1', 'Shank 2', 'Shank 3'])
    plt.imshow(voxels[int(np.round(np.mean(z)))], cmap=atlas_cmap, vmin=-1000, vmax=2000)
    plt.scatter(x, y, alpha=.1, s=25, c=unit_colors.T) #, edgecolor='white', linewidth=.5)
    plt.title(', '.join(mouse_list))
    plt.show()


if __name__ == '__main__':
    # [make_atlas_pic([mouse]) for mouse in ['ES024', 'ES025', 'ES029', 'ES030', 'ES031']]
    # make_atlas_pic(['ES024', 'ES025', 'ES029', 'ES030', 'ES031', 'ES032', 'ES037', 'ES039'])
    make_atlas_pic(['ES044'])
