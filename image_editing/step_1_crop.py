from PIL import Image
import numpy as np
import os
from image_editing import load_pictures
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import argrelmin, argrelmax
import math

"""
Use this to crop pictures of multislice slides into single slice pictures all of the same size. 
Assumes the original pic is a .tif. Saves the slices as .jpg for space.
Make a folder named 'pics' in the same directory as this script and move your pictures in to copy what I do.
Images should be named with a 5 character name + _ + pic number, like 'ES007_1.tif'. Or change the code to accommodate 
a different naming system.
"""


def crop_slices(show_pics=False, save_pics=False):
    slice_size = [2000, 1500]
    pictures, names = load_pictures()
    mouse_names = [name[:5] for name in names]
    mice = np.unique(mouse_names)
    mouse_pics = dict(zip(mice, [[]] * len(mice)))
    # mouse_pic_names = dict(zip(mice, [[]] * len(mice)))
    for i, picture in enumerate(pictures):
        count = 0
        slice_pics = []
        name = names[i]
        if (name[-4:] == 'jpeg') or (name[-3:] == 'jpg'):
            continue
        arr = np.array(picture)
        arr = np.flip(arr, axis=1)
        row_mean = np.mean(arr[:, :, 2], axis=1)
        row_mean_filter = savgol_filter(row_mean, 1001, 3)
        if save_pics:
            im = Image.fromarray(arr)
            im.save(os.path.join('pics', f'{name[:-4]}.jpg'))
        # plt.plot(row_mean)
        # plt.plot(row_mean_filter)
        # plt.title(name)
        # plt.show()

        peaks = argrelmax(row_mean_filter, order=1000, mode='clip')[0]
        y_bounds = [[int(val - slice_size[1] / 2), int(val + slice_size[1] / 2)] for val in peaks]
        for y_bound in y_bounds:
            if y_bound[0] < 0:
                y_bound = [b + abs(y_bound[0]) for b in y_bound]
            if y_bound[1] > len(arr):
                y_bound = [b - (y_bound[1] - len(arr)) for b in y_bound]

            half_pic = arr[y_bound[0]:y_bound[1], :, :]
            col_mean = np.mean(half_pic[:, :, 2], axis=0)
            col_mean_filter = savgol_filter(col_mean, 2001, 3)

            # plt.imshow(half_pic)
            # plt.title(name)
            # plt.show()

            # plt.plot(col_mean)
            # plt.plot(col_mean_filter)
            # plt.title(name)
            # plt.show()

            peaks = argrelmax(col_mean_filter, order=300, mode='clip')[0]
            x_bounds = [[int(val - slice_size[0] / 2), int(val + slice_size[0] / 2)] for val in peaks]
            for x_bound in x_bounds:
                if x_bound[0] < 0:
                    x_bound = [b + abs(x_bound[0]) for b in x_bound]
                if x_bound[1] > len(arr[0]):
                    x_bound = [b - (x_bound[1] - len(arr[0])) for b in x_bound]
                slice_pic = half_pic[:, x_bound[0]:x_bound[1]]
                slice_pics.append(slice_pic)
                mouse_pics[name[:5]] = mouse_pics[name[:5]] + [slice_pic]
                # mouse_pic_names[name[:5]] = mouse_pic_names[name[:5]] + [f'{name}_{count_str}']
                count += 1
                if save_pics:
                    count_str = f'0{count}' if count < 10 else f'{count}'
                    im = Image.fromarray(slice_pic)
                    im.save(os.path.join('pics', f'{name[:-4]}_{count_str}.jpg'))
        if show_pics:
            fig, axes = plt.subplots(2, math.ceil(len(slice_pics) / 2))
            axes = [item for sublist in axes for item in sublist]
            for j, ax in enumerate(axes):
                if len(slice_pics[j]):
                    ax.imshow(slice_pics[j])
                ax.axis('off')
            plt.suptitle(name)
            plt.show()
    return mouse_pics


if __name__ == '__main__':
    crop_slices(show_pics=False, save_pics=True)
