from image_editing import load_pictures
from pystackreg import StackReg
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import os
import backend

"""
Align the pictures so they are a bit easier to use in step 3
https://github.com/glichtner/pystackreg
"""


def align_pics(regenerate=False, reverse=True):
    pic_dict = {}
    pics, file_names = load_pictures("pics")
    for pic, name in zip(pics, file_names):
        for delimiter in ['_', '.']:
            name = " ".join(name.split(delimiter))
        name = name.split()
        if (name[-1] == 'jpg' or name[-1] == 'jpeg') and len(name) == 4:
            if name[0] in pic_dict.keys():
                pic_dict[name[0]].append(np.asarray(pic))
            else:
                pic_dict[name[0]] = [np.asarray(pic)]
    for name, mouse_pics in pic_dict.items():
        save_path = os.path.join(os.getcwd(), 'reg_pics', name)
        if os.path.exists(save_path) and not regenerate:
            continue
        print(f'starting {name}...   ', end='')
        img_stack = mouse_pics[::-1] if reverse else mouse_pics
        probe_stack = np.stack([p[:, :, 0] for p in img_stack])
        null_stack = np.stack([p[:, :, 1] for p in img_stack])
        blue_stack = np.stack([p[:, :, 2] for p in img_stack])
        # plt.imshow(probe_stack[24,:,::-1])
        # plt.show()
        probe_com = backend.center_of_mass(np.max(probe_stack, axis=1), axis=1)
        slice_com = backend.center_of_mass(np.max(blue_stack, axis=1), axis=1)
        dif = probe_com - slice_com
        for i, diff in enumerate(dif):
            if diff * np.mean(dif) < 0:
                probe_stack[i] = probe_stack[i, :, ::-1]
                blue_stack[i] = blue_stack[i, :, ::-1]
                null_stack[i] = null_stack[i, :, ::-1]

        # for i in range(len(blue_stack)):
        #     blue_stack[i][np.where(blue_stack[i] >= 235)] = 0
        #     fade = 100
        #     for j, k in zip(*np.where(blue_stack[i] == 0)):
        #         x_lim = [max(0, j - fade), min(blue_stack[i].shape[0], j + fade)]
        #         y_lim = [max(0, k - fade), min(blue_stack[i].shape[1], k + fade)]
        #         near_values = blue_stack[i][x_lim[0]:x_lim[1], y_lim[0]:y_lim[1]].flatten()
        #         blue_stack[i][j, k] = int(np.round(np.mean(near_values[near_values > 0])))
        #     # plt.imshow(blue_stack[i])
        #     # plt.show()
        #     # time.sleep(.5)
        # print()

        sr = StackReg(StackReg.TRANSLATION)
        tmats = sr.register_stack(blue_stack, reference='previous')

        blue_reg = sr.transform_stack(blue_stack)
        probe_reg = sr.transform_stack(probe_stack)
        null_reg = sr.transform_stack(null_stack)
        if plot:
            for i in range(len(blue_reg)):
                plt.imshow(blue_reg[i])
                plt.title(f'{name}_{i}')
                plt.show()
                time.sleep(.5)
        blue_reg = blue_reg[::-1]
        probe_reg = probe_reg[::-1]
        null_reg = null_reg[::-1]
        for i in range(len(blue_reg)):
            arr = np.transpose(np.stack([probe_reg[i], null_reg[i], blue_reg[i]]), (1, 2, 0))
            im = Image.fromarray(arr.astype(np.uint8), 'RGB')
            save_name = f'{name}_reg_s0{i}.jpg' if i >= 10 else f'{name}_reg_s00{i}.jpg'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            im.save(os.path.join(save_path, save_name))
        print('done')


plot = False
if __name__ == '__main__':
    align_pics(regenerate=False, reverse=True)
