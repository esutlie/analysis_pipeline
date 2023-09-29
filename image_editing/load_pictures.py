from PIL import Image
import numpy as np
import os
import backend
import matplotlib.pyplot as plt


def load_pictures(folder='pics'):
    path = os.path.join(os.getcwd(), folder)
    paths, file_names = backend.get_file_paths(path)
    pics = [Image.open(img_path) for img_path in paths]
    return pics, file_names


def rename():
    path = os.path.join(os.getcwd(), 'pics')
    paths, file_names = backend.get_file_paths(path)
    for name in file_names:
        old_name = name
        if name[-4:] == 'jpeg':
            os.rename(os.path.join(path, name), os.path.join(path, name[:-4] + 'jpg'))
        if name[-4:] == 'jpeg' or name[-3:] == 'jpg' or name[-3:] == 'tif':
            for delimiter in ['_', '.']:
                name = " ".join(name.split(delimiter))
            name = name.split()
            if name[1] == 'slide':
                name.pop(1)
                if len(name) == 3:
                    name = f'{name[0]}_{name[1]}.{name[2]}'
                    if os.path.exists(os.path.join(path, name)):
                        os.remove(os.path.join(path, name))
                    os.rename(os.path.join(path, old_name), os.path.join(path, name))
            if len(name) == 4 and len(name[2]) == 1:
                name = f'{name[0]}_{name[1]}_0{name[2]}.{name[3]}'
                os.rename(os.path.join(path, old_name), os.path.join(path, name))

                print()


if __name__ == '__main__':
    # rename()
    pictures, names = load_pictures()
    for picture in pictures:
        plt.imshow(picture)
        plt.show()
