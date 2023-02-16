from PIL import Image
import numpy as np
import os
import backend
import matplotlib.pyplot as plt


def load_pictures():
    path = os.path.join(os.getcwd(), 'pics')
    paths, file_names = backend.get_file_paths(path)
    pics = [Image.open(img_path) for img_path in paths]
    return pics, file_names


if __name__ == '__main__':
    pictures, names = load_pictures()
    for picture in pictures:
        plt.imshow(picture)
        plt.show()
