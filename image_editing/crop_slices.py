from PIL import Image
import numpy as np
import os
import backend
from load_pictures import load_pictures


def crop_slices():
    pictures, names = load_pictures()
    for i, picture in enumerate(pictures):
        name = names[i]
        