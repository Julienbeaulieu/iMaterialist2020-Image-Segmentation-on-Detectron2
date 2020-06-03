import numpy as np
from PIL import Image
# Rle helper functions
from matplotlib.pyplot import imsave
from environs import Env
from pathlib import Path
from tqdm import tqdm
import os
import glob
import math
env = Env()
env.read_env()

path_data_interim = Path(env("path_interim"))
path_test_data = Path(env("path_test"))
path_output = Path(env("path_output"))

def downscale_folder(path_test_data="/home/dyt811/Git/cvnnig/data_imaterialist2020/raw/test"):
    """
    Take the entire test data set, try to downscale the long edge to 1024.
    :param path_test_data:
    :return:
    """
    list_files = glob.glob(f"{path_test_data}/*.jpg")
    for file in tqdm(list_files):
        downscale_image(file, path_data_interim)

def downscale_image(img, path_data_interim = path_data_interim, max_size=1024):
    '''
    Adaptive funciton to first DOWNSCALE the image before running RLE
    # Source: https://stackoverflow.com/a/28453021
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    #img is a tensor
    img_data = Image.open(img)
    img_array = np.array(img_data)
    pil_image = Image.fromarray(img_array)
    width_current = pil_image.size[0]
    height_current = pil_image.size[1]

    longest_edge = max(width_current, height_current)

    # Longest edge MUST be 1024, even if smaller or larger images
    if (width_current > height_current):
        new_width = max_size
        scaled_height = max_size / float(width_current) * height_current
        new_height = int(math.floor(scaled_height))
    else:
        scale_width = max_size / float(height_current) * width_current
        new_width = int(math.floor(scale_width))
        new_height = max_size
    # Always resizing.
    pil_image = pil_image.resize((new_width, new_height), Image.NEAREST)

    image_array = np.array(pil_image)

    imsave(f"{path_data_interim}/resized_test/{Path(img).stem}.jpg", image_array)

if __name__ =="__main__":
    downscale_folder()