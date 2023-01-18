import numpy as np
#import openslide # for window
import os

from collections import Counter
from make_mask import make_mask
from tqdm import tqdm
from utils import import_openslide
openslide = import_openslide()

def is_background(slide_img):
    np_img = np.array(slide_img)
    if np.mean(np_img[:, :, 1]) > 240:  # White area
        return True
    elif np.sum(np_img == 0) / (np_img.shape[0] * np_img.shape[1]) > 0.2:  # Padding area
        return True
    return False

if __name__ == '__main__':
    #0. Set Parameters
    project_name = 'Qupath2'
    patch_size = 1024
    step = 1.0
    mask_threshold = 0.8
    mask_ratio = 0.3

    svs_dir = f'./Data/{project_name}/data'
    patch_save_dir = f'./Data/{project_name}/patch'

    # 1. Get SVS Paths
    svs_paths = {}
    for path, dir, files in os.walk(svs_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1].lower()
            if ext != '.svs':
                continue
            file_index = filename.strip(ext)
            svs_path = os.path.join(path, filename)
            svs_paths[file_index] = svs_path

    # 2. Make Patches
    for svs_idx, (file_index, svs_path) in enumerate(svs_paths.items()):
        slide = openslide.OpenSlide(svs_path)
        w_pixels, h_pixels = slide.level_dimensions[0]

        os.makedirs(os.path.join(patch_save_dir, file_index), exist_ok=True)
        for w_i in tqdm(range(0, w_pixels, int(patch_size * step)), desc="Processing {}/{}".format(svs_idx + 1, len(svs_paths))):
            for h_i in range(0, h_pixels, int(patch_size * step)):
                slide_img = slide.read_region((w_i, h_i), 0, (patch_size, patch_size))
                if is_background(slide_img):  # Check if slide image is bg
                    continue
                save_path = os.path.join(patch_save_dir, file_index, '{}_patch_{}_{}.png'.format(file_index, w_i, h_i))
                slide_img.save(save_path)
