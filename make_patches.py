import os
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm

from make_mask import make_mask
# import openslide
# for window
from utils import import_openslide

openslide = import_openslide()

svs_dir = './data/GC_cancer_slides'
patch_save_dir = './data/GC_cancer_patch'
mask_save_dir = './data/GC_cancer_patch_mask'

read_size = 1024
step = 1.0
mask_threshold = 0.8


def is_background(slide_img):
    np_img = np.array(slide_img)
    if np.mean(np_img[:, :, 1]) > 240:  # White area
        return True
    elif np.sum(np_img == 0) / (np_img.shape[0] * np_img.shape[1]) > 0.2:  # Padding area
        return True
    return False


if __name__ == '__main__':
    # 1. Get SVS Paths
    svs_paths = {}
    for path, dir, files in os.walk(svs_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext.lower() != '.svs':
                continue
            file_index = filename.strip(ext)
            svs_path = os.path.join(path, filename)
            svs_paths[file_index] = svs_path

    # 3. Make Patches
    for svs_idx, (file_index, svs_path) in enumerate(svs_paths.items()):
        slide = openslide.OpenSlide(svs_path)
        w_pixels, h_pixels = slide.level_dimensions[0]

        # Open Mask
        # mask_path = mask_paths[file_index]
        # tissue_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        svs_name = svs_path.split(os.sep)[-1]
        svs_name = svs_name[:-4]
        annotation_geojson_path = f'./data/GC_cance_geojson/{svs_name}.geojson'
        label_info_path = f'./data/Project/classifiers/classes.json'
        tissue_mask = make_mask(svs_path, annotation_geojson_path, label_info_path)

        os.makedirs(os.path.join(patch_save_dir, file_index), exist_ok=True)
        os.makedirs(os.path.join(mask_save_dir, file_index), exist_ok=True)
        for w_i in tqdm(range(0, w_pixels, int(read_size * step)), desc="Processing {}/{}".format(svs_idx + 1, len(svs_paths))):
            for h_i in range(0, h_pixels, int(read_size * step)):
                slide_img = slide.read_region((w_i, h_i), 0, (read_size, read_size))
                if is_background(slide_img):  # Check if slide image is bg
                    continue

                slide_mask = tissue_mask[w_i:w_i + read_size, h_i:h_i + read_size]
                cnt = Counter(list(slide_mask.reshape(-1)))
                most_label = cnt.most_common(1)[0][0]

                slide_save_path = os.path.join(patch_save_dir, file_index, '{}_patch_{}_{}_{}.png'.format(file_index, w_i, h_i, most_label))
                mask_save_path = os.path.join(mask_save_dir, file_index, '{}_patch_{}_{}_{}.png'.format(file_index, w_i, h_i, most_label))

                slide_img.save(slide_save_path)
                cv2.imwrite(mask_save_path, slide_mask)
