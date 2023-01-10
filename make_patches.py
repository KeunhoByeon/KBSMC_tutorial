import os

import cv2
import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm

svs_dir = './data/raw_data/TCGA_Stomach_452'
mask_dir = './data/raw_data/TCGA_Stomach_452_tissue_mask'
anno_path = './data/raw_data/STAD_molecular_subtype TCGA data.xlsx'
save_dir = './data/patch_data/TCGA_Stomach_452'

read_size = 1024
step = 1.0
mask_threshold = 0.8

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

    # 2. Get Mask Paths
    mask_paths = {}
    for path, dir, files in os.walk(mask_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            file_index = filename.strip(ext)
            mask_path = os.path.join(path, filename)
            mask_paths[file_index] = mask_path

    # 3. Load Annotation
    pd_exel = pd.read_excel(anno_path)
    anno_dict = dict(zip(pd_exel['patient'], pd_exel['EBV.positive']))

    # 4. Make Patches
    for file_index, svs_path in tqdm(svs_paths.items()):
        slide = openslide.OpenSlide(svs_path)
        w_pixels, h_pixels = slide.level_dimensions[0]

        # Open Mask
        mask_path = mask_paths[file_index]
        tissue_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        tissue_mask = tissue_mask.astype(float) / 255.  # 0.0 ~ 1.0

        # Get EBV Positive or Negative
        patient_index = file_index.split('-')[:3]
        patient_index = '-'.join(patient_index)
        label = anno_dict[patient_index]
        label = 2 if label == 0 else label  # Please ignore this line

        patch_save_dir = os.path.join(save_dir, file_index)
        os.makedirs(patch_save_dir, exist_ok=True)
        for w_i in range(0, w_pixels, int(read_size * step)):
            for h_i in range(0, h_pixels, int(read_size * step)):
                # Mask Processing
                relative_pos_x_1 = w_i / w_pixels
                relative_pos_x_2 = (w_i + read_size) / w_pixels
                relative_pos_y_1 = h_i / h_pixels
                relative_pos_y_2 = (h_i + read_size) / h_pixels

                mask_pos_x_1 = int(relative_pos_x_1 * tissue_mask.shape[1])
                mask_pos_x_2 = int(relative_pos_x_2 * tissue_mask.shape[1])
                mask_pos_y_1 = int(relative_pos_y_1 * tissue_mask.shape[0])
                mask_pos_y_2 = int(relative_pos_y_2 * tissue_mask.shape[0])

                mask_mean_value = np.mean(tissue_mask[mask_pos_y_1:mask_pos_y_2, mask_pos_x_1:mask_pos_x_2])
                if mask_mean_value < mask_threshold:
                    continue

                slide_img = slide.read_region((w_i, h_i), 0, (read_size, read_size))
                save_path = os.path.join(patch_save_dir, '{}_patch_{}_{}_class_{}.png'.format(file_index, w_i, h_i, label))
                slide_img.save(save_path)
