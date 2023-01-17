import os

import cv2
import numpy as np
import openslide

svs_dir = '../data/GC_cancer_slides'
patch_dir = '../data/GC_cancer_patch'
mask_dir = '../data/GC_cancer_patch_mask'


def get_thumbnail(svs_path, return_info=False, thumbnail_size=1024):
    slide = openslide.OpenSlide(svs_path)
    w_pixels, h_pixels = slide.level_dimensions[0]

    ratio = min(thumbnail_size / w_pixels, thumbnail_size / h_pixels)
    thumbnail_shape = (int(w_pixels * ratio), int(h_pixels * ratio))

    thumbnail = slide.get_thumbnail(thumbnail_shape)
    thumbnail = np.array(thumbnail)
    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)

    if return_info:
        return thumbnail, (w_pixels, h_pixels), ratio
    else:
        return thumbnail


if __name__ == "__main__":
    svs_paths = {}
    for path, dir, files in os.walk(svs_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext.lower() != '.svs':
                continue
            svs_index = filename.strip(ext)
            svs_path = os.path.join(path, filename)
            svs_paths[svs_index] = svs_path

    patch_paths = {}
    for path, dir, files in os.walk(patch_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext.lower() != '.png':
                continue
            svs_index = filename.strip(ext).split('_patch_')[0]
            if svs_index not in patch_paths.keys():
                patch_paths[svs_index] = []
            patch_paths[svs_index].append(os.path.join(path, filename))

    mask_paths = {}
    for path, dir, files in os.walk(patch_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext.lower() != '.png':
                continue
            svs_index = filename.strip(ext).split('_patch_')[0]
            if svs_index not in mask_paths.keys():
                mask_paths[svs_index] = []
            mask_paths[svs_index].append(os.path.join(path, filename))

    for svs_index, svs_path in svs_paths.items():
        # Load Thumbnail
        thumbnail, num_pixels, thumbnail_ratio = get_thumbnail(svs_path, return_info=True)
        w_pixels, h_pixels = num_pixels

        output_mask = np.zeros_like(thumbnail)
        for mask_path in mask_paths[svs_index]:
            label = int(os.path.basename(mask_path).split('.')[0].split('_')[-1])
            coord_x = int(os.path.basename(mask_path).split('.')[0].split('_')[2])
            coord_y = int(os.path.basename(mask_path).split('.')[0].split('_')[3])

            coord_x_1, coord_y_1 = int(coord_x * thumbnail_ratio), int(coord_y * thumbnail_ratio)
            coord_x_2, coord_y_2 = int(coord_x_1 + 1024 * thumbnail_ratio), int(coord_y_1 + 1024 * thumbnail_ratio)

            channel = label % 3
            output_mask[coord_y_1:coord_y_2, coord_x_1:coord_x_2, channel] = 1

        output = (output_mask + 1) / 2 * thumbnail

        output = np.clip(output, 0, 255).astype(np.uint8)
        cv2.imshow('T', output)
        cv2.waitKey()
