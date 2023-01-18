import os
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm

from make_mask import make_mask
from utils import import_openslide

# import openslide # for window
openslide = import_openslide()


def is_background(slide_img):
    np_img = np.array(slide_img)
    if np.mean(np_img[:, :, 1]) > 240:  # White area
        return True
    elif np.sum(np_img == 0) / (np_img.shape[0] * np_img.shape[1]) > 0.2:  # Padding area
        return True
    return False


def get_label(mask, patch_size, ratio):
    cnt = Counter(list(mask.reshape(-1)))
    cnt = cnt.most_common()
    label = cnt[0][0]
    # 마스크 내부에 존재하는 레이블의 종류가 2가지 이상이면 마스크에 가장 많은 레이블이 0(백그라운드)일 경우 조건문 실행
    if len(cnt) > 1 and label == 0:
        number_of_second_label = cnt[1][1]  # 0이 아닌 그 다음으로 제일 개수가 많은 레이블의 수를 카운팅
        second_label_ratio = number_of_second_label / (patch_size * patch_size)  # 0이 아닌 그 다음으로 제일 개수가 많은 레이블이 마스크 영상의 일정 비율(ratio)보다 많다면 레이블을 변경
        if second_label_ratio >= ratio:
            label = cnt[1][0]
    return label


if __name__ == '__main__':
    # 0. Set Parameters
    project_name = 'Qupath2'
    patch_size = 1024
    step = 1.0
    mask_ratio = 0.3

    svs_dir = f'./Data/{project_name}/data'
    patch_save_dir = f'./Data/{project_name}/patch'
    mask_save_dir = f'./Data/{project_name}/mask'

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

        svs_name = svs_path.split(os.sep)[-1]
        svs_name = svs_name[:-4]
        annotation_geojson_path = f'./Data/{project_name}/data/{svs_name}.geojson'
        label_info_path = f'./Data/{project_name}/project/classifiers/classes.json'
        tissue_mask = make_mask(svs_path, annotation_geojson_path, label_info_path)

        os.makedirs(os.path.join(patch_save_dir, file_index), exist_ok=True)
        os.makedirs(os.path.join(mask_save_dir, file_index), exist_ok=True)
        for w_i in tqdm(range(0, w_pixels, int(patch_size * step)), desc="Processing {}/{}".format(svs_idx + 1, len(svs_paths))):
            for h_i in range(0, h_pixels, int(patch_size * step)):
                slide_img = slide.read_region((w_i, h_i), 0, (patch_size, patch_size))
                if is_background(slide_img):  # Check if slide image is bg
                    continue

                slide_mask = tissue_mask[h_i:h_i + patch_size, w_i:w_i + patch_size]
                label = get_label(mask=slide_mask, patch_size=patch_size, ratio=mask_ratio)
                slide_save_path = os.path.join(patch_save_dir, file_index, '{}_patch_x{}_y{}_{}.png'.format(file_index, w_i, h_i, label))
                mask_save_path = os.path.join(mask_save_dir, file_index, '{}_patch_x{}_y{}_{}.png'.format(file_index, w_i, h_i, label))

                slide_img.save(slide_save_path)
                cv2.imwrite(mask_save_path, slide_mask)
