import json
import os
from glob import glob

import cv2
import mahotas.polygon as mp
import numpy as np
from tqdm import tqdm

# import openslide
# for window
from utils import import_openslide

openslide = import_openslide()


def load_label_info(info, order):
    result = {}
    try:
        if "pathClasses" in info.keys():
            info = info["pathClasses"]
    except AttributeError:
        # already a list?
        pass

    for i, info in enumerate(info):
        result[info['name'].upper()] = [order.index(info['name']) + 1, info['color']]
    return result


def load_annotation_info(annotation_info):
    try:
        if "features" in annotation_info.keys():
            annotation_info = annotation_info["features"]
        elif "geometries" in annotation_info.keys():
            annotation_info = annotation_info["geometries"]
    except AttributeError:
        # already a list?
        pass

    return annotation_info


def make_mask(svs_path, annotation_geojson_path, label_info_path):
    order = ['BN', 'WD', 'MD', 'PD', 'T_W', 'T_M', 'T_P', 'T_LS',
             'papillary', 'Mucinous', 'signet', 'poorly', 'LVI',
             'mucosa', 'mucus', 'submucosa', 'subserosa', 'MM', 'PM', 'Immune cells']

    # load svs
    slide = openslide.OpenSlide(svs_path)
    w_pixels, h_pixels = slide.level_dimensions[0]
    image_masked = np.zeros((w_pixels, h_pixels), dtype=np.int8)

    # load annotation_ info
    annotation_geojson_file = open(annotation_geojson_path)
    annotation_geojson_file = json.load(annotation_geojson_file)
    annotation_info = load_annotation_info(annotation_geojson_file)

    # load label info
    label_info_json_file = open(label_info_path)
    label_info_json_file = json.load(label_info_json_file)
    label_info = load_label_info(label_info_json_file, order)

    for object_idx in range(len(annotation_info)):
        for i, annotation_pts in enumerate(annotation_info[object_idx]['geometry']['coordinates']):
            pts = [(round(px), round(py)) for px, py in annotation_pts]
            if i == 0:
                name = annotation_info[object_idx]['properties']['classification']['name'].upper()
                label = label_info[name][0]
                mp.fill_polygon(pts, image_masked, label)
            else:
                mp.fill_polygon(pts, image_masked, 0)

    return image_masked


if __name__ == '__main__':
    root_dir = './data'
    svs_list = glob(f"{root_dir}/GC_cancer_slides/*.svs")

    for svs_name in tqdm(svs_list):
        svs_name = svs_name.split(os.sep)[-1]
        svs_name = svs_name[:-4]

        svs_path = f'{root_dir}/GC_cancer_slides/{svs_name}.svs'
        annotation_geojson_path = f'{root_dir}/GC_cance_geojson/{svs_name}.geojson'
        label_info_path = f'{root_dir}/Project/classifiers/classes.json'

        image_masked = make_mask(svs_path, annotation_geojson_path, label_info_path)

        patch_save_dir = f'{root_dir}/GC_cancer_mask'
        os.makedirs(patch_save_dir, exist_ok=True)
        cv2.imwrite(root_dir + f'/GC_cancer_mask/{svs_name}.png', image_masked)
