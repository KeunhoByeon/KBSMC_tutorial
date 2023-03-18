import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import json
from glob import glob
import cv2
import mahotas.polygon as mp
import numpy as np
from tqdm import tqdm
# import openslide for window
from utils import import_openslide
openslide = import_openslide()

def get_roi_points(ptr_list, roi_list):
    ''''
    ptr_list = [x, y]
    roi_list = [sx, sy, ex, ey]
    '''
    roi_list = roi_list
    roi_points = []
    for point in ptr_list:
        for roi in roi_list:
            sx, sy, ex, ey = roi
            if sx <= point[0] <= ex and sy <= point[1] <= ey:
                roi_points.append(point)
    return roi_points

def load_label_info(info, order=None):
    result = {}
    try:
        if "pathClasses" in info.keys():
            info = info["pathClasses"]
    except AttributeError:
        # already a list?
        pass

    for i, info in enumerate(info):
        result[info['name'].upper()] = [i+1, info['color']]
    # for i, info in enumerate(info):
    #     result[info['name'].upper()] = [order.index(info['name']) + 1, info['color']]
    return result


def load_annotation_info(annotation_info):
    try:
        if "features" in annotation_info.keys():
            annotation_info = annotation_info["features"]
        elif "geometries" in annotation_info.keys():
            annotation_info = annotation_info["geometries"]
        # find roi
        annotation_list = []
        roi_list = []
        for object_idx in range(len(annotation_info)):
            if annotation_info[object_idx]['properties']['classification']['name'].upper() == 'ROI':
                sx, sy = 1e12, 1e12
                ex, ey = 0, 0
                for roi_pts in annotation_info[object_idx]['geometry']['coordinates'][0]:
                    sx = roi_pts[0] if sx > roi_pts[0] else sx
                    sy = roi_pts[1] if sy > roi_pts[1] else sy
                    ex = roi_pts[0] if ex < roi_pts[0] else ex
                    ey = roi_pts[1] if ey < roi_pts[1] else ey

                roi_list.append([sx, sy, ex, ey])
            else:
                annotation_list.append(annotation_info[object_idx])
    except AttributeError:
        # already a list?
        pass

    return annotation_list, roi_list

def make_mask(svs_path, annotation_geojson_path, label_info_path):
    # load svs
    slide = openslide.OpenSlide(svs_path)
    w_pixels, h_pixels = slide.level_dimensions[0]
    image_masked = np.zeros((h_pixels, w_pixels), dtype=np.int8)

    # load annotation_ info
    annotation_geojson_file = open(annotation_geojson_path)
    annotation_geojson_file = json.load(annotation_geojson_file)
    annotation_info, roi_info = load_annotation_info(annotation_geojson_file)

    # load label info
    label_info_json_file = open(label_info_path)
    label_info_json_file = json.load(label_info_json_file)
    label_info = load_label_info(label_info_json_file)

    for object_idx in range(len(annotation_info)):
        if annotation_info[object_idx]['geometry']['type'].lower() == 'polygon':
            for i, annotation_pts in enumerate(annotation_info[object_idx]['geometry']['coordinates']):

                roi_annotation_pts = get_roi_points(annotation_pts, roi_info)

                pts = [(round(py), round(px)) for px, py in annotation_pts]
                if i == 0:
                    name = annotation_info[object_idx]['properties']['classification']['name'].upper()
                    label = label_info[name][0]
                    mp.fill_polygon(pts, image_masked, label)
                else:
                    mp.fill_polygon(pts, image_masked, 0)
        elif annotation_info[object_idx]['geometry']['type'].lower() == 'multipolygon':
            for annotations in annotation_info[object_idx]['geometry']['coordinates']:
                for i, annotation_pts in enumerate(annotations):
                    pts = [(round(py), round(px)) for px, py in annotation_pts]
                    name = annotation_info[object_idx]['properties']['classification']['name'].upper()
                    label = label_info[name][0]
                    mp.fill_polygon(pts, image_masked, label)
    del slide
    return image_masked, roi_info

if __name__ == '__main__':
    project_name = 'Qupath2'

    svs_list = glob(f"./Data/{project_name}/data/*.svs")
    for svs_name in tqdm(svs_list):
        svs_name = svs_name.split(os.sep)[-1]
        svs_name = svs_name[:-4]

        svs_path = f'./Data/{project_name}/data/{svs_name}.svs'
        annotation_geojson_path = f'./Data/{project_name}/data/{svs_name}.geojson'
        label_info_path = f'./Data/{project_name}/project/classifiers/classes.json'

        patch_save_dir = f'./Data/{project_name}/mask'
        os.makedirs(patch_save_dir, exist_ok=True)

        image_masked = make_mask(svs_path, annotation_geojson_path, label_info_path)
        cv2.imwrite(f'./Data/{project_name}/mask/{svs_name}.png', image_masked)

        image_masked_thumnail = cv2.resize(image_masked.astype('float32'), (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
        del image_masked
        image_masked_thumnail[image_masked_thumnail>0] = 255
        cv2.imwrite(f'./Data/{project_name}/mask/{svs_name}_thumnail.png', image_masked_thumnail)
        del image_masked_thumnail
