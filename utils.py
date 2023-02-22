import json
import math
import os

import cv2
import pandas as pd


def load_color_info(classifier_json_path):
    classifier_json = open(classifier_json_path)
    info = json.load(classifier_json)

    result = {}
    try:
        if "pathClasses" in info.keys():
            info = info["pathClasses"]
    except AttributeError:
        # already a list?
        pass

    result[0] = (0, 0, 0)
    for i, info in enumerate(info):
        color = 16777216 + info['color']
        R = math.floor(color / (256 * 256))
        G = math.floor(color / 256) % 256
        B = color % 256
        if R == G == B == 0:  # TEMP
            G = 255
        elif R == G == B == 255:
            G = 0
        result[i + 1] = (R, G, B)

    return dict(sorted(result.items()))  # RGB


def load_name_info(classifier_json_path):
    classifier_json = open(classifier_json_path)
    info = json.load(classifier_json)

    result = {}
    try:
        if "pathClasses" in info.keys():
            info = info["pathClasses"]
    except AttributeError:
        pass

    result[0] = "Background"
    for i, info in enumerate(info):
        result[i + 1] = info['name']

    return dict(sorted(result.items()))  # RGB


def import_openslide():
    # The path can also be read from a config file, etc.
    OPENSLIDE_PATH = f'{os.getcwd()}' + r'/openslide-win64/bin'

    if hasattr(os, 'add_dll_directory'):
        # Python >= 3.8 on Windows
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
    else:
        import openslide
    return openslide


def resize_and_pad_image(img, target_size=(512, 512), keep_ratio=False, padding=False, interpolation=None):
    # 1) Calculate ratio
    old_size = img.shape[:2]
    if keep_ratio:
        ratio = min(float(target_size[0]) / old_size[0], float(target_size[1]) / old_size[1])
        new_size = tuple([int(x * ratio) for x in old_size])
    else:
        new_size = target_size

    # 2) Resize image
    if interpolation is None:
        interpolation = cv2.INTER_AREA if new_size[0] < old_size[0] else cv2.INTER_CUBIC
    img = cv2.resize(img.copy(), (new_size[1], new_size[0]), interpolation=interpolation)

    # 3) Pad image
    if padding:
        delta_w = target_size[1] - new_size[1]
        delta_h = target_size[0] - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        if (isinstance(padding, list) or isinstance(padding, tuple)) and len(padding) == 3:
            value = padding
        else:
            value = [0, 0, 0]
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)

    return img


def load_annotation(xlsx_path):
    pd_exel = pd.read_excel(xlsx_path)
    return dict(zip(pd_exel['patient'], pd_exel['EBV.positive']))
