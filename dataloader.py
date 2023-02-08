import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import resize_and_pad_image


def prepare_KBSMCDataset(data_dir, no_testset=False):
    anno_indices = []
    for anno_index in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, anno_index)):
            continue
        anno_indices.append(anno_index)
    anno_indices.sort()

    if len(anno_indices) >= 10:
        num_test_set = len(anno_indices) // 10
        num_val_set = len(anno_indices) // 10
    elif len(anno_indices) >= 3:
        num_test_set = 1
        num_val_set = 1
    elif len(anno_indices) >= 2:
        num_test_set = 0
        num_val_set = 1
    else:
        num_test_set = 0
        num_val_set = 0

    if no_testset:
        num_test_set = 0

    num_train_set = len(anno_indices) - num_val_set - num_test_set

    train_set = anno_indices[0:num_train_set]
    val_set = anno_indices[num_train_set:num_train_set + num_val_set] if num_val_set > 0 else []
    test_set = anno_indices[num_train_set + num_val_set: num_train_set + num_val_set + num_test_set] if num_test_set > 0 else []

    print(len(train_set), len(val_set), len(test_set))

    return train_set, val_set, test_set


class ClassificationDataset(Dataset):
    def __init__(self, data_dir, svs_indices=None, input_size: int = None, return_path: bool = False):
        self.input_size = input_size
        self.return_path = return_path

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self.samples = []
        gt_cnt = {}
        for path, dir, files in os.walk(data_dir):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue

                file_index = filename.strip(ext)
                svs_index = file_index.split('_patch_')[0]
                gt = int(file_index.split('_')[-1])
                if svs_indices is not None and svs_index not in svs_indices:  # if svs is not available
                    continue

                if gt not in gt_cnt:
                    gt_cnt[gt] = 0
                gt_cnt[gt] += 1

                self.samples.append((os.path.join(path, filename), gt))

        print("Loaded {} samples ({})".format(len(self.samples), gt_cnt))

    def __getitem__(self, index):
        img_path, gt = self.samples[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.input_size is not None:
            img = resize_and_pad_image(img, target_size=(self.input_size, self.input_size), keep_ratio=True, padding=True)

        img = img.astype(np.float32) / 255.
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        if self.return_path:
            return img_path, img, gt
        return img, gt

    def __len__(self):
        return len(self.samples)


class SegmentationDataset(Dataset):
    def __init__(self, data_dir, mask_dir, svs_indices=None, input_size: int = None, return_path: bool = False):
        self.input_size = input_size
        self.return_path = return_path

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        img_paths = {}
        for path, dir, files in os.walk(data_dir):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue
                img_paths[filename.strip(ext)] = os.path.join(path, filename)

        mask_paths = {}
        for path, dir, files in os.walk(mask_dir):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue

                mask_paths[filename.strip(ext)] = os.path.join(path, filename)

        self.samples = []
        for file_index, img_path in img_paths.items():
            if file_index not in mask_paths.keys():  # if mask is not exist
                continue

            svs_index = file_index.split('_patch_')[0]
            if svs_indices is not None and svs_index not in svs_indices:  # if svs is not available
                continue

            mask_path = mask_paths[file_index]
            self.samples.append((img_path, mask_path))

    def __getitem__(self, index):
        img_path, mask_path = self.samples[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.input_size is not None:
            img = resize_and_pad_image(img, target_size=(self.input_size, self.input_size), keep_ratio=True, padding=True)
            mask = resize_and_pad_image(mask, target_size=(self.input_size, self.input_size), keep_ratio=True, padding=True, interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        mask = torch.from_numpy(mask).long()

        if self.return_path:
            return img_path, img, mask
        return img, mask

    def __len__(self):
        return len(self.samples)
