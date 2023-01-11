import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import resize_and_pad_image


class KBSMCDataset(Dataset):
    def __init__(self, data_dir, mask_dir, input_size: int = None):
        self.input_size = input_size

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

                if int(filename.strip(ext).split('_')[-1]) == 0:
                    continue

                mask_paths[filename.strip(ext)] = os.path.join(path, filename)

        self.samples = []
        for file_index, img_path in img_paths.items():
            if file_index not in mask_paths.keys():
                continue
            mask_path = mask_paths[file_index]
            self.samples.append((img_path, mask_path))

    def __getitem__(self, index):
        img_path, mask_path = self.samples[index]  # 0 Benign, 1 EBV-GC, 2 Non-EBV-GC

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

        mask = torch.from_numpy(mask)

        return img, mask

    def __len__(self):
        return len(self.samples)
