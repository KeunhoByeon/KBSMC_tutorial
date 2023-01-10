import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import resize_and_pad_image


class EBVGCDataset(Dataset):
    def __init__(self, data_dir, input_size: int = None):
        self.input_size = input_size

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self.samples = []
        for path, dir, files in os.walk(data_dir):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue
                img_path = os.path.join(path, filename)
                gt = int(filename.strip(ext).split('_class_')[-1])
                self.samples.append((img_path, gt))

    def __getitem__(self, index):
        img_path, gt = self.samples[index]  # 0 Benign, 1 EBV-GC, 2 Non-EBV-GC

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.input_size is not None:
            img = resize_and_pad_image(img, target_size=(self.input_size, self.input_size), keep_ratio=True, padding=True)

        img = img.astype(np.float32) / 255.
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        return img, gt

    def __len__(self):
        return len(self.samples)
