import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from tiatoolbox.tools.stainnorm import VahadaneNormalizer
from torch.utils.data import Dataset

from data_utils import prepare_gastric_EBV_data_json
from utils import resize_and_pad_image


class EBVGCDataset(Dataset):
    def __init__(self, samples, input_size: int = None, is_train=False, stain_norm_path=None):
        self.input_size = input_size
        self.is_train = is_train

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Augmentation setting (Not yet implemented all)
        self.affine_seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Sometimes(0.5, iaa.Affine(rotate=(-45, 45)))
        ], random_order=True)
        self.color_seq = iaa.Sequential([
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
            iaa.Sometimes(0.2, iaa.Cutout(nb_iterations=(1, 5), size=0.2, cval=255, squared=False)),
            iaa.Sometimes(0.2, iaa.Grayscale(alpha=(0.0, 1.0))),
            iaa.Sometimes(0.5, iaa.MultiplyHue((0.6, 1.4))),
            iaa.Sometimes(0.5, iaa.MultiplySaturation((0.6, 1.4))),
            iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.2)),
            iaa.Sometimes(0.5, iaa.LogContrast((0.6, 1.4))),
        ], random_order=True)

        self.samples = samples

        if stain_norm_path is not None:
            target = cv2.imread(stain_norm_path)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            self.normalizer = VahadaneNormalizer()
            self.normalizer.fit(target)

    def __getitem__(self, index):
        img_path, gt = self.samples[index]

        if self.is_train and gt == 0 and np.random.random() < 0.001:
            img_path = './data/empty.png'
            img = np.random.normal(255, np.random.randint(0, 127), (self.input_size, self.input_size, 3)) / 2
            img = img.astype(float)
            img[:, :, 0] *= np.random.random()
            img[:, :, 1] *= np.random.random()
            img[:, :, 2] *= np.random.random()
            img = img.astype(np.uint8)
        else:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.input_size is not None:
            img = resize_and_pad_image(img, target_size=(self.input_size, self.input_size), keep_ratio=True, padding=True)

        if self.is_train:
            img = self.affine_seq.augment_image(img)
            img = self.color_seq.augment_image(img)
        elif hasattr(self, "normalizer"):
            img = self.normalizer.transform(img)

        img = img.astype(np.float32) / 255.
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        gt = torch.from_numpy(np.array(gt)).type(torch.LongTensor)  # 0 benign, 1 EBV-GC, 2 Non-EBV-GC

        return img_path, img, gt

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    data_dir = '/media/kwaklab_103/sda/data/patch_data/KBSMC/gastric/gastric_EBV_1024'
    input_size = 512
    seed = 103

    import random

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    train_set, val_set, test_set = prepare_gastric_EBV_data_json(data_dir)
    dataset = EBVGCDataset(val_set, input_size=input_size, do_aug=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    for i, (img_paths, inputs, targets) in enumerate(dataloader):
        for img_path, input, target in zip(img_paths, inputs, targets):
            original_img = cv2.imread(img_path)
            original_img = resize_and_pad_image(original_img, target_size=(input_size, input_size), keep_ratio=True, padding=True)
            input = input.numpy()
            input = input.transpose(1, 2, 0)
            input = input * std + mean
            input = input * 255.
            input = input.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('T', np.hstack([original_img, input]))
            cv2.waitKey(0)
