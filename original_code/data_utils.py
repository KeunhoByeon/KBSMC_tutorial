import glob
import json
import os.path
from collections import Counter

import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from PIL import Image, ImageEnhance
from PIL import ImageStat
import math
# from RandAugment import rand_augment_transform


####
def print_number_of_sample(train_set, valid_set, test_set):
    train_label = [train_set[i][1] for i in range(len(train_set))]
    print("train", Counter(train_label))
    valid_label = [valid_set[i][1] for i in range(len(valid_set))]
    print("valid", Counter(valid_label))
    test_label = [test_set[i][1] for i in range(len(test_set))]
    print("test", Counter(test_label))
    return 0


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class DatasetSerial(data.Dataset):

    def __init__(self, pair_list, transform=None, train=False):
        self.train = train
        self.pair_list = pair_list
        self.transform = transform

    def __getitem__(self, idx):
        path, target = self.pair_list[idx]
        input_img = pil_loader(path)

        if self.transform:
            input_img = self.transform(input_img)

        return input_img, target

    def __len__(self):
        return len(self.pair_list)


def prepare_gastric_cancer_data(data, type, fold_list, data_root):
    core_list = []
    file_list = []
    for fold in fold_list:
        core_list.extend(data[fold][type])

    for WSI_name in core_list:
        pathname = glob.glob(f'{data_root}/{WSI_name}/*.jpg')
        file_list.extend(pathname)
    label_list = [int(file_path.split('_')[-1].split('.')[0]) - 1 for file_path in file_list]

    list_out = list(zip(file_list, label_list))
    return list_out


def prepare_gastric_cancer_data_wsi(data, type, fold_list, data_root):
    core_list = []
    file_list = []
    for fold in fold_list:
        core_list.extend(data[fold][type])

    for WSI_name in core_list:
        pathname = glob.glob(f'{data_root}/{WSI_name}/*/*.jpg')
        file_list.extend(pathname)
    label_list = [int(file_path.split('_')[-1].split('.')[0]) - 1 for file_path in file_list]

    list_out = list(zip(file_list, label_list))
    return list_out


def prepare_gastric_EBV_data_json(img_dir, fold=0, split_type='V0', colornorm=False):
    if split_type == 'wsi':
        json_dir = './data/EBV_split_wsi.json'
    elif split_type == 'patient':
        json_dir = './data/EBV_split_patient.json'
    else:
        json_dir = './data/EBV_split.json'

    with open(json_dir) as json_file:
        data = json.load(json_file)

    if fold == 0:
        train_list = ['fold1', 'fold2', 'fold3']
        val_list = ['fold4', ]
        test_list = ['fold5', ]
    elif fold == 1:
        train_list = ['fold1', 'fold2', 'fold5']
        val_list = ['fold3', ]
        test_list = ['fold4', ]
    elif fold == 2:
        train_list = ['fold1', 'fold4', 'fold5']
        val_list = ['fold2', ]
        test_list = ['fold3', ]
    elif fold == 3:
        train_list = ['fold3', 'fold4', 'fold5']
        val_list = ['fold1', ]
        test_list = ['fold2', ]
    else:
        train_list = ['fold2', 'fold3', 'fold4']
        val_list = ['fold5', ]
        test_list = ['fold1', ]

    # if colornorm:
    #     data_root_dir_wsi = '/media/kwaklab_103/sda/data/patch_data/KBSMC/gastric/gastric_EBV_1024/Gastric_WSI/gastric_wsi_1024_08_resize05_MacenkoNormed/'
    #     data_root_dir = '/media/kwaklab_103/sda/data/patch_data/KBSMC/gastric/gastric_EBV_1024/Gastric_TMAs/Gastric_EBV_V2_1024/tma_image_resize05_MacenkoNormed/'
    # else:
    #     data_root_dir_wsi = '/media/kwaklab_103/sda/data/patch_data/KBSMC/gastric/gastric_EBV_1024/Gastric_WSI/gastric_wsi_1024_08_resize05/'
    #     data_root_dir = '/media/kwaklab_103/sda/data/patch_data/KBSMC/gastric/gastric_EBV_1024/Gastric_TMAs/Gastric_EBV_V2_1024/core_image_resize05/'

    data_root_dir_wsi = os.path.join(img_dir, 'Gastric_WSI/gastric_wsi_1024_08')
    data_root_dir = os.path.join(img_dir, 'Gastric_TMAs/Gastric_EBV_V2_1024/core_image/')

    if split_type == 'wsi':
        train_set = prepare_gastric_cancer_data_wsi(data, 'tma', train_list, data_root_dir) + prepare_gastric_cancer_data(data, 'wsi', train_list, data_root_dir_wsi)
        val_set = prepare_gastric_cancer_data_wsi(data, 'tma', val_list, data_root_dir) + prepare_gastric_cancer_data(data, 'wsi', val_list, data_root_dir_wsi)
        test_set = prepare_gastric_cancer_data_wsi(data, 'tma', test_list, data_root_dir) + prepare_gastric_cancer_data(data, 'wsi', test_list, data_root_dir_wsi)
    else:
        train_set = prepare_gastric_cancer_data(data, 'tma', train_list, data_root_dir) + prepare_gastric_cancer_data(data, 'wsi', train_list, data_root_dir_wsi)
        val_set = prepare_gastric_cancer_data(data, 'tma', val_list, data_root_dir) + prepare_gastric_cancer_data(data, 'wsi', val_list, data_root_dir_wsi)
        test_set = prepare_gastric_cancer_data(data, 'tma', test_list, data_root_dir) + prepare_gastric_cancer_data(data, 'wsi', test_list, data_root_dir_wsi)

    train_label = [train_set[i][1] for i in range(len(train_set))]
    print("train", Counter(train_label))
    valid_label = [val_set[i][1] for i in range(len(val_set))]
    print("valid", Counter(valid_label))
    test_label = [test_set[i][1] for i in range(len(test_set))]
    print("test", Counter(test_label))

    print(len(train_label) + len(valid_label) + len(test_label),
          len(train_label), len(valid_label), len(test_label))  # 137128
    return train_set, val_set, test_set


def get_randaug(img_size=448):
    # transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(
        translate_const=100,
        img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
    )
    if img_size == 384:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                   ra_params),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
            transforms.CenterCrop(448),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize])
    elif img_size == 512:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                   ra_params),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                   ra_params),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize])

    return train_transform, val_transform


def visualize(ds, batch_size, nr_steps=10):
    data_idx = 0
    cmap = plt.get_cmap('jet')
    for i in range(0, nr_steps):
        if data_idx >= len(ds):
            data_idx = 0
        for j in range(1, batch_size + 1):
            sample = ds[data_idx + j]
            if len(sample) == 2:
                img = sample[0]
            else:
                img = sample[0]
                # TODO: case with multiple channels
                aux = np.squeeze(sample[-1])
                aux = cmap(aux)[..., :3]  # gray to RGB heatmap
                aux = (aux * 255).astype('unint8')
                img = np.concatenate([img, aux], axis=0)
                img = cv2.resize(img, (40, 80), interpolation=cv2.INTER_CUBIC)
            plt.subplot(1, batch_size, j)
            plt.title(str(sample[1]))
            plt.imshow(img)
            # plt.imshow(np.array(img).transpose(1,2,0))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        plt.show()
        data_idx += batch_size
