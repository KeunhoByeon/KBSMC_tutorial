import random

import cv2
import numpy as np
import torch

from dataloader import EBVGCDataset

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def rollback_image(tensor_image):
    input_image = tensor_image.numpy()
    input_image = input_image.transpose(1, 2, 0)
    input_image = input_image * std + mean
    input_image = input_image * 255.
    input_image = input_image.astype(np.uint8)
    img = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    return img


if __name__ == '__main__':
    data_dir = './data/patch_data'
    input_size = 512
    seed = 103

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    dataset = EBVGCDataset(data_dir, input_size=input_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    print("Dataset Length: {}".format(len(dataloader)))

    for i, (input_images, targets) in enumerate(dataloader):
        for input_image, target in zip(input_images, targets):
            image = rollback_image(input_image)
            print(target)
            cv2.imshow('T', image)
            cv2.waitKey(0)