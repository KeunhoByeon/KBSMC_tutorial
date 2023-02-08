import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataloader import ClassificationDataset, prepare_KBSMCDataset
from logger import Logger
from models import Classifier
from utils import import_openslide, load_color_info

# import openslide # for window
openslide = import_openslide()


# SVS 파일에서 thumbnail 이미지 생성
def get_thumbnail(svs_path, thumbnail_size):
    slide = openslide.OpenSlide(svs_path)
    w_pixels, h_pixels = slide.level_dimensions[0]

    ratio = min(thumbnail_size / w_pixels, thumbnail_size / h_pixels)
    thumbnail_shape = (int(w_pixels * ratio), int(h_pixels * ratio))

    thumbnail = slide.get_thumbnail(thumbnail_shape)
    thumbnail = np.array(thumbnail)
    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)

    return thumbnail, (w_pixels, h_pixels), ratio


# 모델 출력 결과와 svs파일로 디버깅 이미지 생성
def make_debug_image(outputs, svs_path, colors):
    # Load Thumbnail
    thumbnail, thumbnail_num_pixels, thumbnail_ratio = get_thumbnail(svs_path, 1024)

    target_mask = np.zeros_like(thumbnail)
    pred_mask = np.zeros_like(thumbnail)

    # Make Debugging Image
    for patch_path, pred in outputs:
        patch_filename = os.path.basename(patch_path)
        patch_info = patch_filename.strip('.png').split('_patch_')[1]
        coord_x = int(patch_info.split('_')[0].strip('x'))
        coord_y = int(patch_info.split('_')[1].strip('y'))
        target = int(patch_info.split('_')[2])

        # Calculate coord on thumbnail
        coord_x_1 = coord_x * thumbnail_ratio
        coord_y_1 = coord_y * thumbnail_ratio
        coord_x_2 = coord_x_1 + args.patch_size * thumbnail_ratio
        coord_y_2 = coord_y_1 + args.patch_size * thumbnail_ratio
        coord_x_1, coord_y_1 = int(coord_x_1), int(coord_y_1)
        coord_x_2, coord_y_2 = int(coord_x_2), int(coord_y_2)

        # Apply target abd pred on mask
        target_mask[coord_y_1:coord_y_2, coord_x_1:coord_x_2, 2] = colors[target][0]
        target_mask[coord_y_1:coord_y_2, coord_x_1:coord_x_2, 1] = colors[target][1]
        target_mask[coord_y_1:coord_y_2, coord_x_1:coord_x_2, 0] = colors[target][2]

        pred_mask[coord_y_1:coord_y_2, coord_x_1:coord_x_2, 2] = colors[pred][0]
        pred_mask[coord_y_1:coord_y_2, coord_x_1:coord_x_2, 1] = colors[pred][1]
        pred_mask[coord_y_1:coord_y_2, coord_x_1:coord_x_2, 0] = colors[pred][2]

    # Image Overlay
    pred_image = cv2.addWeighted(thumbnail, 0.4, pred_mask, 0.6, 0)
    target_image = cv2.addWeighted(thumbnail, 0.4, target_mask, 0.6, 0)

    # Stack Images
    debug_image = np.vstack([thumbnail, target_image, pred_image]).astype(np.uint8)

    return debug_image


def evaluate(model, eval_loader, svs_index, logger=None):
    model.eval()

    outputs = []

    confusion_mat = [[0 for _ in range(args.num_classes)] for _ in range(args.num_classes)]
    with torch.no_grad():  # Disable gradient calculation
        for i, (input_paths, inputs, targets) in tqdm(enumerate(eval_loader), leave=False, desc=svs_index, total=len(eval_loader)):
            # CUDA
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            output = model(inputs)

            # Calculate Accuracy
            preds = torch.argmax(output, dim=1)
            acc = torch.sum(preds == targets).item() / len(inputs) * 100.

            for img_path, pred in zip(input_paths, preds):
                outputs.append((img_path, pred.item()))

            # Save history
            if logger is not None:
                logger.add_history('total', {'accuracy': acc})
            for t, p in zip(targets, preds):
                confusion_mat[int(t.item())][p.item()] += 1

    if logger is not None:
        logger(svs_index, history_key='total', time=time.strftime('%Y.%m.%d.%H:%M:%S'))
    if args.print_confusion_mat:
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        print(pd.DataFrame(confusion_mat))

    return outputs


def run(args):
    # Model
    model = Classifier(args.model, num_classes=args.num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint))

    # CUDA
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    # Dataset
    _, eval_set, _ = prepare_KBSMCDataset(args.patch_data, no_testset=True)

    # Get .svs Paths
    svs_paths = {}
    for path, dir, files in os.walk(args.svs_data):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext.lower() != '.svs':
                continue
            svs_index = filename.strip(ext)
            svs_paths[svs_index] = os.path.join(path, filename)

    # Load Debugging Colors
    colors = load_color_info(args.json_path)

    # Logger
    logger = Logger(os.path.join(args.result, 'log.txt'), float_round=5)
    logger.set_sort(['accuracy', 'time'])
    logger(str(args))

    for svs_index in eval_set:
        svs_patch_dir = os.path.join(args.patch_data, svs_index)
        eval_dataset = ClassificationDataset(svs_patch_dir, input_size=args.input_size, return_path=True)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

        # Evaluate
        outputs = evaluate(model, eval_loader, svs_index, logger=logger)

        # Make and Save Debugging Image
        debug_image = make_debug_image(outputs, svs_paths[svs_index], colors=colors)
        save_path = os.path.join(args.result, "{}.png".format(svs_index))
        cv2.imwrite(save_path, debug_image)


if __name__ == '__main__':
    # Arguments 설정
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Model Arguments
    parser.add_argument('--model', default='efficientnet_b0')  # [변경] 사용할 모델 이름
    parser.add_argument('--num_classes', default=18, type=int, help='number of classes')  # [변경] 데이터의 클래스 종류의 수
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint, not necessary')
    parser.add_argument('--checkpoint_name', default='20230118191754', type=str)
    parser.add_argument('--checkpoint_epoch', default=100, type=int)
    # Data Arguments
    parser.add_argument('--patch_data', default='./Data/Qupath2/patch', help='path to patch data')  # [변경] 이미지 패치 저장 경로
    parser.add_argument('--svs_data', default='./Data/Qupath2/data', help='path to svs data')  # [변경] svs파일 저장 경로
    parser.add_argument('--json_path', default='./Data/Qupath2/project/classifiers/classes.json', help='path to json file')  # [변경] json파일 저장 경로
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--input_size', default=512, type=int, help='image input size')  # [변경] 입력 이미지의 크기
    parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size')  # [변경] 배치 사이즈
    # Validation and Debugging Arguments
    parser.add_argument('--print_confusion_mat', default=False, action='store_true')
    parser.add_argument('--patch_size', default=1024, type=int, help='num pixels of patch')
    parser.add_argument('--result', default=None, help='path to results, not necessary')
    parser.add_argument('--result_tag', default='eval')
    args = parser.parse_args()

    # Paths setting
    if args.checkpoint is None or len(args.checkpoint) == 0:
        if args.checkpoint_name is not None and args.checkpoint_epoch is not None:
            args.checkpoint = './results_classifier/{}/checkpoints/{}.pth'.format(args.checkpoint_name, args.checkpoint_epoch)
        if args.checkpoint is None or not os.path.isfile(args.checkpoint):
            print('Cannot find checkpoint file!: {} {} {}'.format(args.checkpoint, args.checkpoint_name, args.checkpoint_epoch))
            raise AssertionError

    if args.result is None:
        if args.checkpoint_name is not None and args.checkpoint_epoch is not None:
            args.result = './results_classifier/{}/{}/{}'.format(args.checkpoint_name, args.result_tag, args.checkpoint_epoch)
        else:
            print('Please specify result dir: {} {} {} {}'.format(args.result, args.checkpoint_name, args.result_tag, args.checkpoint_epoch))
            raise AssertionError
    args.result = os.path.expanduser(args.result)
    os.makedirs(args.result, exist_ok=True)

    run(args)
