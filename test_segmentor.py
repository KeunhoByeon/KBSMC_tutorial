import argparse
import os
import time
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from geojson import Polygon, Feature, FeatureCollection, dump
from tqdm import tqdm

from dataloader import SegmentationDataset, prepare_KBSMCDataset
from logger import Logger
from models import Segmentor
from utils import import_openslide, load_color_info, load_name_info

warnings.filterwarnings("ignore", category=UserWarning)

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
def save_debug_image(args, outputs, svs_path, colors):
    # Make Debugging Image

    for patch_path, pred, target in tqdm(outputs, leave=False, desc="Post Processing"):
        patch = cv2.imread(patch_path)
        patch = cv2.resize(patch, (pred.shape[1], pred.shape[0]))

        pred = np.stack((pred for _ in range(3)), axis=-1)
        temp_pred = pred.copy()
        for label, color in colors.items():
            pred[:, :, 0][temp_pred[:, :, 0] == label] = color[2]
            pred[:, :, 1][temp_pred[:, :, 1] == label] = color[1]
            pred[:, :, 2][temp_pred[:, :, 2] == label] = color[0]

        target = np.stack((target for _ in range(3)), axis=-1)
        temp_target = target.copy()
        for label, color in colors.items():
            target[:, :, 0][temp_target[:, :, 0] == label] = color[2]
            target[:, :, 1][temp_target[:, :, 1] == label] = color[1]
            target[:, :, 2][temp_target[:, :, 2] == label] = color[0]

        pred = cv2.addWeighted(patch, 0.4, pred.astype(np.uint8), 0.6, 0)
        target = cv2.addWeighted(patch, 0.4, target.astype(np.uint8), 0.6, 0)

        debug_image = np.hstack([patch, pred, target])
        save_path = os.path.join(args.result, 'patch', os.path.basename(patch_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, debug_image)


# 모델 출력 결과와 svs파일로 geojson 생성
def save_geojson(args, outputs, svs_path, colors, names, min_area=256):
    features = []

    for i, (patch_path, pred, target) in tqdm(enumerate(outputs), leave=False, desc="Post Processing (Geojson)"):
        patch_filename = os.path.basename(patch_path)
        patch_info = patch_filename.strip('.png').split('_patch_')[1]
        base_x = int(patch_info.split('_')[0].strip('x'))
        base_y = int(patch_info.split('_')[1].strip('y'))
        for label_index, label_name in names.items():
            if label_index == 0:
                continue

            temp_pred_map = np.where(pred == label_index, 255, 0).astype(np.uint8)
            temp_pred_map = cv2.morphologyEx(temp_pred_map, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16)))
            if np.max(temp_pred_map.flatten()) == 0:
                continue

            temp_pred_map = np.where(pred == label_index, 255, 0).astype(np.uint8)
            temp_pred_map = cv2.morphologyEx(temp_pred_map, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            contours, hierarchy = cv2.findContours(temp_pred_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    continue
                contour = np.array(contour)[:, 0, :]
                contour = contour.astype(float) * args.patch_size / args.input_size
                contour[:, 0] += base_x
                contour[:, 1] += base_y
                contour = contour.astype(int).tolist()
                contour.append(contour[0])

                point = Polygon([contour])
                features.append(Feature(id=i, geometry=point, properties={"objectType": "annotation", "classification": {"name": label_name, "color": colors[label_index]}}))


    feature_collection = FeatureCollection(features)
    with open(os.path.join(args.result, os.path.basename(svs_path).replace('.svs', '.geojson')), 'w') as f:
        dump(feature_collection, f)


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
            acc = torch.sum(preds == targets).item() / (targets.shape[0] * targets.shape[1] * targets.shape[2]) * 100.

            for img_path, pred, target in zip(input_paths, preds, targets):
                if torch.cuda.is_available():
                    pred, target = pred.cpu(), target.cpu()
                outputs.append((img_path, pred.numpy(), target.numpy()))

            # Save history
            if logger is not None:
                logger.add_history('total', {'accuracy': acc})

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
    model = Segmentor(
        encoder_name=args.encoder_model,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=args.num_classes,  # model output channels (number of classes in your dataset)
    )

    state_dict = torch.load(args.checkpoint, map_location='cpu')
    try:
        model.load_state_dict(state_dict)
    except:
        for key in list(state_dict.keys()):
            state_dict["model." + key] = state_dict.pop(key)
        model.load_state_dict(state_dict)

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
    names = load_name_info(args.json_path)

    # Logger
    logger = Logger(os.path.join(args.result, 'log.txt'), float_round=5)
    logger.set_sort(['accuracy', 'time'])
    logger(str(args))

    for svs_index in eval_set:
        svs_patch_dir = os.path.join(args.patch_data, svs_index)
        svs_mask_dir = os.path.join(args.mask_dir, svs_index)
        eval_dataset = SegmentationDataset(svs_patch_dir, svs_mask_dir, input_size=args.input_size, return_path=True)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

        # Evaluate
        outputs = evaluate(model, eval_loader, svs_index, logger=logger)

        # Make and Save Debugging Image
        save_geojson(args, outputs, svs_paths[svs_index], colors=colors, names=names)
        save_debug_image(args, outputs, svs_paths[svs_index], colors=colors)


if __name__ == '__main__':
    # Arguments 설정
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Model Arguments
    parser.add_argument('--encoder_model', default='resnet34')  # [변경] 사용할 encoder 모델 이름
    parser.add_argument('--num_classes', default=18, type=int, help='number of classes')  # [변경] 데이터의 클래스 종류의 수
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint, not necessary')
    parser.add_argument('--checkpoint_name', default='202302160001', type=str)
    parser.add_argument('--checkpoint_epoch', default=20, type=int)
    # Data Arguments
    parser.add_argument('--patch_data', default='./Data/Qupath2/patch', help='path to patch data')  # [변경] 이미지 패치 저장 경로
    parser.add_argument('--mask_dir', default='./Data/Qupath2/mask', help='path to mask dir')  # [변경] 패치 마스크 저장 경로
    parser.add_argument('--svs_data', default='./Data/Qupath2/data', help='path to svs data')  # [변경] svs파일 저장 경로
    parser.add_argument('--json_path', default='./Data/Qupath2/project/classifiers/classes.json', help='path to json file')  # [변경] json파일 저장 경로
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--input_size', default=512, type=int, help='image input size')  # [변경] 입력 이미지의 크기
    parser.add_argument('--batch_size', default=32, type=int, help='mini-batch size')  # [변경] 배치 사이즈
    # Validation and Debugging Arguments
    parser.add_argument('--print_confusion_mat', default=False, action='store_true')
    parser.add_argument('--patch_size', default=1024, type=int, help='num pixels of patch')
    parser.add_argument('--result', default=None, help='path to results, not necessary')
    parser.add_argument('--result_tag', default='eval')
    args = parser.parse_args()

    # Paths setting
    if args.checkpoint is None or len(args.checkpoint) == 0:
        if args.checkpoint_name is not None and args.checkpoint_epoch is not None:
            args.checkpoint = './results_segmentor/{}/checkpoints/{}.pth'.format(args.checkpoint_name, args.checkpoint_epoch)
        if args.checkpoint is None or not os.path.isfile(args.checkpoint):
            print('Cannot find checkpoint file!: {} {} {}'.format(args.checkpoint, args.checkpoint_name, args.checkpoint_epoch))
            raise AssertionError

    if args.result is None:
        if args.checkpoint_name is not None and args.checkpoint_epoch is not None:
            args.result = './results_segmentor/{}/{}/{}'.format(args.checkpoint_name, args.result_tag, args.checkpoint_epoch)
        else:
            print('Please specify result dir: {} {} {} {}'.format(args.result, args.checkpoint_name, args.result_tag, args.checkpoint_epoch))
            raise AssertionError
    args.result = os.path.expanduser(args.result)
    os.makedirs(args.result, exist_ok=True)

    run(args)
