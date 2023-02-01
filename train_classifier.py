import argparse
import os
import random
import time

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from dataloader import ClassificationDataset, prepare_KBSMCDataset
from logger import Logger
from models import Classifier


def val(epoch, model, criterion, val_loader, logger=None):
    """
    epoch: 학습할 에포크 횟수
    model: 학습할 모델 객체
    criterion: 손실 함수
    val_loader: 평가 데이터를 담은 데이터 로더
    logger: 로그를 저장할 객체
    """

    model.eval()  # 모델을 평가 모드로

    # Confusion matrix 초기화
    confusion_mat = [[0 for _ in range(args.num_classes)] for _ in range(args.num_classes)]

    # 평가 데이터를 가져오기 위한 반복문
    with torch.no_grad():  # Disable gradient calculation
        for i, (inputs, targets) in tqdm(enumerate(val_loader), leave=False, desc='Validation {}'.format(epoch), total=len(val_loader)):
            # CUDA 사용 가능 시 GPU 사용
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            # 모델 출력
            output = model(inputs)
            loss = criterion(output, targets)

            # Accuracy 계산
            preds = torch.argmax(output, dim=1)
            acc = torch.sum(preds == targets).item() / len(inputs) * 100.

            # 로그 히스토리 저장
            logger.add_history('total', {'loss': loss.item(), 'accuracy': acc})

            # Confusion matrix 저장
            for t, p in zip(targets, preds):
                confusion_mat[int(t.item())][p.item()] += 1

    # 히스토리 출력
    if logger is not None:
        logger('*Validation {}'.format(epoch), history_key='total', time=time.strftime('%Y.%m.%d.%H:%M:%S'))
    # Confusion matrix 출력
    if args.print_confusion_mat:
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        print(pd.DataFrame(confusion_mat))


def train(args, epoch, model, criterion, optimizer, train_loader, logger=None):
    """
    args: 학습 설정을 담고 있는 객체
    epoch: 학습할 epoch 횟수
    model: 학습할 모델 객체
    criterion: 손실 함수
    optimizer: 최적화 알고리즘
    train_loader: 학습 데이터를 담은 데이터 로더
    logger: 로그를 저장할 객체
    """

    model.train()  # 모델을 학습 모드로

    # For print progress
    num_progress, next_print = 0, args.print_freq

    # Confusion matrix 초기화
    confusion_mat = [[0 for _ in range(args.num_classes)] for _ in range(args.num_classes)]

    # 학습 데이터를 가져오기 위한 반복문
    for i, (inputs, targets) in enumerate(train_loader):
        # CUDA 사용 가능 시 GPU 사용
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        # 경사 초기화
        optimizer.zero_grad()
        # 모델 출력 값 계산
        output = model(inputs)
        # 손실 계산
        loss = criterion(output, targets)
        # 경사 계산
        loss.backward()
        # 최적화
        optimizer.step()

        # 정확도 계산
        preds = torch.argmax(output, dim=1)
        acc = torch.sum(preds == targets).item() / len(inputs) * 100.

        # 로그 히스토리 저장
        num_progress += len(inputs)
        logger.add_history('total', {'loss': loss.item(), 'accuracy': acc})
        logger.add_history('batch', {'loss': loss.item(), 'accuracy': acc})

        # Confusion matrix 저장
        for t, p in zip(targets, preds):
            confusion_mat[int(t.item())][p.item()] += 1

        # 일정 주기마다 로그 히스토리 출력
        if num_progress >= next_print:
            if logger is not None:
                logger(history_key='batch', epoch=epoch, batch=num_progress, time=time.strftime('%Y.%m.%d.%H:%M:%S'))
            next_print += args.print_freq

    # 전체 로그 히스토리 및 Confusion matrix 출력
    if logger is not None:
        logger(history_key='total', epoch=epoch, lr=round(optimizer.param_groups[0]['lr'], 12))
    if args.print_confusion_mat:
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        print(pd.DataFrame(confusion_mat))


def run(args):
    # Random Seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # [변경] Model 설정
    model = Classifier(args.model, num_classes=args.num_classes, pretrained=args.pretrained)
    if args.resume is not None:  # resume
        model.load_state_dict(torch.load(args.resume))

    # [변경] Criterion (Loss Function, 손실 함수)  설정
    criterion = nn.CrossEntropyLoss()

    # [변경] Optimizer 옵티마이저  설정
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # [변경] 스케줄러 설정
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # CUDA
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    # Dataset
    train_set, val_set, _ = prepare_KBSMCDataset(args.data, no_testset=True)
    train_dataset = ClassificationDataset(args.data, input_size=args.input_size, svs_indices=train_set)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)
    val_dataset = ClassificationDataset(args.data, input_size=args.input_size, svs_indices=val_set)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

    # Logger
    logger = Logger(os.path.join(args.result, 'log.txt'), epochs=args.epochs, dataset_size=len(train_loader.dataset), float_round=5)
    logger.set_sort(['loss', 'accuracy', 'lr', 'time'])
    logger(str(args))

    # Run
    save_dir = os.path.join(args.result, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(args.start_epoch, args.epochs):
        # Train
        train(args, epoch, model, criterion, optimizer, train_loader, logger=logger)

        # Validation
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            val(epoch, model, criterion, val_loader, logger=logger)
            torch.save(model.state_dict(), os.path.join(save_dir, '{}.pth'.format(epoch)))

        # Scheduler Step
        scheduler.step()


if __name__ == '__main__':
    # Arguments 설정
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Model Arguments
    parser.add_argument('--model', default='efficientnet_b0')  # [변경] 사용할 모델 이름
    parser.add_argument('--num_classes', default=18, type=int, help='number of classes')  # [변경] 데이터의 클래스 종류의 수
    parser.add_argument('--pretrained', default=True, action='store_true', help='Load pretrained model.')
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
    # Data Arguments
    parser.add_argument('--data', default='./Data/Qupath2/patch', help='path to dataset')  # [변경] 이미지 패치 저장 경로
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--input_size', default=512, type=int, help='image input size')  # [변경] 입력 이미지의 크기
    # Training Arguments
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')  # [변경]훈련 반복 수
    parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size')  # [변경]배치 사이즈
    parser.add_argument('--lr', default=0.00001, type=float, help='initial learning rate', dest='lr')  # [변경] 초기 Learning rate
    parser.add_argument('--seed', default=103, type=int, help='seed for initializing training.')
    # Validation and Debugging Arguments
    parser.add_argument('--val_freq', default=10, type=int, help='validation frequency')
    parser.add_argument('--print_freq', default=1000, type=int, help='print frequency')
    parser.add_argument('--print_confusion_mat', default=False, action='store_true')  # [변경] Validation이 끝날 때 Confusion Matrix 출력 여부.
    parser.add_argument('--result', default='results', type=str, help='path to results')
    parser.add_argument('--tag', default=None, type=str)
    args = parser.parse_args()

    # Paths setting
    args.data = os.path.expanduser(args.data)
    args.result = os.path.expanduser(args.result)
    args.result = os.path.join(args.result, time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    if args.tag is not None:
        args.result = '{}_{}'.format(args.result, args.tag)
    os.makedirs(args.result, exist_ok=True)

    run(args)
