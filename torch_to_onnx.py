import argparse
import os

import torch

from models import Classifier, Segmentor


def run(args):
    # Model
    if args.model_type == 'classifier':
        model = Classifier(args.model, num_classes=args.num_classes, pretrained=False)
    elif args.model_type == 'segmentor':
        model = Segmentor(
            encoder_name=args.model,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=args.num_classes,  # model output channels (number of classes in your dataset)
        )
        model.load_state_dict(torch.load(args.checkpoint, map_location=None if torch.cuda.is_available() else 'cpu'))
    else:
        print("Model Type {} not yet implemented!".format(args.model_type))
        raise AssertionError
    model.eval()

    # 모델에 대한 입력값
    dummy_input = torch.randn(1, 3, args.input_size, args.input_size, requires_grad=True)
    sample_output = model(dummy_input)

    # 모델 변환
    torch.onnx.export(model,  # 실행될 모델
                      dummy_input,  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                      args.result,  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                      export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                      opset_version=10,  # 모델을 변환할 때 사용할 ONNX 버전
                      do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                      input_names=['input'],  # 모델의 입력값을 가리키는 이름
                      output_names=['output'],  # 모델의 출력값을 가리키는 이름
                      dynamic_axes={'input': {0: 'batch_size'},  # 가변적인 길이를 가진 차원
                                    'output': {0: 'batch_size'}})


if __name__ == '__main__':
    # Arguments 설정
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--model_type', default='classifier', type=str, choices=['classifier', 'segmentor'], help="classifier, segmentor")  # [변경] 사용할 모델 종류
    parser.add_argument('--model', default='efficientnet_b0', type=str)  # [변경] 사용할 모델 이름  (Segmentor 모델에서는 encoder model로 사용)
    parser.add_argument('--num_classes', default=18, type=int, help='number of classes')  # [변경] 데이터의 클래스 종류의 수
    parser.add_argument('--input_size', default=512, type=int, help='image input size')  # [변경] 입력 이미지의 크기
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint, not necessary')
    parser.add_argument('--checkpoint_name', default='202302160001', type=str)
    parser.add_argument('--checkpoint_epoch', default=100, type=int)
    parser.add_argument('--result', default=None, help='path to results, not necessary')
    args = parser.parse_args()

    # Paths setting
    args.model_type = args.model_type.lower()

    if args.checkpoint is None or len(args.checkpoint) == 0:
        if args.checkpoint_name is not None and args.checkpoint_epoch is not None:
            args.checkpoint = './results_{}/{}/checkpoints/{}.pth'.format(args.model_type, args.checkpoint_name, args.checkpoint_epoch)
        if args.checkpoint is None or not os.path.isfile(args.checkpoint):
            print('Cannot find checkpoint file!: {} {} {}'.format(args.checkpoint, args.checkpoint_name, args.checkpoint_epoch))
            raise AssertionError

    if args.result is None:
        if args.checkpoint is not None:
            args.result = args.checkpoint.replace('.pth', '.onnx')
        else:
            raise AssertionError
    run(args)
