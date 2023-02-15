import segmentation_models_pytorch as smp
import torch.nn as nn


class Segmentor(nn.Module):
    def __init__(self, encoder_name, encoder_weights, in_channels, classes):
        super(Segmentor, self).__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=classes,  # model output channels (number of classes in your dataset)
        )

    def forward(self, x):
        return self.model(x)
