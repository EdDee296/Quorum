"""
deeplabv3plus_model.py

this file defines the DeepLabV3+ model used in the project.

what this file does:

- implements DeepLabV3+ with a ResNet backbone
- supports resnet50 / resnet101 backbones
- supports 3-class segmentation:
    0 background
    1 nucleus
    2 chromocenter
- uses GroupNorm in ASPP image pooling branch
  so training can work safely with batch_size = 1

notes:

- the shared dataset returns single-channel images [1,H,W]
- DeepLabV3+ expects 3-channel input
- this conversion should be handled outside this file
  by repeating the channel in train/eval/infer scripts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvGNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, groups=32):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            bias=False,
        )

        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1

        self.gn = nn.GroupNorm(g, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    """

    def __init__(self, in_ch, out_ch=256, atrous_rates=(6, 12, 18)):
        super().__init__()

        self.branch1 = ConvBNReLU(in_ch, out_ch, k=1, s=1, p=0)
        self.branch2 = ConvBNReLU(in_ch, out_ch, k=3, s=1, p=atrous_rates[0], d=atrous_rates[0])
        self.branch3 = ConvBNReLU(in_ch, out_ch, k=3, s=1, p=atrous_rates[1], d=atrous_rates[1])
        self.branch4 = ConvBNReLU(in_ch, out_ch, k=3, s=1, p=atrous_rates[2], d=atrous_rates[2])

        # important:
        # use GroupNorm here because global pooling gives [B,C,1,1]
        # BatchNorm can fail when batch_size=1
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvGNReLU(in_ch, out_ch, k=1, s=1, p=0),
        )

        self.project = nn.Sequential(
            ConvBNReLU(out_ch * 5, out_ch, k=1, s=1, p=0),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        size = x.shape[-2:]

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode="bilinear", align_corners=False)

        x = torch.cat([b1, b2, b3, b4, gp], dim=1)
        return self.project(x)


class ResNetBackbone(nn.Module):
    """
    Returns:
      low-level feature from layer1
      high-level feature from layer4
    """

    def __init__(self, name="resnet50", pretrained=True, output_stride=16):
        super().__init__()

        if output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
        elif output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
        else:
            raise ValueError("output_stride must be 8 or 16")

        if name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(
                weights=weights,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )
            self.low_ch = 256
            self.high_ch = 2048

        elif name == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            backbone = models.resnet101(
                weights=weights,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )
            self.low_ch = 256
            self.high_ch = 2048
        else:
            raise ValueError("backbone must be resnet50 or resnet101")

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        high = self.layer4(x)

        return low, high


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ for semantic segmentation.
    """

    def __init__(
        self,
        backbone="resnet50",
        pretrained_backbone=True,
        num_classes=3,
        output_stride=16,
    ):
        super().__init__()

        self.backbone = ResNetBackbone(
            name=backbone,
            pretrained=pretrained_backbone,
            output_stride=output_stride,
        )

        if output_stride == 16:
            rates = (6, 12, 18)
        else:
            rates = (12, 24, 36)

        self.aspp = ASPP(self.backbone.high_ch, out_ch=256, atrous_rates=rates)

        # low-level feature projection
        self.low_proj = ConvBNReLU(self.backbone.low_ch, 48, k=1, s=1, p=0)

        # decoder
        self.decoder = nn.Sequential(
            ConvBNReLU(256 + 48, 256, k=3, s=1, p=1),
            ConvBNReLU(256, 256, k=3, s=1, p=1),
        )

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        in_size = x.shape[-2:]

        low, high = self.backbone(x)

        x = self.aspp(high)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)

        low = self.low_proj(low)
        x = torch.cat([x, low], dim=1)

        x = self.decoder(x)
        x = self.classifier(x)

        x = F.interpolate(x, size=in_size, mode="bilinear", align_corners=False)
        return x


def build_deeplabv3plus(
    backbone="resnet50",
    pretrained_backbone=True,
    out_channels=3,
    output_stride=16,
):
    return DeepLabV3Plus(
        backbone=backbone,
        pretrained_backbone=pretrained_backbone,
        num_classes=out_channels,
        output_stride=output_stride,
    )


if __name__ == "__main__":
    model = build_deeplabv3plus(
        backbone="resnet50",
        pretrained_backbone=False,
        out_channels=3,
        output_stride=16,
    )

    x = torch.randn(1, 3, 256, 256)
    y = model(x)

    print("input shape:", tuple(x.shape))
    print("output shape:", tuple(y.shape))