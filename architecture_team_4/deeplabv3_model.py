"""
deeplabv3_model.py

this file defines the DeepLabV3 model used in the project.

what this file does:

- wraps torchvision DeepLabV3
- supports resnet50 / resnet101 backbones
- supports 3-class segmentation:
    0 background
    1 nucleus
    2 chromocenter
- replaces the ASPP pooling branch BatchNorm with GroupNorm
  so training can work safely even with batch_size = 1

notes:

- the shared dataset returns single-channel images [1,H,W]
- DeepLabV3 expects 3-channel input
- this conversion should be handled outside this file
  by repeating the channel in train/eval/infer scripts
"""

import torch
import torch.nn as nn

try:
    from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
    from torchvision.models.segmentation.deeplabv3 import DeepLabHead
except Exception as e:
    raise ImportError(
        "Please install/upgrade torchvision. DeepLabV3 requires torchvision.models.segmentation.\n"
        f"Original error: {e}"
    )


class DeepLabV3(nn.Module):
    """
    DeepLabV3 for multi-class segmentation.

    Input:
        [B, 3, H, W]

    Output:
        logits [B, C, H, W]

    For this project:
        C = 3
        0 background
        1 nucleus
        2 chromocenter
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained_backbone: bool = True,
        out_channels: int = 3,
    ):
        super().__init__()

        if backbone not in ["resnet50", "resnet101"]:
            raise ValueError("backbone must be 'resnet50' or 'resnet101'")

        # build model
        if backbone == "resnet50":
            model = deeplabv3_resnet50(weights=None, weights_backbone=None)
        else:
            model = deeplabv3_resnet101(weights=None, weights_backbone=None)

        # optional pretrained backbone
        if pretrained_backbone:
            try:
                if backbone == "resnet50":
                    model = deeplabv3_resnet50(weights=None, weights_backbone="DEFAULT")
                else:
                    model = deeplabv3_resnet101(weights=None, weights_backbone="DEFAULT")
            except Exception:
                # fallback if current torchvision does not support this form
                pass

        # replace classifier head for project class count
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
            model.classifier = DeepLabHead(2048, out_channels)
        else:
            raise RuntimeError("Unexpected torchvision DeepLabV3 structure: missing classifier.")

        # important:
        # patch ASPP pooling branch BN -> GN for batch_size=1 support
        self._patch_aspp_pooling_norm(model)

        self.net = model
        self.backbone_name = backbone
        self.out_channels = out_channels

    def _patch_aspp_pooling_norm(self, model: nn.Module):
        """
        Replace BatchNorm in ASPP image pooling branch with GroupNorm.

        Why:
            ASPP pooling branch produces [B, C, 1, 1].
            BatchNorm can fail when batch_size=1.
            GroupNorm is stable for this case.
        """
        try:
            # torchvision DeepLabHead:
            # classifier[0] is ASPP
            aspp = model.classifier[0]

            # ASPP.convs is usually a ModuleList
            # final branch is image pooling branch
            pooling_branch = aspp.convs[-1]

            replaced = False
            for i, m in enumerate(pooling_branch):
                if isinstance(m, nn.BatchNorm2d):
                    num_channels = m.num_features

                    groups = min(32, num_channels)
                    while num_channels % groups != 0 and groups > 1:
                        groups -= 1

                    pooling_branch[i] = nn.GroupNorm(groups, num_channels)
                    replaced = True
                    break

            if replaced:
                print("[INFO] DeepLabV3: replaced ASPP pooling BatchNorm with GroupNorm for batch_size=1 support.")
            else:
                print("[WARNING] DeepLabV3: no ASPP pooling BatchNorm found to replace.")

        except Exception as e:
            print(f"[WARNING] DeepLabV3: failed to patch ASPP pooling branch: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            logits [B, C, H, W]
        """
        out = self.net(x)["out"]
        return out


def build_deeplabv3(
    backbone: str = "resnet50",
    pretrained_backbone: bool = True,
    out_channels: int = 3,
) -> nn.Module:
    """
    Factory function for project scripts.
    """
    return DeepLabV3(
        backbone=backbone,
        pretrained_backbone=pretrained_backbone,
        out_channels=out_channels,
    )


if __name__ == "__main__":
    # quick sanity check
    model = build_deeplabv3(
        backbone="resnet50",
        pretrained_backbone=False,
        out_channels=3,
    )

    x = torch.randn(1, 3, 256, 256)
    y = model(x)

    print("input shape:", tuple(x.shape))
    print("output shape:", tuple(y.shape))