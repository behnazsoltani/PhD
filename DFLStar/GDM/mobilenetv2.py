import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupNormBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100, num_groups=2):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.GroupNorm(num_groups, in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.GroupNorm(num_groups, in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.GroupNorm(num_groups, out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual

class MobileNetV2(nn.Module):
    def __init__(self, class_num=100, num_groups=2):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.GroupNorm(num_groups, 32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = GroupNormBottleNeck(32, 16, 1, 1, num_groups)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6, num_groups)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6, num_groups)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6, num_groups)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6, num_groups)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6, num_groups)
        self.stage7 = GroupNormBottleNeck(160, 320, 1, 6, num_groups)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.GroupNorm(num_groups, 1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        extracted_features = x
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x, extracted_features

    def _make_stage(self, repeat, in_channels, out_channels, stride, t, num_groups):
        layers = []
        layers.append(GroupNormBottleNeck(in_channels, out_channels, stride, t, num_groups))
        while repeat - 1:
            layers.append(GroupNormBottleNeck(out_channels, out_channels, 1, t, num_groups))
            repeat -= 1
        return nn.Sequential(*layers)

def group_norm_mobilenetv2(num_groups=2):
    return GroupNormMobileNetV2(num_groups=num_groups)
