import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    batch_size, num_channels, depth, height, width = x.size()
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, depth, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batch_size, -1, depth, height, width)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        if not 1 <= stride <= 3:
            raise ValueError("illegal stride value")

        self.stride = stride
        branch_features = oup // 2
        assert self.stride != 1 or inp == branch_features << 1

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm3d(inp),
                nn.Conv3d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv3d(
                inp if self.stride > 1 else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm3d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm3d(branch_features),
            nn.Conv3d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(inp, out, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv3d(inp, out, kernel_size, stride, padding, bias=bias, groups=inp)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels):
        super().__init__()
        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as a list of 3 positive integers")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as a list of 5 positive integers")

        self._stage_out_channels = stages_out_channels
        input_channels = 1
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        for name, repeats, output_channels in zip(
            ["stage2", "stage3", "stage4"],
            stages_repeats,
            self._stage_out_channels[1:],
        ):
            blocks = [InvertedResidual(input_channels, output_channels, 2)]
            for _ in range(repeats - 1):
                blocks.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*blocks))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x.mean([2, 3, 4])


def shufflenet_v2_x0_5(pretrained=False, progress=True, **kwargs):
    if pretrained:
        raise ValueError("Pretrained weights are not provided for the 3D ShuffleNetV2 encoder.")
    return ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)
