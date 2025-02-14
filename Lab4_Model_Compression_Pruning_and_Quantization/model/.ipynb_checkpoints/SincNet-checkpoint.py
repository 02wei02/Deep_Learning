import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import SincConv1d


class LogAbs(nn.Module):
    def __init__(self):
        super(LogAbs, self).__init__()

    def forward(self, input):
        return torch.log10(torch.abs(input) + 1)


class _Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, max_pool_size, use_sinc):
        super(_Layer, self).__init__()

        self.use_sinc = use_sinc

        if use_sinc:
            self.conv0 = SincConv1d(
                out_channels=out_channels, kernel_size=kernel_size)
            self.logabs = LogAbs()
        else:
            # layer.append(nn.Conv1d(in_channels=in_channels,
            #                        out_channels=out_channels, kernel_size=kernel_size))
            self.conv0 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, stride=stride, groups=in_channels)
            self.conv1 = nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            self.relu = nn.ReLU()

        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.AvgPool1d(max_pool_size)

    def forward(self, input):
        if self.use_sinc:
            out = self.conv0(input)
            out = self.logabs(out)
        else:
            out = self.conv0(input)
            out = self.conv1(out)
            out = self.relu(out)

        out = self.bn(out)
        out = self.pool(out)

        return out


class SincNet(nn.Module):
    def __init__(self):
        super(SincNet, self).__init__()

        # self.ln0 = nn.LayerNorm(16000)
        self.sincconv = _Layer(
            in_channels=1, out_channels=40, kernel_size=101, stride=1, max_pool_size=2, use_sinc=True)

        self.features = nn.ModuleList()     

        self.features.append(_Layer(in_channels=40, out_channels=256,
                                    kernel_size=25, stride=2, max_pool_size=2, use_sinc=False))

        self.features.append(_Layer(in_channels=256, out_channels=256,
                                    kernel_size=9, stride=1, max_pool_size=2, use_sinc=False))

        self.features.append(_Layer(in_channels=256, out_channels=256,
                                    kernel_size=15, stride=1, max_pool_size=2, use_sinc=False))

        self.features.append(_Layer(in_channels=256, out_channels=256,
                                    kernel_size=9, stride=1, max_pool_size=2, use_sinc=False))

        self.features.append(_Layer(in_channels=256, out_channels=160,
                                    kernel_size=9, stride=1, max_pool_size=2, use_sinc=False))

        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(160, 10)

    def forward(self, input):
        out = self.sincconv(input)

        for l in self.features:
            out = l(out)

        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

