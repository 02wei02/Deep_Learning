import torch
import torch.nn as nn
import torch.nn.functional as F


class quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, num_of_bits):
        n = 2 ** (num_of_bits-1)
        input_ = input_ * n

        return input_.round() / n

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class Quantize(nn.Module):
    def __init__(self, num_of_bits=4):
        super(Quantize, self).__init__()
        self.num_of_bits = num_of_bits
        self.quan = quantize.apply

    def forward(self, input):
        return self.quan(input, self.num_of_bits)

    def extra_repr(self):
        s = ('num_of_bits={num_of_bits}')

        return s.format(**self.__dict__)



class QuaternaryConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, num_of_bits=4):
        super(QuaternaryConv1d, self).__init__(in_channels, out_channels,
                                               kernel_size, stride, padding, dilation, groups, bias=bias)
        self.num_of_bits = num_of_bits
        self.quan = quantize.apply

        nn.init.normal_(self.weight.data, mean=0.0, std=0.3)

    def forward(self, input, quantized_weight=True):

        if quantized_weight is True:
            # if not hasattr(self.weight, 'org'):
            #     self.weight.org = self.weight.data.clone()
            self.weight.data = F.hardtanh(self.weight.data)
            self.weight.data = self.quan(self.weight.data, self.num_of_bits-1)

        out = F.conv1d(input, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)

        return out


class QuaternaryLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=False, num_of_bits=8):
        super(QuaternaryLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.num_of_bits = num_of_bits
        self.quan = quantize.apply

    def forward(self, input, quantized_weight=True):

        input = self.quan(input, 4)

        if quantized_weight is True:
            if not hasattr(self.weight, 'org'):
                self.weight.org = self.weight.data.clone()
                if self.bias is not None:
                    self.bias.org = self.bias.data.clone()
            # self.weight.data = self.weight.data * self.weight_scale
            self.weight.data = F.hardtanh(self.weight.data)
            self.weight.data = self.quan(self.weight.data, self.num_of_bits-1)
            if self.bias is not None:
                self.bias.data = F.hardtanh(self.bias.data)
                self.bias.data = self.quan(self.bias.data, self.num_of_bits-1)

        out = F.linear(input, self.weight, self.bias)

        # self.weight.data = self.weight.data * (1/self.weight_scale)

        return out
