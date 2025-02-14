import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import Quantize, QuaternaryConv1d, QuaternaryLinear
from .module import QuatSincConv1d

class LogAbs(nn.Module):
    def __init__(self):
        super(LogAbs, self).__init__()

    def forward(self, input):
        return torch.log10(torch.abs(input) + 1)
    
class _Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, max_pool_size, use_sinc,num_of_bit=8):
        super(_Layer, self).__init__()

        self.use_sinc = use_sinc

        if use_sinc:
            self.conv = QuatSincConv1d(
                out_channels=out_channels, kernel_size=kernel_size)
            self.logabs = LogAbs()
            
          
            self.relu = nn.ReLU()
            self.quan = Quantize(num_of_bit)
        else:
            self.conv0 = QuaternaryConv1d(in_channels=in_channels, out_channels=in_channels,
                                          kernel_size=kernel_size, stride=stride)
            
            self.conv1 = QuaternaryConv1d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=kernel_size, stride=stride)
                
            self.bn = nn.BatchNorm1d(out_channels)
            
            
            self.quan = Quantize(num_of_bit//2)
         
            self.relu = nn.ReLU()
            
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.AvgPool1d(max_pool_size)

    def forward(self, input):
        if self.use_sinc:
            out = self.conv(input)
            out = self.logabs(out)
           
        else:
            out = self.conv0(input)
            out = self.conv1(out)
            out = self.relu(out)
        
        out = self.bn(out)
        out = self.quan(out)
        out = self.pool(out)

        
        return out


class SincNet_Quat(nn.Module):
    def __init__(self, expansion=1.0):
        super(SincNet_Quat, self).__init__()


        self.sincconv = _Layer(
            in_channels=1, out_channels=int(32), kernel_size=101, stride=1, max_pool_size=2, use_sinc=True,num_of_bit=8)

        self.features = nn.ModuleList()

        self.features.append(_Layer(in_channels=32, out_channels=int(32*expansion),
                                    kernel_size=25, stride=2, max_pool_size=2, use_sinc=False,num_of_bit=8))

        self.features.append(_Layer(in_channels=int(32*expansion), out_channels=int(64*expansion),
                                    kernel_size=9, stride=1, max_pool_size=2, use_sinc=False,num_of_bit=8))

        self.features.append(_Layer(in_channels=int(64*expansion), out_channels=int(64*expansion),
                                    kernel_size=9, stride=1, max_pool_size=2, use_sinc=False,num_of_bit=8))

        self.features.append(_Layer(in_channels=int(64*expansion), out_channels=int(64*expansion),
                                    kernel_size=9, stride=1, max_pool_size=2, use_sinc=False,num_of_bit=8))

        self.features.append(_Layer(in_channels=int(64*expansion), out_channels=int(64*expansion),
                                    kernel_size=9, stride=1, max_pool_size=2, use_sinc=False,num_of_bit=8))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.quan_gap = Quantize(num_of_bits=8)
        self.fc = nn.Linear(int(64*expansion), 10, bias=True)

        #self.fc = QuaternaryLinear(int(64*expansion), 10, bias=True,num_of_bits=8)

        self.outs = {}

    def forward(self, input): 
        out = self.sincconv(input)

        for i, l in enumerate(self.features):
            out = l(out)
           
       
        out = self.gap(out)
        out = self.quan_gap(out)
        out = out.view(out.size(0), -1)
        
        out = self.fc(out)

        return out

    