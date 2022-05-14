import math
import torch
import dws_cpp

try:
    import dws_gpu_cpp
except ImportError:
    pass

class DwsConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, filter_dw, filter_pw, stride):
        if input.is_cuda:
            return dws_gpu_cpp.dws_conv(input, filter_dw, filter_pw, stride)
        return dws_cpp.dws_conv(input, filter_dw, filter_pw, stride)

class DwsConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad = 0):
        super(DwsConv, self).__init__()
        self.stride = stride
        self.weight_dw = torch.nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        self.weight_pw = torch.nn.Parameter(torch.empty(in_channels, out_channels, 1, 1))
        dws_cpp.init_conv(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        if torch.cuda.is_available():
            dws_gpu_cpp.init_conv(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        self.pad = pad

    def forward(self, input):
        if self.pad:
            input = torch.nn.functional.pad(input, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0.0)
        return DwsConvFunction.apply(input, self.weight_dw, self.weight_pw, self.stride)