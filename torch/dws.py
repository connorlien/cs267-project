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
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DwsConv, self).__init__()
        self.stride = stride
        self.weight_dw = torch.nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        self.weight_pw = torch.nn.Parameter(torch.empty(in_channels, out_channels, 1, 1))
        dws_cpp.init_conv(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        if torch.gpu.is_available():
            dws_gpu_cpp.init_conv(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    def forward(self, input):
        return DwsConvFunction.apply(input, self.weight_dw, self.weight_pw, self.stride)