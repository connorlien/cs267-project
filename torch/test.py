import numpy as np
import torch
from time import perf_counter
from torch import nn
from dws import DwsConv

def get_input(B, H, W):
    args = [2.0, 4.0, 4.0, 3.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0, 2.0, 2.0]
    X = torch.rand(B, 3, H, W)
    F2d = torch.rand(3, 1, 2, 2)
    F1d = torch.rand(3, 3, 1, 1)
    Y = torch.rand(B, 3, H, W)
    return args, X, F2d, F1d, Y

def test_torch(args, X, F2d, F1d, Y):
    conv = nn.Conv2d(in_channels=int(args[3]), out_channels=int(args[3]), kernel_size=int(args[4]), stride=int(args[10]), groups=int(args[3]), bias=False)
    conv.weight = torch.nn.Parameter(F2d)
    point_conv = nn.Conv2d(in_channels=int(args[3]), out_channels=int(args[9]), kernel_size=1, bias=False)
    point_conv.weight = torch.nn.Parameter(F1d)
    depthwise_separable_conv = torch.nn.Sequential(conv, point_conv)
    time_before = perf_counter()
    out = depthwise_separable_conv(X)
    time_after = perf_counter()
    print("PyTorch completed in %f seconds with %d threads" \
        % (time_after - time_before, torch.get_num_threads()))
    return out

def test_custom(args, X, F2d, F1d, Y):
    conv = DwsConv(in_channels=int(args[3]), out_channels=int(args[3]), kernel_size=int(args[4]), stride=int(args[10]))
    conv.weight_dw = torch.nn.Parameter(F2d)
    conv.weight_pw = torch.nn.Parameter(F1d)
    time_before = perf_counter()
    out = conv(X)
    time_after = perf_counter()
    print("Custom completed in %f seconds with %d threads" \
        % (time_after - time_before, torch.get_num_threads()))
    return out

# Compute convolutions
i = get_input(128, 512, 512)
out1 = test_torch(*i)
out2 = test_custom(*i)

# Test correctness
eps = 1e-5
assert (out1 - out2).abs().max().item() <= eps, "FAILED CORRECTNESS"
print("Passed Correctness!")
