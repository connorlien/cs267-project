import numpy as np
import torch
from time import perf_counter
from torch import nn
import dws_cpp

def get_input():
    args = [2.0, 4.0, 4.0, 3.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0, 2.0, 2.0]
    X = torch.rand(128, 3, 512, 512)
    F2d = torch.rand(3, 1, 2, 2)
    F1d = torch.rand(3, 3, 1, 1)
    Y = torch.rand(128, 3, 256, 256)
    return args, X, F2d, F1d, Y

def test_torch():
    args, X, F2d, F1d, Y = get_input()
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

test_torch()