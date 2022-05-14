import torch
import torch.nn as nn
import torch.nn.functional as F
from time import perf_counter

class MobileNetV1(nn.Module):
    def __init__(self, ch_in = 3, n_classes = 1000):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def benchmark(bs = 128, img_size = 224, num_iters = 5):
    model =  MobileNetV1()
    rt = torch.rand((bs, 3, img_size, img_size))
    with torch.no_grad():
        time_before = perf_counter()
        for _ in range(num_iters):
                x = model(rt)
        time_after = perf_counter()
        print((time_after - time_before)/num_iters)

benchmark(bs = 32)
benchmark(bs = 64)
benchmark(bs = 128)
benchmark(bs = 256)
benchmark(bs = 512)