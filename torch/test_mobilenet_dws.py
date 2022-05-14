import torch
import torch.nn as nn
import torch.nn.functional as F
from time import perf_counter
from dws import DwsConv

class MobileNetV1(nn.Module):
    def __init__(self, ch_in = 3, n_classes = 1000):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            DwsConv(32, 64, 3, 1, 1),
            DwsConv(64, 128, 3, 2, 1),
            DwsConv(128, 128, 3, 1, 1),
            DwsConv(128, 256, 3, 2, 1),
            DwsConv(256, 256, 3, 1, 1),
            DwsConv(256, 512, 3, 2, 1),
            DwsConv(512, 512, 3, 1, 1),
            DwsConv(512, 512, 3, 1, 1),
            DwsConv(512, 512, 3, 1, 1),
            DwsConv(512, 512, 3, 1, 1),
            DwsConv(512, 512, 3, 1, 1),
            DwsConv(512, 1024, 3, 2, 1),
            DwsConv(1024, 1024, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, n_classes)

        # self.dwconv32_64 = DwsConv(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        # self.dwconv64_128 = DwsConv(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        # self.dwconv128_128 = DwsConv(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        # self.dwconv128_256 = DwsConv(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        # self.dwconv256_512 = DwsConv(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        # self.dwconv512_512 = DwsConv(in_channels=64, out_channels=128, kernel_size=3, stride=1)

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
