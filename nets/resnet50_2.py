import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ctypes   # Load the shared library  

class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(CustomConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x_padded, in_height, in_width):
        return self.custom_conv2d(x_padded, in_height, in_width, self.weight, self.bias, self.stride, self.padding)

    def custom_conv2d(self):
        return nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, bias=self.bias)
    
class Tile(nn.Module):
    def __init__(self, dim: int, reps: int):
        super().__init__()
        self.dim = dim
        self.reps = reps

    def forward(self, x):
        return x.repeat_interleave(self.reps, dim=self.dim)

class Bottleneck2(nn.Module):
    '''
    Contains three types of convolutional layers
    conv1-Number of compression channels
    conv2-Extract features
    conv3-extended number of channels
    This structure can better extract features, deepen the network, and reduce the number of network parameters。
    inplanes - in_channels
    planes = out_channels
    '''

    expansion = 4
    skip_expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_custom_conv=False, planes_per_use_custom_planes=4):
        super(Bottleneck2, self).__init__()

        self.use_custom_conv = use_custom_conv
        
        # use_custom_conv: 암호화 내적을 하는가?
        if use_custom_conv:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)   # 1x1 conv
            self.bn1 = nn.BatchNorm2d(planes)
            # print(f'bottleneck conv1: ({inplanes} -> {planes})')
            
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            # print(f'bottleneck conv2: ({planes} -> {planes})')
            
            self.conv3 = nn.Conv2d(planes, planes * 4 * planes_per_use_custom_planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4 * planes_per_use_custom_planes)
            # print(f'bottleneck conv3: ({planes} -> {planes * 4 * planes_per_use_custom_planes})')
            
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)

            self.bn1 = nn.BatchNorm2d(planes)
            # print(f'bottleneck conv1: ({inplanes} -> {planes})')
            
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            # print(f'bottleneck conv2: ({planes} -> {planes})')
            
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)
            # print(f'bottleneck conv3: ({planes} -> {planes * 4})')
            
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        # print('====')


        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet2(nn.Module):
    def __init__(self, block, layers, num_classes=1000, custom_conv_layer_index=1):
        
        self.inplanes = 64
        super(ResNet2, self).__init__()

        self.custom_conv_layer_index = custom_conv_layer_index
        
        # self.conv1 = CustomConv2D(1, 64, kernel_size=3, stride=1, padding=2, bias=False)            #FIXME: custom conv
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, bias=False,)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0], skip_planes=32, layer_index=1, use_custom_planes=16)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, skip_planes=128, layer_index=2, use_custom_planes=64)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, skip_planes=256, layer_index=3, use_custom_planes=128)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, skip_planes=512, layer_index=4, use_custom_planes=256)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, skip_planes, stride=1, layer_index=1, use_custom_planes=16):
        downsample = None
        use_custom = (layer_index == self.custom_conv_layer_index)

        if stride != 1 or self.inplanes != planes * block.expansion:

            # print(f'use_custom: {use_custom}')

            if use_custom:
                # print(planes * block.expansion / skip_planes, stride, skip_planes, planes * block.expansion)
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, skip_planes, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(skip_planes),
                    Tile(dim=1, reps=int(planes * block.expansion / skip_planes)),
                    nn.BatchNorm2d(planes * block.expansion),
                )
                # print(f'skip_connections: ({self.inplanes} -> {skip_planes})')
                # print(f'skip_connections: ({skip_planes} -> {planes * block.expansion})')
                
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion)
                )

                # print(f'skip_connections: ({self.inplanes} -> {planes * block.expansion})')
            
       
        layers = []

        if use_custom:
            # print(f'<use custom> make block - 1 : 입력 채널 {self.inplanes} / 중간 채널 {use_custom_planes}')
            layers.append(block(self.inplanes, use_custom_planes, stride, downsample, use_custom_conv=use_custom, planes_per_use_custom_planes=planes // use_custom_planes))
            self.inplanes = use_custom_planes * block.expansion * (planes // use_custom_planes)
        
        else:
            # print(f'make block - 1 : 입력 채널 {self.inplanes} / 중간 채널 {planes}')
            layers.append(block(self.inplanes, planes, stride, downsample, use_custom_conv=use_custom))
            self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            # print(f'make block - {i + 1} : 입력 채널 {self.inplanes} / 중간 채널 {planes}')
            layers.append(block(self.inplanes, planes))

        # print(f'make block - 1 : 입력 채널 {self.inplanes} / 중간 채널 {planes} / 최종 채널 {planes * block.expansion}')
        
        
        # print("================")

        return nn.Sequential(*layers)

    def forward(self, x):
        padding = 2
        x_padded = F.pad(x, (padding, padding, padding, padding))
        x = self.conv1(x_padded)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x
