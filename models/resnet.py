from __future__ import absolute_import
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        # cfg should be a number in this case
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def downsample_basic_block(x, planes):
    x = nn.AvgPool2d(2,2)(x)
    zero_pads = torch.Tensor(
        x.size(0), planes - x.size(1), x.size(2), x.size(3)).zero_()
    if isinstance(x.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([x.data, zero_pads], dim=1))

    return out

class ResNet(nn.Module):

    def __init__(self, depth, num_classes=10, cfg=None, num_blocks=None):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        # assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = BasicBlock
        if cfg == None:
            cfg = [[16]*n, [32]*n, [64]*n]
            cfg = [item for sub_list in cfg for item in sub_list]
        
        if num_blocks != None:
            assert(sum(num_blocks*2)==(depth-2))
            cfg = [[16]*num_blocks[0], [32]*num_blocks[1], [64]*num_blocks[2]]
            cfg = [item for sub_list in cfg for item in sub_list]
            
        self.cfg = cfg

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # import pdb; pdb.set_trace()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], cfg=cfg[sum(num_blocks[:0]):sum(num_blocks[0:1])])
        self.layer2 = self._make_layer(block, 32, num_blocks[1], cfg=cfg[sum(num_blocks[0:1]):sum(num_blocks[0:2])], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], cfg=cfg[sum(num_blocks[0:2]):sum(num_blocks[0:3])], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=planes*block.expansion)

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

if __name__ == '__main__':
    # net = resnet(depth=20, num_blocks=[3,3,3])
    # net = resnet(depth=32, num_blocks=[5,5,5])
    # net = resnet(depth=44, num_blocks=[7,7,7])
    # net = resnet(depth=56, num_blocks=[9,9,9])
    # net = resnet(depth=110, num_blocks=[18,18,18])
    net = resnet(depth=110, num_blocks=[18,18,18]).cuda() # resnet(depth=20, num_blocks=[3,3,3])
    x=Variable(torch.FloatTensor(16, 3, 32, 32).cuda())
    y = net(x)
    print(y.data.shape)