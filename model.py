import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    '''18层和34层的网络'''
    expansion = 1 # 主分支卷积核个数是否发生变化
    def __init__(self, in_channel, out_channel, stride = 1, downsample = None):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channel,out_channels = out_channel, stride = stride, 
                                kernel_size = 3, padding = 1, bias = False)  #使用bn层，则不用bias
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = out_channel,out_channels = out_channel, stride = 1, 
                                kernel_size = 3, padding = 1, bias = False)  
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self,x):
        identity = x # 捷径分支上的输出值
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out  = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # 50,101,152层残差结构
    expansion = 4 #残差结构使用第三层卷积卷积核的变化

    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channel,out_channels = out_channel, stride = 1, 
                                kernel_size = 1, bias = False)  #使用bn层，则不用bias
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels = out_channel,out_channels = out_channel, stride = stride, 
                                kernel_size = 3, padding = 1, bias = False)  
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(in_channels = out_channel,out_channels = out_channel*self.expansion, stride = 1, 
                                kernel_size = 1, bias = False)  
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out  = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out  = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    
    def __init__(self,block,blocks_num,num_classes=1000,include_top=True):
        '''
        block = BasicBlock  或 Bottleneck
        blocks_num 每一个layer有多少个block
        include_top 括展resnet使用，本节用不到
        '''
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 =nn.Conv2d(in_channels=3,out_channels=self.in_channel,kernel_size=7,
                                stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # 自适应平均池化下采样，不论输入什么，输出都是(1,1)'
            self.fc = nn.Linear(512 * block.expansion,num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def _make_layer(self, block, channel, blocks_num, stride=1): # 难点、重点
        '''
        block : BasicBlock or Bottleneck
        channel: 残差结构第一层的卷积核个数
        blocks_num: 该层包含多少个残差结构
        '''
        downsample =None
        if stride != 1 or self.in_channel != channel*block.expansion:# 虚线残差结构，18,34层网络用不到
            downsample =nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel,out_channels=channel*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(channel*block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel,channel,downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        # 压入实线残差结构
        for _ in range(1,blocks_num):
            layers.append(block(self.in_channel,channel))

        return nn.Sequential(*layers) #非关键字参数
        
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x,1)
            x = self.fc(x)

        return x

def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)