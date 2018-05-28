import torch
import os,sys
sys.path.append('../')
from torch import nn
from torchvision import models

from utils.misc import get_upsampling_weight


class FCN16VGG(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(FCN16VGG, self).__init__()
        vgg = models.vgg16()

        if pretrained:
            vgg.load_state_dict(torch.load(os.path.join('models','vgg16.pth')))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features4 = nn.Sequential(*features[: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)

        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)

        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()

        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

        # skip layers
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16, bias=False)

        self.upscore2.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore16.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 32))

    def forward(self, x):
        x_size = x.size()
        pool4 = self.features4(x) # pool4的channel维度为512，大小相比原来缩小了16倍

        pool5 = self.features5(pool4)
        score_fr = self.score_fr(pool5) # score_fr 的channel维度为21，大小相比原来缩小了32倍


        # 将score_fr上采样扩大两倍,变为缩小16倍
        upscore2 = self.upscore2(score_fr)

        # 用1*1的卷积进行降维，channel变为21
        score_pool4 = self.score_pool4(0.01 * pool4)

        #score_pool4 进行crop后和upscore2相加
        upscore16 = self.upscore16(score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])]
                                   + upscore2)
                                   
        return upscore16[:, :, 27: (27 + x_size[2]), 27: (27 + x_size[3])].contiguous()

def test():
    net = FCN16VGG(21)
    y = net(torch.randn(1,3,45,78))
    print(y.size())

#test()
