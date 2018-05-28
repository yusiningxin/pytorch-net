import torch
import os,sys
sys.path.append('../')
from torch import nn
from torchvision import models

from utils.misc import get_upsampling_weight


class FCN32VGG(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(FCN32VGG, self).__init__()
        vgg = models.vgg16()

        # 需要 fine-tuning 过程，预训练出参数。
        if pretrained:
            #vgg.load_state_dict(torch.load(os.path.join('models','vgg16.pth')))
            vgg.load_state_dict(torch.load(os.path.join('vgg16.pth')))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        #classifier: (25088,4096) RELU Dropout (4096,4096) RELU Dropout (4096,1000)

        '''
        100 padding for 2 reasons:
            1) support very small input size
            2) allow cropping in order to match size of different layers' feature maps
        Note that the cropped part corresponds to a part of the 100 padding
        Spatial information of different layers' feature maps cannot be align exactly because of cropping, which is bad
        '''
        #padding 变为100，为了防止之后输出的尺寸太小
        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                # when True, will use ceil instead of floor to compute the output shape
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features5 = nn.Sequential(*features)

        # fc6 卷基层参数数量 7*7*512*4096 = 25088 * 4096
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

        #上采样过程，转置卷积
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, bias=False)
        self.upscore.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 64))


    def forward(self, x):
        x_size = x.size()
        pool5 = self.features5(x)
        score_fr = self.score_fr(pool5)

        upscore = self.upscore(score_fr)
        print(score_fr.size(),upscore.size())
        return upscore[:, :, 19: (19 + x_size[2]), 19: (19 + x_size[3])].contiguous()

def test():
    net = FCN32VGG(10)
    y = net(torch.randn(1,3,45,78))

#test()
