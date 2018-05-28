# pytorch-net
## CIFAR10
1、VGG  

2、ResNet  

3、GoogleNet  


## VOC
Models：FCN32S FCN16S FCN8S

## Requirement
Pytorch0.4

## Data
1、Visit [this](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/data/pascal), download SBD and PASCAL VOC 2012

2、Extract them, you will get benchmark_RELEASE and VOC2012 folders.

3、Add file seg11valid.txt ([download](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/data/pascal/seg11valid.txt)) into VOC2012/ImageSets/Segmentation

4、Put the benchmark_RELEASE and VOC2012 folders in a folder called data

5、Put the data folder as the same level as the main.py file


## VGG parameters

[Here](https://download.pytorch.org/models/vgg16-397923af.pth) download vgg pretrained parameter and put it(named vgg16.pth) in the models folder


## Result

Mean-IU 56%
