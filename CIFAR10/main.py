import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import time
import torchvision.transforms as transforms
from vgg import *
from googlenet import *
from densenet import *


parser = argparse.ArgumentParser(description="Pytorch CIFAR-X")
parser.add_argument('--wd', type=float, default=0.00, help='weight decay')
parser.add_argument('--trainBatchSize', type=int, default=128, help='input batch size for training')
parser.add_argument('--testBatchSize', type=int, default=100, help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--trainInterval', type=int, default=30,  help='how many batches to wait before logging training status')
parser.add_argument('--testInterval', type=int, default=50,  help='how many epochs to wait before another test')
parser.add_argument('--decreasingLR', default='80,120', help='decreasing strategy')
parser.add_argument('--resume', type=bool,default=False, help='resume from checkpoint')
args = parser.parse_args()

#加载数据
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.trainBatchSize, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.testBatchSize, shuffle=False, num_workers=2)

#当前cuda是否可用
args.cuda = torch.cuda.is_available()
print("CUDA: ", args.cuda)

#设置随机数种子，使随机初始化过程相同
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#加载模型
#net = VGG('VGG19')
#net = GoogLeNet()
net = densenet_cifar()
if args.cuda:
    net.cuda()

startEpoch = 0
best_acc = 0

#查看是否需要resume
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7',map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    startEpoch = checkpoint['epoch']


# optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

decreasingLR = list(map(int, args.decreasingLR.split(',')))
print('decreasingLR: ' + str(decreasingLR))

def train(epoch):
    net.train()

    if epoch in decreasingLR:
            optimizer.param_groups[0]['lr'] *= 0.1

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predicted = outputs.max(1)[1]
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % args.trainInterval == args.trainInterval-1:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' %(train_loss/(batch_idx+1), 100.*correct/total, correct, total),"lr =",optimizer.param_groups[0]['lr'])
        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    print("Test:\n")
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            predicted = outputs.max(1)[1]
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % args.testInterval == args.testInterval-1:
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)' %(test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

t_begin = time.time()
for epoch in range(startEpoch, startEpoch+args.epochs):
    print('\nEpoch: %d' % epoch)
    train(epoch)
    test(epoch)

print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-t_begin, best_acc))

# #torchvision.transforms函数初探
# img = Image.open('CIFAR10/train/2.png') #以PIL Image格式打开图片，该格式为 transforms各类函数的操作对象
# img1= mpimg.imread('CIFAR10/train/2.png') #img1 实际上为narray格式
# flip = transforms.RandomHorizontalFlip()
# normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.1, 0.1, 0.1))#该对象为tensor
# crop = transforms.RandomCrop(32, padding=4)
# plt.imshow(img) # 显示原始图片
# plt.show()
# plt.imshow(flip(img))
# plt.show()
# plt.imshow(crop(img))
# plt.show()
