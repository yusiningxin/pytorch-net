from loadData import *
import torch
import argparse
import random
import datetime
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
#from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import utils.trans as trans
import models.fcn32 as fcn32
from torch.optim.lr_scheduler import ReduceLROnPlateau
import models.fcn16 as fcn16
import models.fcn8 as fcn8
import torchvision.transforms as transforms
from utils.misc import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d


# 不清楚这个的具体意思
#cudnn.benchmark = True

#设置tensorboard路径
ckpt_path = 'ckpt'
exp_name = 'voc-fcn-1'
# writer = SummaryWriter(os.path.join(ckpt_path, 'exp', exp_name))


#参数设置
parser = argparse.ArgumentParser(description="Pytorch CIFAR-X")
parser.add_argument('--wd', type=float, default=0, help='weight decay')
parser.add_argument('--trainBatchSize', type=int, default=1, help='input batch size for training')
parser.add_argument('--testBatchSize', type=int, default=1, help='input batch size for testing')
parser.add_argument('--momentum', type=float,default=0.95, help='momentum')
parser.add_argument('--lr_patience', type=int,default=10, help='large patience denotes fixed lr')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 1e-3)')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--trainInterval', type=int, default=30,  help='how many batches to wait before logging training status')
parser.add_argument('--testInterval', type=int, default=50,  help='how many epochs to wait before another test')
parser.add_argument('--decreasingLR', default='80,120', help='decreasing strategy')
parser.add_argument('--resume',default='', help='resume from checkpoint')
parser.add_argument('--valImgSampleRate', type=float,default=0.9, help='rate')
parser.add_argument('--val_save_to_img_file', type=bool,default=True, help='save')
args = parser.parse_args()


#当前cuda是否可用
args.cuda = torch.cuda.is_available()
print("CUDA: ", args.cuda)

#设置随机数种子，使随机初始化过程相同
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


#加载模型
#net = fcn32.FCN32VGG(num_classes = num_classes)
#net = fcn16.FCN16VGG(num_classes = num_classes)
net = fcn8.FCN8VGG(num_classes = num_classes)
if args.cuda:
    net = net.cuda()


best_record={}
curr_epoch=0

# 查看是否需要resume
if len(args.resume)!=0:
    # Load checkpoint.
    print('training resumes from ' + args.resume)
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args.resume)))
    split_snapshot = args.resume.split('_')
    curr_epoch = int(split_snapshot[1]) + 1
    best_record = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                                 'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                                 'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11])}
else:
    curr_epoch = 1
    best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}



mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

inputTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
])
# 转换为tensor格式
labelTransform = trans.MaskToTensor()

restore_transform = transforms.Compose([
    trans.DeNormalize(*mean_std),
    transforms.ToPILImage()
])

# visualize = transforms.Compose([
#     transforms.Scale(400),
#     transforms.CenterCrop(400),
#     transforms.ToTensor()
# ])

trainset = VOC('train', inputTransform=inputTransform, labelTransform=labelTransform)
trainloader = DataLoader(trainset, batch_size=args.trainBatchSize, num_workers=4, shuffle=True)

valset = VOC('val', inputTransform=inputTransform, labelTransform=labelTransform)
valloader = DataLoader(valset, batch_size=args.testBatchSize, num_workers=4, shuffle=False)


#Loss function
if args.cuda:
    criterion = CrossEntropyLoss2d(size_average=True, ignore_index=ignore_label).cuda()
else:
    criterion = CrossEntropyLoss2d(size_average=True, ignore_index=ignore_label)



optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.lr_patience, min_lr=1e-10, verbose=True)
# optimizer = optim.Adam([ {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],'lr': 2 * args.lr},
# {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],'lr': args.lr, 'weight_decay': args.wd}],
# betas=(args.momentum, 0.999))

# resume 优化器参数加载
if len(args.resume) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_' + args.resume)))
        optimizer.param_groups[0]['lr'] = 2 * args.lr
        optimizer.param_groups[1]['lr'] = args.lr

check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
#open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(args) + '\n\n')



def train(epoch):
    net.train()
    # 计算平均损失，每个epoch更新为0
    train_loss = AverageMeter()
    #每次迭代调用 _getitem_ 方法，进行transform变换。
    curr_iter = (epoch - 1) * len(trainloader)
    for i, (inputs,labels) in enumerate(trainloader):
        if args.cuda:
            inputs,labels = inputs.cuda(),labels.cuda()
        N = inputs.size(0)
        # 清空梯度
        optimizer.zero_grad()
        outputs = net(inputs)
        # 计算单个样本的loss
        loss = criterion(outputs, labels) / N
        # 反向传导，更新参数
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), N)
        curr_iter += 1

        #writer.add_scalar('train_loss', train_loss.avg, curr_iter)
        if (i + 1) % args.trainInterval == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (epoch, i + 1, len(trainloader), train_loss.avg))

# 验证过程
def validate (epoch):
    net.eval()
    val_loss = AverageMeter()
    inputs_all, labels_all, predictions_all = [], [], []

    for i, (inputs,labels) in enumerate(valloader):
        if args.cuda:
            inputs,labels = inputs.cuda(),labels.cuda()
        N = inputs.size(0)
        outputs = net(inputs)
        # predictions 为输入图片尺寸大小的对应每一像素点的分类值
        predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        loss = criterion(outputs, labels)/N
        val_loss.update(loss.item(), N)

        if random.random() > args.valImgSampleRate:
            inputs_all.append(None)
        else:
            inputs_all.append(inputs.data.squeeze_(0).cpu())

        labels_all.append(labels.data.squeeze_(0).cpu().numpy())

        predictions_all.append(predictions)

    # 计算本次epoch之后对验证集的正确率等评价指标
    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, labels_all, num_classes)

    if mean_iu > best_record['mean_iu']:
        best_record['val_loss'] = val_loss.avg
        best_record['epoch'] = epoch
        best_record['acc'] = acc
        best_record['acc_cls'] = acc_cls
        best_record['mean_iu'] = mean_iu
        best_record['fwavacc'] = fwavacc
        snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
            epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, args.lr)
        torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
        #torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_' + snapshot_name + '.pth'))

        if args.val_save_to_img_file:
            to_save_dir = os.path.join(ckpt_path, exp_name, str(epoch))
            check_mkdir(to_save_dir)

        #val_visual = []
        for idx, data in enumerate(zip(inputs_all, labels_all, predictions_all)):
            if data[0] is None:
                continue
            input_pil = restore_transform(data[0])
            labels_pil = colorize_mask(data[1])
            predictions_pil = colorize_mask(data[2])
            if args.val_save_to_img_file:
                input_pil.save(os.path.join(to_save_dir, '%d_input.png' % idx))
                predictions_pil.save(os.path.join(to_save_dir, '%d_prediction.png' % idx))
                labels_pil.save(os.path.join(to_save_dir, '%d_label.png' % idx))
            # val_visual.extend([visualize(input_pil.convert('RGB')), visualize(labels_pil.convert('RGB')),
            #                    visualize(predictions_pil.convert('RGB'))])
        # val_visual = torch.stack(val_visual, 0)
        # val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
        # writer.add_image(snapshot_name, val_visual)

    print('--------------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
        best_record['val_loss'], best_record['acc'], best_record['acc_cls'],
        best_record['mean_iu'], best_record['fwavacc'],best_record['epoch']))

    print('--------------------------------------------------------------------')

    # writer.add_scalar('val_loss', val_loss.avg, epoch)
    # writer.add_scalar('acc', acc, epoch)
    # writer.add_scalar('acc_cls', acc_cls, epoch)
    # writer.add_scalar('mean_iu', mean_iu, epoch)
    # writer.add_scalar('fwavacc', fwavacc, epoch)
    # writer.add_scalar('lr', optimizer.param_groups[1]['lr'], epoch)

    return val_loss.avg


for epoch in range(curr_epoch, args.epochs):
    train(epoch)
    val_loss = validate(epoch)
    scheduler.step(val_loss)
