from __future__ import print_function
import argparse
import numpy as np
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from models import resnet
from models import densenet
from utils import get_logger, makedirs

parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./experiment1', type=str, metavar='PATH',
                    help='path to save model (default: current directory)')
parser.add_argument('--arch', default='resnet', type=str, 
                    help='architecture to use')
parser.add_argument('--data', default='../ESNB-cifar/', type=str, 
                    help='path of dataset')
parser.add_argument('--depth', default=20, type=int,
                    help='depth of the neural network')

parser.add_argument('--teacher', default='./experiments_densenet/', type=str, 
                    help='path of teacher model')
parser.add_argument('--depth_dense', type=int, default=100, help='Model depth.')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=0.5, help='Compression Rate (theta) for DenseNet.')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')

parser.add_argument('--t', '--temperature', default=2, type=float,
                    metavar='Temperature', help='Temperature')
parser.add_argument('--alpha', default=0.5, type=float,
                    metavar='Alpha', help='Alpha')


args = parser.parse_args()

makedirs(args.save)
logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)
    
    
args.cuda = not args.no_cuda and torch.cuda.is_available()
num_blocks = [(args.depth-2)//(3*2),] * 3

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data_path = args.data
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
num_classes = 10
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    num_classes = 100
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(data_path+'data.cifar100', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(data_path+'data.cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

logger.info("Loading teacher model from {}".format(args.teacher))
model_teacher = densenet.densenet(
        num_classes=num_classes,
        depth=args.depth_dense,
        block=densenet.Bottleneck,
        growthRate=args.growthRate,
        compressionRate=args.compressionRate,
        dropRate=args.drop,
        )
checkpoint_path = args.teacher + "/model_best.pth.tar"
model_teacher.load_state_dict(torch.load(checkpoint_path)['state_dict'])
logger.info(model_teacher)

logger.info("Initializing student model...")
model = resnet.resnet(depth=args.depth, num_classes=num_classes, num_blocks=num_blocks)
logger.info(model)

if args.cuda:
    model_teacher.cuda()
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume))

def train(epoch):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def train_kd(epoch, model_student, model_teacher, T=2, alpha=0.5):
    # import pdb; pdb.set_trace()
    model_student.train()
    model_teacher.eval()
    avg_loss = 0.
    train_acc = 0.
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        
        with torch.no_grad():
            output_teacher = model_teacher(data)
        
        optimizer.zero_grad()
        output_student = model_student(data)
        
        # KD loss
        loss1 = F.cross_entropy(output_student, target)
        # loss2 = F.kl_div(F.log_softmax(output_student/T,dim=1), F.softmax(output_teacher/T,dim=1), reduction="sum") * T * T / output_student.shape[0] 
        # correct outputs from teacher #
        # import pdb; pdb.set_trace()
        pred_teacher = output_teacher.data.max(1, keepdim=True)[1]
        correct_idx = (target == torch.squeeze(pred_teacher)) # find out the indices of samples correctly classified by the teacher
        loss2_improved = F.kl_div(F.log_softmax(output_student[correct_idx]/T,dim=1), \
                         F.softmax(output_teacher[correct_idx]/T,dim=1), reduction="sum") * T * T / target.size(0) 
        # correct outputs from teacher #
        loss = loss1*(1-alpha) + loss2_improved*alpha

        avg_loss += loss.item()
        total += target.size(0)
        pred = output_student.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total += target.size(0)

    test_loss /= total
    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, total, 100. * correct / total))
    return (100. * correct / total)

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

        
acc_teacher = test(model_teacher)
logger.info("Best acc of teacher model:{}\n".format(acc_teacher))
logger.info("Temperature:{}, alpha:{}".format(args.t, args.alpha))
best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    if epoch in [args.epochs*0.25, args.epochs*0.5, args.epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train_kd(epoch, model, model_teacher, T=args.t, alpha=args.alpha)
    prec1 = test(model)
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    logger.info("Best acc:{}\n".format(best_prec1))
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
        'cfg': model.cfg
    }, is_best, filepath=args.save)
