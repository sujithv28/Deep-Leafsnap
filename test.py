import argparse
import cv2
import json
import numpy as np
import os
import pandas as pd
import scipy.misc
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torchvision
import torchvision.models as models
import utils

from PIL import Image
from averagemeter import *
from models import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.utils.data import sampler
from torchvision import datasets
from torchvision import transforms

# GLOBAL CONSTANTS
INPUT_SIZE = 224
NUM_CLASSES = 185
NUM_EPOCHS = 35
LEARNING_RATE = 1e-1
USE_CUDA = torch.cuda.is_available()
best_prec1 = 0
classes = []

# ARGS Parser
parser = argparse.ArgumentParser(description='PyTorch LeafSnap Training')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
args = parser.parse_args()

# Test method
def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    class_correct = list(0. for i in range(185))
    class_total = list(0. for i in range(185))

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        if USE_CUDA:
            input = input.cuda(async=True)
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        _, predicted = torch.max(output.data, 1)
        print('\nGroundTruth: ', ' '.join('%5s' % classes[target_var[0][0]]))
        print('Predicted: ', ' '.join('%5s' % classes[predicted[0][0]]))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i, len(test_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

print('\n[INFO] Creating Model')
model = models.resnet101(pretrained=False)
model.fc = nn.Linear(2048, 185)

criterion = nn.CrossEntropyLoss()
if USE_CUDA:
    model = torch.nn.DataParallel(model).cuda()
    criterion = criterion.cuda()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                      momentum=0.9, weight_decay=1e-4, nesterov=True)

checkpoint = torch.load('model_best.pth.tar')
best_prec1 = checkpoint['best_prec1']
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
print('\n[INFO] Model Architecture: \n{}'.format(model))

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

print('\n[INFO] Reading Training and Testing Dataset')
traindir = os.path.join('dataset', 'train')
testdir = os.path.join('dataset', 'test')
iphonedir = os.path.join('dataset', 'iphone')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
data_train = datasets.ImageFolder(traindir, transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize]))
data_test = datasets.ImageFolder(testdir, transforms.Compose([
    transforms.ToTensor(),
    normalize]))
iphone_test = datasets.ImageFolder(iphonedir, transforms.Compose([
    transforms.ToTensor(),
    normalize]))
classes = data_train.classes

train_loader = torch.utils.data.DataLoader(data_train, batch_size=64, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=64, shuffle=False, num_workers=2)
iphone_loader = torch.utils.data.DataLoader(iphone_test, batch_size=64, shuffle=False, num_workers=2)

print('\n[INFO] Testing on Original Test Data Started')
prec1 = test(test_loader, model, criterion)
print(prec1)

print('\n[INFO] Testing on iPhone Test Data Started')
prec1 = test(iphone_loader, model, criterion)
print(prec1)

print('\n[DONE]')
