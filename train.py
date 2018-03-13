import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnnz
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import *
from utils.augmentations import SSDAugmentation,StixelAugmentation
from layers.modules import MultiBoxLoss,StixelLoss
from StixelNet import build_net
import numpy as np
import time



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=3e-5, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.9, type=float, help='Gamma update for SGD')
parser.add_argument('--basepath',type=str)
parser.add_argument('--resume',type=str)

args = parser.parse_args()


torch.set_default_tensor_type('torch.cuda.FloatTensor')
ssd_dim = (800,370)  # the size of image after resize (width,height)
means = (104, 117, 123)
num_classes = 9 + 1
batch_size = args.batch_size

def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

if args.resume is None:
    net = build_net('train', ssd_dim, num_classes)
    vgg_weights = torch.load('weights/vgg16_reducedfc.pth')
    print('Loading base network...')
    net.vgg.load_state_dict(vgg_weights)
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
else:
    net=torch.load(args.resume)


savename='weights/kitti_%f_%.3f'%(args.lr,args.gamma)

cudnnz.benchmark = True





net = net.cuda()



optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)





def dection_train():
    logfile=open("./log/log_%f_%.3f.txt"%(args.lr,args.gamma),"w")
    net.train()
    step=0
    index=1
    criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, True)
    dataset=KittiTracking(args.basepath,index,SSDAugmentation(size=ssd_dim, mean=means))
    data_loader=data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                      shuffle=True, collate_fn=detection_collate, pin_memory=True)
    for epoch in range(500):
        if epoch%10==0:
            lr=adjust_learning_rate(optimizer, args.gamma, step)
            step = step + 1
            print("present learning rate is %.6f"%lr)
        loc_loss = 0  # epoch
        conf_loss = 0
        for i,(images,targets) in enumerate(data_loader):
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            # forward
            dec, stixel = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(dec, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            loc_loss += loss_l.data[0]
            conf_loss += loss_c.data[0]
            if i % 10 == 0:
                print("Epoch:%d batch:%d/%d  loss:%.4f L(loc):%.4f L(conf):%.4f"%(epoch,i,len(data_loader),loss.data[0],loss_l.data[0],loss_c.data[0]))
        loc_loss=loc_loss*batch_size/len(dataset)
        conf_loss=conf_loss*batch_size/len(dataset)
        print("Totol loss:%.4f L(loc):%.4f L(conf):%.4f"%(loc_loss+conf_loss,loc_loss,conf_loss))
        logfile.write("Epoch:%d Totol loss:%.4f L(loc):%.4f L(conf):%.4f\n"%(epoch,loc_loss+conf_loss,loc_loss,conf_loss))
        if epoch % 10 == 0 and epoch>0:
                print('Saving state, epoch:', epoch)
                torch.save(net, savename+('_%d.pth'%epoch))
    torch.save(net, savename+('_%d.pth'%epoch))
    logfile.close()

def stixel_train():
    logfile=open("./log/log_%f_%.3f.txt"%(args.lr,args.gamma),"w")
    net.train()
    printfrq=10
    step=0
    dataset = StixelKitti('/data/Stixel','/data/Stixel/StixelsGroundTruth.txt',StixelAugmentation(size=ssd_dim, mean=means))
    data_loader=data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=False, pin_memory=True)
    lossfunction=StixelLoss()
    minloss=9999
    for epoch in range(200):
        if epoch%10==0:
            lr=adjust_learning_rate(optimizer, args.gamma, step)
            step = step + 1

        avgloss = 0
        for i,(images,havetargets,targets) in enumerate(data_loader):
            images=Variable(images).cuda()
            havetargets=Variable(havetargets).cuda()
            targets=Variable(targets).cuda()
            dec , stixel =net(images)
            optimizer.zero_grad()
            loss=lossfunction(stixel,havetargets,targets)
            loss.backward()
            optimizer.step()
            avgloss=avgloss+loss.data[0]
            if i % printfrq == 0:
                if i!=0:
                    avgloss=avgloss/printfrq
                print("Epoch: %d batch: %d lr: %.6f loss: %.6f" % (
                    epoch, i, lr,avgloss))
                logfile.write("Epoch: %d batch: %d lr: %.6f loss: %.6f\n" % (
                    epoch, i, lr,avgloss))
                if avgloss < minloss:
                    minloss = avgloss
                avgloss=0
        if epoch % 10 == 0 and epoch>0:
            torch.save(net, savename+('_%d.pth'%epoch))
    torch.save(net, savename+('_%d.pth'%epoch))
    logfile.close()



def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    #dection_train()
    stixel_train()
