from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import KittiTracking,StixelKitti,BaseTransform
from data import classify
import torch.utils.data as data
import os
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--basepath',type=str,help='The basepath of KittiTracking')
parser.add_argument('--outpath',default='./eval/',type=str,help='The path of output demo')
parser.add_argument('--model',type=str,help='The path of model to evaluate')
parser.add_argument('--index',type=int,default=1,help='the index of KittiTracking video')

args = parser.parse_args()

means = (104, 117, 123)
ssd_dim = (800,370)

if not os.path.exists(args.outpath):
    os.mkdir(args.outpath)

def dection_test(dataset,model):
    num_images = len(dataset)
    for i in range(num_images):
        im, gt, h, w, oimg = dataset.pull_item(i)
        x = Variable(im.unsqueeze(0))
        x = x.cuda()
        detections,stx = model(x)
        detections=detections.data
        for j in range(detections.size(1)):
            for k in range(detections.size(2)):
                confidence = detections[0, j, k, 0]
                if confidence > 0.3:
                    l = int(detections[0, j, k, 1] * w)
                    t = int(detections[0, j, k, 2] * h)
                    r = int(detections[0, j, k, 3] * w)
                    b = int(detections[0, j, k, 4] * h)
                    cv2.rectangle(oimg, (l, t), (r, b), (0, 255, 0), 1)
        cv2.imwrite(os.path.join(args.outpath,'%d.png'%i), oimg)
        print("finish %d/%d" % (i, num_images))


def stixel_test(dataset,model):
    num_images = len(dataset)
    for i in range(num_images):
        im, tar, h, w, oimg = dataset.pull_item(i)
        x = Variable(im.unsqueeze(0)).cuda()
        dec,stixel = model(x)
        predict=stixel.data.cpu().numpy()[0]
        predict=predict.argmax(1)
        for x,py in enumerate(predict):
            x0=int(x*w/100)
            x1=int((x+1)*w/100)
            y=int((py+0.5)*h/50)
            cv2.line(oimg,(x0,y),(x1,y),(0,255,255),1)
        cv2.imwrite(os.path.join(args.outpath,'%d.png'%i),oimg)
        print("finish %d/%d"%(i,num_images))






if __name__ == '__main__':
    # load net
    num_classes = 9 + 1 # +1 background
    net=torch.load(args.model)
    net.eval()
    print('Finished loading model!')
    dataset=KittiTracking(args.basepath, 20, BaseTransform(ssd_dim, means))
    net = net.cuda()
    cudnn.benchmark = True
    dection_test(dataset,net)
    #stixel_test(dataset,net)
