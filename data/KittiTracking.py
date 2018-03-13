"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np




classify={'Car':0, 'Van':1, 'Truck':2, 'Pedestrian':3, 'Person_sitting':4, 'Cyclist':5, 'Tram':6, 'Misc':7, 'DontCare':8}



class KittiTracking(data.Dataset):

    def __init__(self, basepath,index,transform=None):
        self.lpath='%s/tracking/training/label_02/%04d.txt'%(basepath,index)
        self.imgpath='%s/tracking/training/image_02/%04d'%(basepath,index)
        self.transform = transform
        self.gt={}
        gtfile = open(self.lpath)
        data = gtfile.readlines()
        for line in data:
            line_info = line.split(' ')
            frame=int(line_info[0])
            l = int(float(line_info[6]))
            t = int(float(line_info[7]))
            r = int(float(line_info[8]))
            b = int(float(line_info[9]))
            if frame in self.gt.keys():
                self.gt[frame].append([l,t,r,b,classify[line_info[2]]])
            else:
                self.gt[frame]=[]
                self.gt[frame].append([l, t, r, b, classify[line_info[2]]])
        self.lenth=max(self.gt.keys())

    def pull_item(self,index):
        img=cv2.imread('%s/%06d.png'%(self.imgpath,index))
        oimg=img
        height, width, channels = img.shape
        target=[]
        for i in self.gt[index]:
            t=[]
            t.append(float(i[0]) / img.shape[1])
            t.append(float(i[1]) / img.shape[0])
            t.append(float(i[2]) / img.shape[1])
            t.append(float(i[3]) / img.shape[0])
            t.append(i[4])
            target.append(t)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target,height,width,oimg

    def __getitem__(self, index):
        if index not in self.gt.keys():
            return self.__getitem__(index-1)
        im, gt, h, w, oimg = self.pull_item(index)

        return im, gt

    def __len__(self):
        return self.lenth

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets
