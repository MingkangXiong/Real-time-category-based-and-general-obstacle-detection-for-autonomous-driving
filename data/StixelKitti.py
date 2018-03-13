import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

class StixelKitti(data.Dataset):

    def __init__(self, basepath,labeltpath,transform=None, target_transform=None):
        self.basepath=basepath
        self.lpath=labeltpath
        self.imgpath='%s/2011_%s/2011_%s_drive_%04d_sync/image_02/data/%010d.png'
        self.transform = transform
        self.target_transform = target_transform
        self.gt={}
        gtfile = open(self.lpath)
        data = gtfile.readlines()
        date=''
        index=''
        frame=''
        self.imgpos=[]
        self.imginfo=[]
        pointset=[]
        for line in data:
            line_info = line.split('\t')
            if not os.path.isfile(self.imgpath%(self.basepath,line_info[0],line_info[0],int(line_info[1]),int(line_info[2]))):
                continue
            if (line_info[0]==date and line_info[1]==index and line_info[2]==frame):
                pointset.append((int(line_info[3]),int(line_info[4])))
            else:
                if date!='':
                    self.imginfo.append(pointset)
                pointset=[]
                pointset.append((int(line_info[3]), int(line_info[4])))
                date=line_info[0]
                index=line_info[1]
                frame=line_info[2]
                self.imgpos.append([date,int(index),int(frame)])
        self.imginfo.append(pointset)
        self.lenth=len(self.imginfo)


    def __getitem__(self, index):
        date=self.imgpos[index][0]
        id=self.imgpos[index][1]
        frame=self.imgpos[index][2]
        img=cv2.imread(self.imgpath%(self.basepath,date,date,id,frame))
        h,w,c=img.shape
        img, _a, _b = self.transform(img,None,None)
        img = img[:, :, (2, 1, 0)]
        img=torch.from_numpy(img).permute(2, 0, 1)
        havegt, gt =self.get_target(self.imginfo[index],w,h)

        return img,havegt,gt

    def get_target(self,pointset,w,h):
        rownum=100
        binnum=50
        pointset=np.array(pointset,dtype=np.float32)
        pointset[:,0]=pointset[:,0]/w
        pointset[:,1]=pointset[:,1]/h
        havegt=np.zeros((rownum),dtype=np.float32)
        gt=np.zeros((rownum),dtype=np.float32)
        for p in pointset:
            index=int(p[0]*rownum)
            op=p[1]*binnum
            if havegt[index]==1:
                gt[index]=(gt[index]+op)/2
            else:
                gt[index]=op
                havegt[index]=1
        gt=np.clip(gt,0.51,49.49)
        return havegt,gt


    def __len__(self):
        return 1