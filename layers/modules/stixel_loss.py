import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import pickle

class StixelLoss(nn.Module):

    def __init__(self):
        super(StixelLoss, self).__init__()
        self.epoch=0

    def forward(self, predect , havetarget, target):
        target=(target-0.5).view(target.size(0),target.size(1),1)
        target=(target-torch.floor(target))+torch.floor(target)+0.0001
        target=target.view(target.size(0),target.size(1),1)
        fp=torch.gather(predect,2,torch.floor(target).type(torch.cuda.LongTensor))
        cp=torch.gather(predect,2,torch.ceil(target).type(torch.cuda.LongTensor))
        p=fp*(torch.ceil(target)-target)+cp*(target-torch.floor(target))
        p=p.view(havetarget.size(0),havetarget.size(1))
        loss=-torch.log(p)*havetarget
        loss=torch.sum(loss)/torch.sum(havetarget)
        return loss
