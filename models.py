import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from segmentation import *
from math import log2

parFile=open("params.txt")
params=list(filter(lambda s: len(s)>0 and s[0]!='#',parFile.read().splitlines()))

def nChans(dataset):
    if dataset=="SC":
        return 1
    else:
        return 3

size=int(params[1])
chans=nChans(params[2])

applyDilation=True#untested

if applyDilation and size>=32:
    n=int(log2(size/32))
    d1=pow(2,(n+1)//2)
    d2=pow(2,n//2)
else:
    d1=1
    d2=1

sfn=size//4-d1-2*d2

class NetCCFC(nn.Module):
    def __init__(self,nClasses):
        super(NetCCFC,self).__init__()
        self.conv1=nn.Conv2d(chans,8,5,dilation=d1)
        self.conv2=nn.Conv2d(8,64,5,dilation=d2)
        self.mp=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(64*sfn*sfn,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,nClasses)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.mp(x)
        x=F.relu(self.conv2(x))
        x=self.mp(x)
        x = x.view(-1, 16 * 4 * sfn * sfn)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
    
def img2fc(img,n):
    segs=seg_SLIC(img.cpu(),n)
    segs=segs.flatten()
    zones=[set() for i in range(n)]
    for i in range(len(segs)):
        zones[segs[i]].add(i)
    out=nn.Linear(3072,3072)
    out.bias.data=torch.zeros_like(out.bias.data)
    out.weight.data=torch.zeros_like(out.weight.data)
    for i in range(n):
        if len(zones[i])!=0:
            sz=len(zones[i])
            for p1 in zones[i]:
                for p2 in zones[i]:
                    for c in range(3):
                        out.weight.data[p1+32*32*c][p2+32*32*c]=1/sz
    return out

class seg_NetCCFC(NetCCFC):
    def __init__(self,nClasses):
        super(NetCCFC,self).__init__()
        self.conv1=nn.Conv2d(chans,8,5,dilation=d1)
        self.conv2=nn.Conv2d(8,64,5,dilation=d2)
        self.mp=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(64*sfn*sfn,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,nClasses)
        self.fcAvg=nn.Linear(3072,3072)

    def forward(self,x):
        x=batchSeg(x,256,seg_SLIC)
        """self.fcAvg=img2fc(x[0],64)
        x=x.view(-1)
        x=self.fcAvg(x)
        x=x.view(1,3,32,32)"""
        x=F.relu(self.conv1(x))
        x=self.mp(x)
        x=F.relu(self.conv2(x))
        x=self.mp(x)
        x = x.view(-1, 16 * 4 * sfn * sfn)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
