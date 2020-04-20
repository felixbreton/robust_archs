import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

#The following variables should be set :
#size, the size of the image
#sfn (=size//4-3), the size after two convolutions
#chans, the initial number of channels.

size=32
sfn=size//4-3
chans=3

class NetLin(nn.Module):
    def __init__(self):
        super(NetLin,self).__init__()
        self.fc1=nn.Linear(1024,10).to(device)

    def forward(self,x):
        x=x.view(-1,1024)
        x=self.fc1(x)
        return x

class Net2Fc(nn.Module):
    def __init__(self):
        super(Net2Fc,self).__init__()
        self.fc1=nn.Linear(1024,42).to(device)
        self.ReLU1=nn.ReLU(42).to(device)
        self.fc2=nn.Linear(42,10).to(device)

    def forward(self,x):
        x=x.view(-1,1024)
        x=self.fc1(x)
        x=self.ReLU1(x)
        out=self.fc2(x)
        return out

class NetCCFC(nn.Module):
    def __init__(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("Running on the GPU")
        else:
            device = torch.device("cpu")
            print("Running on the CPU")
        super(NetCCFC,self).__init__()
        self.conv1=nn.Conv2d(chans,8,5).to(device)
        self.conv2=nn.Conv2d(8,64,5).to(device)
        self.mp=nn.MaxPool2d(2,2).to(device)
        self.fc1=nn.Linear(64*sfn*sfn,120).to(device)
        self.fc2=nn.Linear(120,84).to(device)
        self.fc3=nn.Linear(84,10).to(device)

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
