import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from data import SQUARE_CIRCLE,MNIST,CIFAR10,IMGNET12
from segmentation import seg_quant,seg_kMeans_2d
from models import *
from time import *

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
device=torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

size=32
sfn=size//4-3
chans=3
#Remember to update models.py when changing the size

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

load=True
if load==True:
    tr_loader, va_loader, te_loader = CIFAR10(
            bs=4, valid_size=.1,
            size=size, normalize=True)
    trainloader = tr_loader

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def dispGrid_(images):
    imshow(torchvision.utils.make_grid(images.reshape(4,chans,size,size)))

def reshape(img):
    return torch.flatten(img)

def test(net,images,labels):
    # show images
    dispGrid_(images)
    #print labels
    print(' '.join('%5s' % classes[labels[j].item()] for j in range(4)))

    #images=torch.stack([reshape(im) for im in images])
    out=net.forward(images)
    tots=out.exp().sum(1)
    vals,reps=torch.max(out,1)
    print([classes[i] for i in reps],vals.exp()/tots)

def train(epochs,verbose=True,path=""):
    global device
    if path=="":
        net=NetCCFC().to(device)
    else:
        net=NetCCFC()
        net.load_state_dict(torch.load(path))
        net.eval()

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    #optimizer=optim.Adam(net.parameters(),lr=0.0003)

    dataiter = iter(trainloader)
    cuml=0
    accAvg=0
    deb=time()
    tload=0
    ttrain=0
    for e in range(epochs):
        for i,data in enumerate(trainloader):
            dload=time()
            images,labels=data
            images=images.to(device)
            labels=labels.to(device)
            tload+=time()-dload

            dtrain=time()
            out=net.forward(images)
            loss=criterion(out,labels)
            reps=out.argmax(1)
            acc=((labels==reps).sum()).float()/len(images)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ttrain+=time()-dtrain

            cuml+=loss
            accAvg+=acc
            if i%1000==0:
                print(i,loss,cuml/1000,accAvg/1000)
                cuml=0
                accAvg=0
        torch.save(net.state_dict(),"./net/"+str(e)+".pt")
        if verbose:
            print("Epoch : "+str(e)+"Time : "+str(round(time()-deb,2)))
            print("Time spent loading : "+str(round(tload,2)))
            print("Time spent training : "+str(round(ttrain,2)))

train(2)

"""images,labels=dataiter.next()
images=images.to(device)
labels=labels.to(device)
test(net,images,labels)"""

def oracle(image):
    parts={}
    ret=[]
    for i in image.flatten():
        val=i.item()
        if not val in parts:
            parts[val]=len(parts)
        ret.append(parts[val])
    return torch.tensor(ret).view(*image.shape)

def batchH(base,a):
    tab=[avg(a[i],oracle(base[i])) for i in range(len(base))]
    return torch.stack(tab)

def adv():
    images,labels=dataiter.next()
    #images=torch.stack([reshape(im) for im in images])
    images.requires_grad_(True)

    out=net.forward(images)
    loss=criterion(out,(labels+5)%10)

    loss.backward(retain_graph=True)
    optimizer.zero_grad()

def mk_adv(net,images,labels):
    out=net.forward(images)
    loss=criterion(out,(labels+5)%10)

    loss.backward(retain_graph=True)
    optimizer.zero_grad()

    return (images-5000*images.grad.clamp(-0.000001,0.000001)).clamp(-1,1).detach().requires_grad_(True)

def mk_adv_oracle(net,images,labels):
    out=net.forward(images)
    loss=criterion(out,(labels+1)%2)

    loss.backward(retain_graph=True)
    optimizer.zero_grad()

    grad=batchH(images,images.grad)
    return (images-5000*grad.clamp(-0.000001,0.000001)).clamp(-1,1).detach().requires_grad_(True)

def ev_grad():
    images.requires_grad_(True)

    out=net.forward(images)
    loss=criterion(out,labels)

    loss.backward(retain_graph=True)
    for i in range(len(images)):
        print(images.grad[i].norm())
    optimizer.zero_grad()

def segment_deb(images,n):
    ref=quant(images,n)
    return batchH(ref,images)

def batchSeg(images,n,seg):
    tab=[avg_seg(images[i],seg(images[i],n,01.5)) for i in range(len(images))]
    return torch.stack(tab)
