import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import foolbox as fb
from data import *
from segmentation import *
from models import *
from time import *
import sys
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--nsegs',type=int)
parser.add_argument('--eps',type=float)
parser.add_argument('--attack',type=str)
args=parser.parse_args()
nsegs=args.nsegs
eps=args.eps
attack=args.attack
if nsegs==None:
    nsegs=64
if eps==None:
    eps=0.1
if attack==None:
    attack="L2PGD"
#parameters that (currently) must be hardcoded :
#number of tests
#whether to attack the segmenter

parFile=open("params.txt")
params=list(filter(lambda s: len(s)>0 and s[0]!='#',parFile.read().splitlines()))

if params[0].lower()=='true' and torch.cuda.is_available():
    device = torch.device("cuda:0")
    workers=8
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    workers=0
    print("Running on the CPU")

size=int(params[1])
sfn=size//4-3
chans=nChans(params[2])
batchSize=int(params[3])

t0=time()

if params[2]=="CIFAR":
    nClasses=10
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')
    tr_loader, va_loader, te_loader = CIFAR10(
            bs=batchSize, valid_size=.1,
            size=size, normalize=True,num_workers=workers)
    #tr_loader,va_loader,te_loader=FROM_FILE('slic64_',bs=batchSize)
elif params[2]=="SC":
    nClasses=2
    classes = ('square', 'circle')
    tr_loader, va_loader, te_loader = SQUARE_CIRCLE(
            bs=batchSize, valid_size=.1,
            size=size, normalize=False)
else:
    print("Dataset "+params[2]+" unknown")
    sys.exit(0)

print("Initial loading time : "+str(time()-t0))

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def dispGrid_(images):
    imshow(torchvision.utils.make_grid(images.reshape(batchSize,chans,size,size)))

def reshape(img):
    return torch.flatten(img)

def test(net,images,labels):
    # show images
    dispGrid_(images)
    #print labels
    print(' '.join('%5s' % classes[labels[j].item()] for j in range(batchSize)))

    #images=torch.stack([reshape(im) for im in images])
    out=net.forward(images)
    tots=out.exp().sum(1)
    vals,reps=torch.max(out,1)
    print([classes[i] for i in reps],vals.exp()/tots)

def load_net(path):
    net=NetCCFC(nClasses).to(device)
    net.load_state_dict(torch.load(path,map_location=device))
    net.eval()
    return net

def valid(net):
    acTot=0
    va_len=0
    for i,data in enumerate(va_loader):
        print(i)
        images,labels=data
        images=images.to(device)
        labels=labels.to(device)

        out=net.forward(images)
        reps=out.argmax(1)
        acTot+=(labels==reps).sum().item()
        va_len+=len(images)
    return acTot/va_len


def train(epochs,verbose=True,path="",lr=0.00003):
    print("Starting training")
    global device
    if path=="":
        net=NetCCFC(nClasses).to(device)
    else:
        net=load_net(path)

    criterion=nn.CrossEntropyLoss(reduction='sum')
    optimizer=optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    cuml=0
    accAvg=0
    deb=time()
    ttrain=0
    for e in range(epochs):
        for i,data in enumerate(tr_loader):
            images,labels=data
            images=images.to(device)
            labels=labels.to(device)

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
            if i%100==0:
                print(i,cuml/100,accAvg/100)
                cuml=0
                accAvg=0
        if e%10==0 or e==epochs-1:
            torch.save(net.state_dict(),"./net/"+str(e)+".pt")
            print(valid(net))
        if verbose:
            print("Epoch : "+str(e)+" Time : "+str(round(time()-deb,2)))
            print("Time spent training : "+str(round(ttrain,2)))

def eval_adv(net,attack,eps,seg):
    acTot=0
    lossTot=0
    criterion=nn.CrossEntropyLoss()
    for i,(images,labels) in enumerate(te_loader):
        if i==1:
            break
        print(i)
        images=images.to(device)
        labels=labels.to(device)
        if attack!="":
            images=FGSM(net,images,labels,10)
        if seg=="SLIC":
            images=images.cpu()
            images=batchSeg(images,8,seg_kMeans_2d)
            images=images.to(device)

        out=net.forward(images)
        reps=out.argmax(1)
        acTot+=(labels==reps).sum().item()
        lossTot+=criterion(out,labels)
    return (acTot/batchSize,lossTot*batchSize/len(te_loader.dataset))

def ev_grad():
    images.requires_grad_(True)

    out=net.forward(images)
    loss=criterion(out,labels)

    loss.backward(retain_graph=True)
    for i in range(len(images)):
        print(images.grad[i].norm())
    optimizer.zero_grad()

#train(200,path="")

net=load_net("./CIFARnoseg.pt")
#print(valid(net))
"""
net=fseg_NetCCFC(nClasses).to(device)
net.load_state_dict(torch.load("./CIFARseg512.pt",map_location=device))
net.eval()

images,labels=iter(va_loader).next()
images=images.to(device)
labels=labels.to(device)

img=images[0]

net.fcAvg=img2fc(img,256)
"""
g=[]
def vuln_fixedSeg(eps,net,attack,nsegs):
    images,labels=iter(va_loader).next()
    images=images.to(device)
    labels=labels.to(device)
    succTot=0
    n=0
    net=fseg_NetCCFC(nClasses).to(device)
    net.load_state_dict(torch.load("./CIFARseg"+str(nsegs)+".pt",map_location=device))
    net.eval()

    for i in range(len(images)):
        print(i)
        net.fcAvg=img2fc(images[i],nsegs)
        #net.nSegs=nsegs
        fmodel = fb.PyTorchModel(net, bounds=(-1, 1))
        _, advs, success = attack(fmodel, images[None,i], labels[None,i], epsilons=[eps])
        succTot+=success.sum().item()
        n+=1
        global g
        g+=advs[0]
    return succTot/n

deb=time()
print(vuln_fixedSeg(eps,net,getattr(fb.attacks,attack)(),nsegs))
print(time()-deb)
torch.save(torch.stack(g),"advs.pt")

"""
net=load_net("./CIFARseg512.pt")
print(valid(net))
images,labels=iter(va_loader).next()
images=images.to(device)
labels=labels.to(device)
dispGrid_(images)

#print(vuln(0.1,net))
def vuln(eps,net,attack):
    images,labels=iter(va_loader).next()
    images=images.to(device)
    labels=labels.to(device)
    succTot=0
    n=0

    for i in range(len(images)):
        print(i)
        fmodel = fb.PyTorchModel(net, bounds=(-1, 1))
        _, advs, success = attack(fmodel, images[None,i], labels[None,i], epsilons=[eps])
        succTot+=success.sum().item()
        n+=1
    return succTot/n
"""