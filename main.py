from time import time
from shutil import copyfile
import sys
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument('--nsegs', type=int)
parser.add_argument('--eps', type=float)
parser.add_argument('--attack', type=str)
args = parser.parse_args()
nsegs = args.nsegs
eps = args.eps
attack = args.attack
if nsegs == None:
    nsegs = 64
if eps == None:
    eps = 0.1
if attack == None:
    attack = "L2PGD"
#parameters that (currently) must be hardcoded :
#number of tests
#whether to attack the segmenter
nEpochs = 200
fixedSeg=True

parFile=open("params.txt")
params=list(filter(lambda s: len(s)>0 and s[0]!='#', parFile.read().splitlines()))

if params[0].lower() == 'true' and torch.cuda.is_available():
    device = torch.device("cuda:0")
    workers = 8
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    workers = 0
    print("Running on the CPU")

size = int(params[1])
sfn = size//4-3
chans = nChans(params[2])
batchSize = int(params[3])

t0 = time()

if params[2] == "CIFAR":
    nClasses = 10
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')
    tr_loader, va_loader, te_loader = CIFAR10(
        bs=batchSize, valid_size=.1,
        size=size, normalize=True, num_workers=workers)
elif params[2] == "SC":
    nClasses = 2
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
    imshow(torchvision.utils.make_grid(images.reshape(batchSize, chans, size, size)))

def reshape(img):
    return torch.flatten(img)

def test(net, images, labels):
    # show images
    dispGrid_(images)
    #print labels
    print(' '.join('%5s' % classes[labels[j].item()] for j in range(batchSize)))

    #images=torch.stack([reshape(im) for im in images])
    out = net.forward(images)
    tots = out.exp().sum(1)
    vals, reps = torch.max(out, 1)
    print([classes[i] for i in reps], vals.exp()/tots)

def load_net(path):
    net = NetCCFC(nClasses).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net

def valid(net):
    acTot = 0
    va_len = 0
    for i, data in enumerate(va_loader):
        print(i)
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        out = net.forward(images)
        reps = out.argmax(1)
        acTot += (labels == reps).sum().item()
        va_len += len(images)
    return acTot/va_len


def train(epochs, verbose=True, path="", lr=0.00003):
    tr_loader, va_loader, te_loader = FROM_FILE('./data/'+params[2]+str(nsegs), bs=batchSize)
    print("Starting training")
    global device
    if path == "":
        net = NetCCFC(nClasses).to(device)
    else:
        net = load_net(path)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    cuml = 0
    accAvg = 0
    start = time()
    ttrain = 0
    for e in range(epochs):
        for i, data in enumerate(tr_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            dtrain = time()
            out = net.forward(images)
            loss = criterion(out, labels)
            reps = out.argmax(1)
            acc = ((labels == reps).sum()).float()/len(images)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ttrain += time()-dtrain

            cuml += loss
            accAvg += acc
            if i%100 == 0:
                print(i, cuml/100, accAvg/100)
                cuml = 0
                accAvg = 0
        if e%10 == 0:
            torch.save(net.state_dict(), "./net/"+str(e)+".pt")
            print(valid(net))
        if e == epochs-1:
            torch.save(net.state_dict(), "./net/last.pt")
            print(valid(net))
        if verbose:
            print("Epoch : "+str(e)+" Time : "+str(round(time()-start, 2)))
            print("Time spent training : "+str(round(ttrain, 2)))

def train_PGD(epochs, eps, verbose=True, path="", lr=0.00003):
    print("Starting training")
    global device
    if path == "":
        net = NetCCFC(nClasses).to(device)
    else:
        net = load_net(path)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    PGD = fb.attacks.L2PGD()

    cuml = 0
    accAvg = 0
    start = time()
    ttrain = 0
    for e in range(epochs):
        for i, data in enumerate(tr_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            dtrain = time()
            net.eval()
            fmodel = fb.PyTorchModel(net, bounds=(-1, 1))
            _, advs, _ = PGD(fmodel, images, labels, epsilons=[eps])
            net.train()


            out = net.forward(advs[0])
            loss = criterion(out, labels)
            reps = out.argmax(1)
            acc = ((labels == reps).sum()).float()/len(images)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ttrain += time()-dtrain

            cuml += loss
            accAvg += acc
            if i%100 == 0:
                print(i, cuml/100, accAvg/100,ttrain)
                cuml = 0
                accAvg = 0
        if e%10 == 0:
            torch.save(net.state_dict(), "./net/"+str(e)+".pt")
            print(valid(net))
        if e == epochs-1:
            torch.save(net.state_dict(), "./net/last.pt")
            print(valid(net))
        if verbose:
            print("Epoch : "+str(e)+" Time : "+str(round(time()-start, 2)))
            print("Time spent training : "+str(round(ttrain, 2)))

def vuln_seg(eps, path, attack, nsegs, fixedSeg):
    succTot = 0
    n = 0
    advExamples = []
    for batch in range(1):
        print(batch)
        images, labels = iter(te_loader).next()
        images = images.to(device)
        labels = labels.to(device)

        if fixedSeg:
            net = fseg_NetCCFC(nClasses).to(device)
        else:
            net = seg_NetCCFC(nClasses).to(device)
        net.load_state_dict(torch.load(path, map_location=device))
        net.eval()

        for i in range(len(images)):
            #print(i)
            if fixedSeg:
                net.fcAvg = img2fc(images[i], nsegs)
            else:
                net.nSegs = nsegs
            fmodel = fb.PyTorchModel(net, bounds=(-1, 1))
            _, advs, success = attack(fmodel, images[None, i], labels[None, i], epsilons=[eps])
            succTot += success.sum().item()
            n += 1
            advExamples += advs[0]
        torch.save(torch.stack(advExamples), "advs.pt")
    return succTot/n

def vuln(eps, path, attack):
    succTot = 0
    n = 0
    advExamples = []
    for batch in range(10):
        print(batch)
        images, labels = iter(te_loader).next()
        images = images.to(device)
        labels = labels.to(device)

        net = NetCCFC(nClasses).to(device)
        net.load_state_dict(torch.load(path, map_location=device))
        net.eval()

        fmodel = fb.PyTorchModel(net, bounds=(-1, 1))
        _, advs, success = attack(fmodel, images, labels, epsilons=[eps])
        succTot += success.sum().item()
        n += len(images)
        advExamples += advs[0]
        torch.save(torch.stack(advExamples), "advs.pt")
    return succTot/n

train_PGD(200,0.1)

""""modelPath = params[2]+str(nsegs)+".pt"

try:
    net = load_net(modelPath)
except:
    print("Model '"+modelPath+"' not found, starting training.")
    train(nEpochs, path="")
    copyfile("./net/last.pt",modelPath)
    net = load_net(modelPath)

start = time()
print(1-vuln_seg(eps, modelPath, getattr(fb.attacks, attack)(), nsegs, fixedSeg))
print(time()-start)
"""