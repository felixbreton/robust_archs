# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader, TensorDataset

from segmentation import *

import torchvision.transforms as transforms
import torchvision.datasets as datasets


def IMGNET12(root='~/datasets/imgnet12/', bs=32, bs_test=None, num_workers=32,
             valid_size=.1, size=256, crop=False, normalize=False,download=True):

    # Datafolder '~/datasets/imgnet12/' should contain folders train/ and val/,
    # each of which whould contain 12 subfolders (1 per class) with .jpg files

    root = os.path.expanduser(root)

    # original means = [.485, .456, .406]
    # original stds = [0.229, 0.224, 0.225]

    means = [.453, .443, .403]
    stds = {
        256: [.232, .226, .225],
        128: [.225, .218, .218],
        64: [.218, .211, .211],
        32: [.206, .200, .200]
    }

    if normalize:
        normalize = transforms.Normalize(mean=means,
                                         std=stds[size])
    else:
        normalize = transforms.Normalize((0., 0., 0),
                                         (1., 1., 1.))

    if bs_test is None:
        bs_test = bs

    if crop:
        tr_downsamplingOp = transforms.RandomCrop(size)
        te_downsamplingOp = transforms.CenterCrop(size)
    else:
        tr_downsamplingOp = transforms.Resize(size)
        te_downsamplingOp = transforms.Resize(size)

    preprocess = [transforms.Resize(256), transforms.CenterCrop(256)]

    tr_transforms = transforms.Compose([
        *preprocess,
        tr_downsamplingOp,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize, ])

    te_transforms = transforms.Compose([
        *preprocess,
        te_downsamplingOp,
        transforms.ToTensor(),
        normalize, ])

    tr_dataset = datasets.ImageFolder(root + '/train', transform=tr_transforms)
    te_dataset = datasets.ImageFolder(root + '/val', transform=te_transforms)

    # Split training in train and valid set
    num_train = len(tr_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.seed(42)
    np.random.shuffle(indices)
    tr_idx, va_idx = indices[split:], indices[:split]

    tr_sampler = SubsetRandomSampler(tr_idx)
    va_sampler = SubsetRandomSampler(va_idx)

    tr_loader = torch.utils.data.DataLoader(
        tr_dataset, batch_size=bs,
        num_workers=num_workers, pin_memory=True, sampler=tr_sampler)

    va_loader = torch.utils.data.DataLoader(
        tr_dataset, batch_size=bs_test,
        num_workers=num_workers, pin_memory=True, sampler=va_sampler)

    te_loader = torch.utils.data.DataLoader(
        te_dataset, batch_size=bs_test, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    if valid_size > 0.:
        return tr_loader, va_loader, te_loader
    else:
        return tr_loader, te_loader


def CIFAR10(root='~/datasets/cifar10/', bs=128, bs_test=None,
            augment_training=True, valid_size=0., size=32, num_workers=1,
            normalize=False):
    root = os.path.expanduser(root)

    if bs_test is None:
        bs_test = bs

    if normalize:
        #normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                                 (0.2023, 0.1994, 0.2010))
        normalize = transforms.Normalize((0.5, 0.5, 0.5),
                                         (0.5, 0.5, 0.5))
    else:
        normalize = transforms.Normalize((0., 0., 0),
                                         (1., 1., 1.))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size, Image.NEAREST),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size, Image.NEAREST),
        transforms.ToTensor(),
        normalize
    ])

    transform_valid = transform_test

    if augment_training is False:
        transform_train = transform_test

    dataset_tr = datasets.CIFAR10(root=root,
                                  train=True,
                                  transform=transform_train,download=True)

    dataset_va = datasets.CIFAR10(root=root,
                                  train=True,
                                  transform=transform_valid,download=True)

    dataset_te = datasets.CIFAR10(root=root,
                                  train=False,
                                  transform=transform_test,download=True)

    # Split training in train and valid set
    num_train = len(dataset_tr)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    loader_tr = torch.utils.data.DataLoader(dataset_tr,
                                            batch_size=bs,
                                            sampler=train_sampler,
                                            num_workers=num_workers)

    loader_va = torch.utils.data.DataLoader(dataset_va,
                                            batch_size=bs,
                                            sampler=valid_sampler,
                                            num_workers=num_workers)

    # add pin_memory
    loader_te = torch.utils.data.DataLoader(dataset_te,
                                            batch_size=bs_test,
                                            shuffle=False,
                                            num_workers=num_workers)
    if valid_size > 0:
        return loader_tr, loader_va, loader_te
    else:
        return loader_tr, loader_te


def MNIST(root='~/datasets/mnist/', bs=128, bs_test=None,
          augment_training=True, valid_size=0., size=32, num_workers=1,
          normalize=False):
    root = os.path.expanduser(root)

    if bs_test is None:
        bs_test = bs

    if normalize:
        normalize = transforms.Normalize((0.1307,), (0.3081,))
    else:
        normalize = transforms.Normalize((0.,), (1.,))

    transform = transforms.Compose([
        transforms.Resize(32, Image.BILINEAR),
        transforms.Resize(size, Image.NEAREST),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        normalize
    ])

    dataset_tr = datasets.MNIST(root=root,
                                train=True,
                                transform=transform,download=True)

    dataset_va = datasets.MNIST(root=root,
                                train=True,
                                transform=transform)

    dataset_te = datasets.MNIST(root=root,
                                train=False,
                                transform=transform)

    # Split training in train and valid set
    num_train = len(dataset_tr)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    loader_tr = torch.utils.data.DataLoader(dataset_tr,
                                            batch_size=bs,
                                            sampler=train_sampler,
                                            num_workers=num_workers)

    loader_va = torch.utils.data.DataLoader(dataset_va,
                                            batch_size=bs,
                                            sampler=valid_sampler,
                                            num_workers=num_workers)

    # add pin_memory
    loader_te = torch.utils.data.DataLoader(dataset_te,
                                            batch_size=bs_test,
                                            shuffle=False,
                                            num_workers=num_workers)
    if valid_size > 0:
        return loader_tr, loader_va, loader_te
    else:
        return loader_tr, loader_te


def SQUARE_CIRCLE(bs=128, valid_size=.1, size=32,
                  normalize=False, length=30000):

    tr_dataset = SquareCircle(
        img_size=size, length=length, normalize=normalize)
    va_dataset = SquareCircle(
        img_size=size, length=.1*length, normalize=normalize)
    te_dataset = SquareCircle(
        img_size=size, length=length//6, normalize=normalize)

    tr_loader = DataLoader(tr_dataset, batch_size=bs)
    va_loader = DataLoader(va_dataset, batch_size=bs)
    te_loader = DataLoader(te_dataset, batch_size=bs)

    return tr_loader, va_loader, te_loader


class SquareCircle(Dataset):
    def __init__(self, img_size=32, min_hsize=.1, max_hsize=.3,
                 length=10000, normalize=False):
        self.img_size = img_size
        self.min_hsize = min_hsize
        self.max_hsize = max_hsize
        self.length = int(length)
        self.normalize = normalize

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sc = random.randint(0, 1)
        i = random.uniform(0, 1)
        j = random.uniform(0, 1)
        hsize = random.uniform(self.min_hsize, self.max_hsize)
        color_ob = random.uniform(0., 1.)
        color_bg = random.uniform(0., 1.)

        if self.normalize:
            color_ob = (color_ob - .5) * 12
            color_bg = (color_bg - .5) * 12
        else:
            color_ob = (color_ob - .5) * 2
            color_bg = (color_bg - .5) * 2

        return (self.generate_img(sc, i, j, hsize, color_ob, color_bg, self.img_size), sc)

    def generate_img(self, sc, i, j, hsize, color_ob, color_bg, img_size=32):
        '''
        sc: int 0 if square, 1 if circle
        i: float [0,1) center i coordinate
        j: float [0,1) center j coordinate
        hsize: half size of the shape (radius/length) -> (0, .5)
        color_ob: float [0,1) grey scale color of object
        color_bg: float [0,1) grey scale color of back-ground
        '''

        i = int(i * img_size)
        j = int(j * img_size)
        hsize = hsize * img_size

        img = torch.ones(img_size, img_size) * color_bg

        if sc == 0:  # square
            im = max(0, int(i - hsize))
            iM = min(img_size - 1, int(i + hsize))
            jm = max(0, int(j - hsize))
            jM = min(img_size - 1, int(j + hsize))

            img[im:iM, jm:jM] = color_ob

        elif sc == 1:  # circle
            for k in range(img_size):
                for h in range(img_size):
                    r2 = (k - i)**2 + (h - j)**2
                    if r2 <= hsize**2:
                        img[k, h] = color_ob

        return img.unsqueeze(0)

#Generates a dataset by applying a transform to every image in the input data
def gen_dataset(loader,file):
    out=torch.Tensor()
    labs=torch.LongTensor()
    for i,data in enumerate(loader):
        print(i)
        images,labels=data
        out=torch.cat((out,batchSeg(images,512,seg_SLIC)))#transform applied to the data
        labs=torch.cat((labs,labels))
    torch.save((out,labs),file)

#Loads dataset from files generated by gen_dataset()
def FROM_FILE(path, bs=128, augment_training=True):
    normalize = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                                        
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size, Image.NEAREST),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size, Image.NEAREST),
        transforms.ToTensor(),
        normalize
    ])

    transform_valid = transform_test

    if augment_training is False:
        transform_train = transform_test
        
    tr_dataset = FileDataset(path+"tr.pt",transform_train)
    va_dataset = FileDataset(path+"va.pt",transform_valid)
    te_dataset = FileDataset(path+"te.pt",transform_test)

    tr_loader = DataLoader(tr_dataset, batch_size=bs)
    va_loader = DataLoader(va_dataset, batch_size=bs)
    te_loader = DataLoader(te_dataset, batch_size=bs)

    return tr_loader, va_loader, te_loader

class FileDataset(Dataset):
    def __init__(self,path,transform):
        self.images,self.labels=torch.load(path)
        self.transform=transform
    
    def __getitem__(self,idx):
        return self.transform((self.images[idx]+1.0)/2.0),self.labels[idx]
    
    def __len__(self):
        return len(self.images)
