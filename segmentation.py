from math import sqrt
from skimage import color
from skimage.segmentation import slic
import torch
import numpy as np

parFile=open("params.txt")
params=list(filter(lambda s: len(s)>0 and s[0]!='#',parFile.read().splitlines()))

size=int(params[1])

def LAB(image):
    image=image/2+0.5
    image=np.transpose(image,(1,2,0))
    image=color.rgb2lab(image)
    return np.transpose(image,(2,0,1))

def avg_seg(image,segs):
    image=np.transpose(image.numpy(),(1,2,0))
    sums={}
    ns={}
    for x in range(size):
        for y in range(size):
            zone=segs[x][y]
            if not zone in sums:
                sums[zone]=np.zeros_like(image[0][0])
                ns[zone]=0
            sums[zone]+=image[x][y]
            ns[zone]+=1
    for x in range(size):
        for y in range(size):
            zone=segs[x][y]
            image[x][y]=sums[zone]/ns[zone]
    return torch.tensor(np.transpose(image,(2,0,1)))

def batchSeg(images,n,seg):
    tab=[avg_seg(img.cpu(),seg(img.cpu(),n)) for img in images]
    return torch.stack(tab)

def seg_quant(image,n):
    return ((image*n+0.5)//1)/n

def seg_SLIC(image,n):
    im=image.numpy().astype('double').transpose(1,2,0)
    return slic(im,n_segments=n)

def seg_rect(image,n):
    size=image.shape[1]
    sqSize=size//int(sqrt(n))
    rep=torch.zeros(size,size)
    for i in range(size):
        for j in range(size):
            rep[i][j]=i//sqSize+j//sqSize*int(sqrt(n))
    return rep
