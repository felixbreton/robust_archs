from math import sqrt
from skimage import color

def LAB(image):
    image=image/2+0.5
    image=np.transpose(image,(1,2,0))
    image=color.rgb2lab(image)
    return np.transpose(image,(2,0,1))

def avg_seg(image,segs):
    image=np.transpose(image,(1,2,0))
    sums={}
    ns={}
    for x in range(size):
        for y in range(size):
            zone=segs[x][y].item()
            if not zone in sums:
                sums[zone]=torch.zeros_like(image[0][0])
                ns[zone]=0
            sums[zone]+=image[x][y]
            ns[zone]+=1
    for x in range(size):
        for y in range(size):
            zone=segs[x][y].item()
            image[x][y]=sums[zone]/ns[zone]
    return np.transpose(image,(2,0,1))

def seg_quant(image,n,w2d):
    return ((image*n+0.5)//1)/n

def seg_kMeans_2d(image,n,w2d):
    centers=torch.rand(n,5)*2-1
    st=10
    for i in range(st):
        sums=torch.zeros(n,5)
        ns=torch.zeros(n)
        for x in range(size):
            for y in range(size):
                lab=(torch.tensor([image[c][x][y] for c in range(len(image))]))
                xy=torch.tensor([2*x/size-1,2*y/size-1])*w2d
                col=torch.cat((lab,xy))
                opt=50
                iOpt=-1
                for e in range(n):
                    dist=torch.norm(col-centers[e])
                    if dist<opt:
                        opt=dist
                        iOpt=e
                sums[iOpt]+=col
                ns[iOpt]+=1
        for e in range(n):
            if ns[e]==0:
                centers[e]=torch.rand(5)*2-1
            else:
                centers[e]=sums[e]/ns[e]

    out=torch.zeros((size,size))
    for x in range(size):
        for y in range(size):
            lab=(torch.tensor([image[c][x][y] for c in range(len(image))]))
            xy=torch.tensor([2*x/size-1,2*y/size-1])*w2d
            col=torch.cat((lab,xy))
            opt=50
            iOpt=-1
            for e in range(n):
                dist=torch.norm(col-centers[e])
                if dist<opt:
                    opt=dist
                    iOpt=e
            out[x][y]=iOpt
    return out

def delta(image):#size*size*chans format
    out=torch.zeros((size,size))
    for x in range(1,size-1):
        for y in range(1,size-1):
            nx=torch.norm(image[x-1][y]-image[x+1][y])**2
            ny=torch.norm(image[x][y-1]-image[x][y+1])**2
            out[x][y]=nx+ny
    return out

def seg_SLIC(image,n,w2d):
    """image=LAB(image)/50
    image=torch.tensor(image)"""
    c=round(sqrt(n))
    n=c**2
    centers=torch.zeros(n,5)
    gr=delta(np.transpose(image,(1,2,0)))
    for i in range(c):
        for j in range(c):
            x=round((i+0.5)*size/c)
            y=round((j+0.5)*size/c)
            minDelta=100
            minDx=0
            minDy=0
            for dx in range(-1,2):
                for dy in range(-1,2):
                    if gr[x+dx][y+dy]<minDelta:
                        minDelta=gr[x+dx][y+dy]
                        minDx=dx
                        minDy=dy
            x+=minDx
            y+=minDy
            centers[i*c+j]=torch.cat((torch.tensor([image[chan][x][y] for chan in range(len(image))]),torch.tensor([2*x/size-1,2*y/size-1])*w2d))

    st=10
    for i in range(st):
        sums=torch.zeros(n,5)
        ns=torch.zeros(n)
        for x in range(size):
            for y in range(size):
                lab=(torch.tensor([image[c][x][y] for c in range(len(image))]))
                xy=torch.tensor([2*x/size-1,2*y/size-1])*w2d
                col=torch.cat((lab,xy))
                opt=50
                iOpt=-1
                for e in range(n):
                    dist=torch.norm(col-centers[e])
                    if dist<opt:
                        opt=dist
                        iOpt=e
                sums[iOpt]+=col
                ns[iOpt]+=1
        for e in range(n):
            if ns[e]==0:
                centers[e]=torch.rand(5)*2-1
            else:
                centers[e]=sums[e]/ns[e]

    out=torch.zeros((size,size))
    for x in range(size):
        for y in range(size):
            lab=(torch.tensor([image[c][x][y] for c in range(len(image))]))
            xy=torch.tensor([2*x/size-1,2*y/size-1])*w2d
            col=torch.cat((lab,xy))
            opt=50
            iOpt=-1
            for e in range(n):
                dist=torch.norm(col-centers[e])
                if dist<opt:
                    opt=dist
                    iOpt=e
            out[x][y]=iOpt
    return out