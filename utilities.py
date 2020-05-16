from feat_desc import *
import cv2
import numpy as np
from corner_detector import *
from anms import *
from feat_match import *
from ransac_est_homography import *
from matplotlib import pyplot as plt

def interp2(v, xq, yq):
    if len(xq.shape) == 2 or len(yq.shape) == 2:
        dim_input = 2
        q_h = xq.shape[0]
        q_w = xq.shape[1]
        xq = xq.flatten()
        yq = yq.flatten()

    h = v.shape[0]
    w = v.shape[1]
    if xq.shape != yq.shape:
        raise 'query coordinates Xq Yq should have same shape'

    x_floor = np.floor(xq).astype(np.int32)
    y_floor = np.floor(yq).astype(np.int32)
    x_ceil = np.ceil(xq).astype(np.int32)
    y_ceil = np.ceil(yq).astype(np.int32)

    x_floor[x_floor < 0] = 0
    y_floor[y_floor < 0] = 0
    x_ceil[x_ceil < 0] = 0
    y_ceil[y_ceil < 0] = 0

    x_floor[x_floor >= w-1] = w-1
    y_floor[y_floor >= h-1] = h-1
    x_ceil[x_ceil >= w-1] = w-1
    y_ceil[y_ceil >= h-1] = h-1

    v1 = v[y_floor, x_floor]
    v2 = v[y_floor, x_ceil]
    v3 = v[y_ceil, x_floor]
    v4 = v[y_ceil, x_ceil]

    lh = yq - y_floor
    lw = xq - x_floor
    hh = 1 - lh
    hw = 1 - lw

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw

    interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

    if dim_input == 2:
        return interp_val.reshape(q_h, q_w)
    return interp_val


def warp_image(img1,H,shape0, shape1):
    x_r = np.array(range(shape0))
    y_r = np.array(range(shape1))

    x_co1,y_co1 = np.meshgrid(x_r,y_r)
    x_co = x_co1.flatten()
    y_co = y_co1.flatten()
    ones = np.ones(x_co.shape)
    Cord = np.vstack([x_co,y_co,ones])
    CordOg = np.matmul(np.linalg.inv(H),Cord)
    CordOg = CordOg/CordOg[2,:]
    x_og = CordOg[0,:].reshape(x_co1.shape)
    y_og = CordOg[1,:].reshape(y_co1.shape)
    im1BT = interp2(img1[:,:,0],x_og,y_og)
    im1GT = interp2(img1[:,:,1],x_og,y_og)
    im1RT = interp2(img1[:,:,2],x_og,y_og)

    outliersX1 = np.argwhere(x_og>img1.shape[1])
    outliersX2 = np.argwhere(x_og<0)
    outliersY1 = np.argwhere(y_og>img1.shape[0])
    outliersY2 = np.argwhere(y_og<0)
    outliers = np.vstack([outliersX1,outliersX2,outliersY1,outliersY2])
    im1BT[outliers[:,0],outliers[:,1]]=0
    im1GT[outliers[:,0],outliers[:,1]]=0
    im1RT[outliers[:,0],outliers[:,1]]=0

    img1T = np.zeros([shape1,shape0,3])
    img1T[:,:,0] = im1BT
    img1T[:,:,1] = im1GT
    img1T[:,:,2] = im1RT

    return img1T

# Finds a homography to transform img1 -> img2
def get_homography(img1, img2, createPlots=True, imgNum=0):
    imgName = "img"+str(imgNum)
    max_anms = 2000

    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    c = corner_detector(gray)
    
    if createPlots:
        imgCopy = img1.copy()
        cCopy = c.copy()
        cCopy = cv2.dilate(cCopy, None)
        imgCopy[c>0] = [0,0,255]
        plt.imshow(cv2.cvtColor(imgCopy, cv2.COLOR_BGR2RGB))
        fig = plt.gcf()
        fig.savefig(imgName+'corner.png',dpi=200)
        plt.show()

    X1,Y1,rmax=anms(c, max_anms)
    d1 = feat_desc(gray,X1,Y1)
    kp1=[]
    for (_x,_y) in zip(X1,Y1):
        kp1.append(cv2.KeyPoint(_x,_y,40))

    gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    c = corner_detector(gray)
    X2,Y2,rmax = anms(c, max_anms)
    d2 = feat_desc(gray,X2,Y2)
    kp2=[]
    for (_x,_y) in zip(X2,Y2):
        kp2.append(cv2.KeyPoint(_x,_y,40))
    m,dMatch=feat_match(d1, d2)
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    for k,idx in enumerate(m):
        if (idx != -1):
            x1.append(X1[k])
            y1.append(Y1[k])
            x2.append(X2[idx])
            y2.append(Y2[idx])
    x1=np.array(x1)
    x2=np.array(x2)
    y1=np.array(y1)
    y2=np.array(y2)
    print(x1.shape)
    H, inlier_ind=ransac_est_homography(x1,y1,x2,y2,2)
    if createPlots:
        mask = np.array(inlier_ind, dtype=bool)
        mfilter = []
        for idx,i in enumerate(mask):
            if i == True:
                mfilter.append(dMatch[idx])
        img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
        print(len(mfilter))
        f=cv2.drawMatches(img1, kp1, img2, kp2, mfilter, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        fig = plt.gcf()
        fig.savefig(imgName+'inliners.png',dpi=200)
        plt.show()
        f=cv2.drawMatches(img1, kp1, img2, kp2, dMatch, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        fig = plt.gcf()
        fig.savefig(imgName+'outliers.png',dpi=200)
        plt.show()

        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        fig = plt.gcf()
        ax1 = fig.add_subplot(111)
        ax1.scatter(X1,Y1,c='r',s=1,label='ANMS')
        ax1.scatter(x1[mask],y1[mask],c='b',s=1,label='RANSAC')
        fig.savefig(imgName+'ransac.png',dpi=200)
        plt.show()
    return H
