'''
  File name: anms.py
  Author:
  Date created:
'''

'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed 
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''
import numpy as np
def anms(cimg, max_pts):
    keypt = np.asarray(np.nonzero(cimg)).T
    N = min(max_pts,keypt.shape[0])
    keyR = cimg[np.nonzero(cimg)]
    keypts = np.zeros([keypt.shape[0],3])
    keypts[:,:2]=keypt
    keypts[:,2]= keyR
    distance = np.zeros(keypts.shape)
    distance[:,:2]=keypts[:,:2]
    maxdist = np.sqrt(cimg.shape[0]**2 + cimg.shape[1]**2)

    for i in range(keypts.shape[0]):
        condind = np.argwhere(np.logical_and(keyR[:]>keyR[i],keyR[:]<1.4*keyR[i])==1)
        condind = condind[:,0]
        condpts = keypts[condind,:]
        if (condpts.shape[0]!=0):
            dist = np.sqrt((condpts[:,0]-keypts[i,0])**2 + (condpts[:,1]-keypts[i,1])**2)
            minD = dist.min()
            distance[i,2]=minD
        else:
            distance[i,2] = maxdist

    distanceSorted = distance[distance[:,2].argsort(kind='mergesort')]
    distanceSorted = np.flip(distanceSorted,axis=0)
    topN = distanceSorted[:N,:2]
    rmax = distanceSorted[N-1,2]
    y = topN[:,0]
    x = topN[:,1]
    return x, y, rmax
