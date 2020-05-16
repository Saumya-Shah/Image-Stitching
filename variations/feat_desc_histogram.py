import numpy as np
import cv2

# -----------------------------------
# Using Histogram
# -----------------------------------

def feat_desc(img, x, y):
    Mag, Magx, Magy, Ori = findDerivatives(img)
    Ori = Ori+np.pi
    Ori[np.logical_and(Ori>=0,Ori<np.pi/4)] = 0
    Ori[np.logical_and(Ori>=np.pi/4,Ori<np.pi/2)]=1
    Ori[np.logical_and(Ori>=np.pi/2,Ori<3*np.pi/4)]=2
    Ori[np.logical_and(Ori>=3*np.pi/4,Ori<np.pi)]=3
    Ori[np.logical_and(Ori>=np.pi,Ori<5*np.pi/4)]=4
    Ori[np.logical_and(Ori>=5*np.pi/4,Ori<3*np.pi/2)]=5
    Ori[np.logical_and(Ori>=3*np.pi/2,Ori<7*np.pi/4)]=6
    Ori[np.logical_and(Ori>=7*np.pi/4,Ori<3*np.pi/4)]=7
    img =Ori
    descs=np.zeros((64*8,len(x)))
    padImage = np.zeros((img.shape[0]+40,img.shape[1]+40))
    padImage[20:img.shape[0]+20,20:img.shape[1]+20] = img
    k = 0
    for (_x,_y) in zip(x,y):
        _x = int(_x)
        _y = int(_y)
        patch = padImage[_y:_y+40,_x:_x+40]
        blurredPatch = cv2.GaussianBlur(patch,(5,5),1)
        desc = []
        for i in range(0,40,5):
            for j in range(0,40,5):
                smallPatch = patch[i:i+5,j:j+5]
                desc.append(np.where(smallPatch==0)[0].shape[0])
                desc.append(np.where(smallPatch==1)[0].shape[0])
                desc.append(np.where(smallPatch==2)[0].shape[0])
                desc.append(np.where(smallPatch==3)[0].shape[0])
                desc.append(np.where(smallPatch==4)[0].shape[0])
                desc.append(np.where(smallPatch==5)[0].shape[0])
                desc.append(np.where(smallPatch==6)[0].shape[0])
                desc.append(np.where(smallPatch==7)[0].shape[0])
                # print(maxValue)
                # print(smallPatch.shape)
        desc = np.array(desc)
        desc = (desc - desc.mean())/desc.std()
        descs[:,k]=desc
        k=k+1
        # print(descs.shape)
    return descs
