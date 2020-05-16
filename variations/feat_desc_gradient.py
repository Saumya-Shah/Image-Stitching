import numpy as np
import cv2

def findDerivatives(I_gray):
    G = 1/159.0*np.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]])
    dx,dy = np.gradient(G, axis = (1,0))
    Magx = signal.convolve2d(I_gray, dx, 'same')
    Magy = signal.convolve2d(I_gray, dy, 'same')
    Mag = np.sqrt(Magx*Magx + Magy*Magy)

    Ori = np.arctan2(Magy, Magx)
    return (Mag, Magx, Magy, Ori)

def feat_desc(img, x, y):
    Mag, Magx, Magy, Ori = findDerivatives(img)
    # img = Mag
    descs=np.zeros((64,len(x)))
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
                maxValue=np.mean(smallPatch.flatten())
                desc.append(maxValue)
                # print(maxValue)
                # print(smallPatch.shape)
        desc = np.array(desc)
        desc = (desc - desc.mean())/desc.std()
        descs[:,k]=desc
        k=k+1
    return descs
