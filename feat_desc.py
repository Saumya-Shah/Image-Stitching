import numpy as np
import cv2
'''
  File name: feat_desc.py
  Author:
  Date created:
'''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature,
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40
    window to have a nice big blurred descriptor.
    - Input img: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
'''



def feat_desc(img, x, y):
    descs=np.zeros((64,len(x)))
    padImage = np.zeros((img.shape[0]+40,img.shape[1]+40))
    padImage[20:img.shape[0]+20,20:img.shape[1]+20] = img
    k = 0
    for (_x,_y) in zip(x,y):
        _x = int(_x)
        _y = int(_y)
        patch = padImage[_y:_y+40,_x:_x+40]
        desc = []
        for i in range(0,40,5):
            for j in range(0,40,5):
                smallPatch = patch[i:i+5,j:j+5]
                maxValue=np.max(smallPatch.flatten())
                desc.append(maxValue)
        desc = np.array(desc)
        desc = (desc - desc.mean())/desc.std()
        descs[:,k]=desc
        k=k+1
    return descs


