'''
  File name: corner_detector.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''
import cv2
import numpy as np

def corner_detector(img):
    cimg = cv2.cornerHarris(img,2,3,0.04)
    cimg[cimg<0.01*cimg.max()]=0
    res=np.zeros((cimg.shape))
    res[20:cimg.shape[0]-20,20:cimg.shape[1]-20] = cimg[20:cimg.shape[0]-20,20:cimg.shape[1]-20]
    return res