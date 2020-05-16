from matplotlib import pyplot as plt
import cv2
import numpy as np
import utilities
from mymosaic import *
import sys

# more sets of images available in the images/ folder!
imgL = cv2.imread('images/shoemaker-left.jpg')
imgM = cv2.imread('images/shoemaker-middle.jpg')
imgR = cv2.imread('images/shoemaker-right.jpg')

# find left -> middle homography and right->middle homography
HLM = utilities.get_homography(imgL,imgM,False,"L")
HRM = utilities.get_homography(imgR,imgM,False,"R")

canvas = mymosaic(imgL,imgM,imgR,HLM,HRM)
if cv2.imwrite("output.png",canvas):
    print("Output saved to output.png")

