from PIL import Image
import numpy as np
import cv2
import os

img = cv2.imread(r'Predict\ce04.png') # 填要转换的图片存储地址
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(r'Predict\4.png',img) # 填转换后的图片存储地址，若在同一目录，则注意不要重名