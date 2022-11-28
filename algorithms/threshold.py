import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
五种效果
THRESH_BINARY：二进制阈值化
THRESH_BINARY_INV：反二进制阈值化
THRESH_TRUNC：截断阈值化
THRESH_TOZERO：阈值化为0
THRESH_TOZERO_INV：反阈值化为0
"""

img = cv2.imread("../data/dataset/srcImage/{}.jpg".format(str(1)))
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

ret, thresh1 = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(grayImage, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(grayImage, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(grayImage, 127, 255, cv2.THRESH_TOZERO_INV)

titles=['(a)Gray Image', '(b)BINARY', '(c)THRESH_BINARY_INV', '(d)THRESH_TRUNC'
    , '(e)THRESH_TOZERO', '(f)THRESH_TOZERO_INV']

images = [grayImage, thresh1, thresh2, thresh3,thresh4, thresh5]
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
