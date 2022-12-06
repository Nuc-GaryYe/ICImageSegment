import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

'''
首先通过调用np.zeros()函数创建掩码、fgbModel和bgModel，然后定义rect矩形范围，雕塑grabCut()函数实现图像分割   
由于该方法会修改掩码，像素会被标记为不同的标志来知名它们是背景或前景。最后将所有的0像素点和2像素点赋值为0(背景)，而
所有的1像素点和3像素点赋值为1(前景)
'''

img = cv2.imread("../data/testData/colarDemo.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#灰度图
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#设置掩码、fgbModel、bgModel
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

#矩形坐标
rect=(100, 100, 500, 800)

#图像分割
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

#设置新掩码:0和2做背景
mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

#设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

#显示原图
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1)
plt.imshow(img)
plt.title(u'(a)原始图像')

plt.xticks([])
plt.yticks([])

#使用蒙版来获取前景图像
img = img*mask2[:,:,np.newaxis]
plt.subplot(1,2,2)
plt.imshow(img)
plt.title(u'(b)目标图像')
plt.colorbar()
plt.xticks([])
plt.yticks([])

plt.show()