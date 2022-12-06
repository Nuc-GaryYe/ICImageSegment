import cv2
import numpy as np
import matplotlib.pyplot as plt

#读图
img = cv2.imread("../data/testData/colarDemo.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#灰度图
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#阈值处理
ret, binary = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

#Roberts算子
kernelx = np.array([[-1,0],[0,1]], dtype = int)
kernely = np.array([[0,-1],[1,0]], dtype = int)
x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
y = cv2.filter2D(binary, cv2.CV_16S, kernely)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5,0)

#Prewitt算子
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype = int)
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype = int)
x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
y = cv2.filter2D(binary, cv2.CV_16S, kernely)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5,0)

#Sobel
x = cv2.filter2D(binary, cv2.CV_16S, 1, 0)
y = cv2.filter2D(binary, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5,0)

#Laplacian
dst = cv2.Laplacian(binary, cv2.CV_16S, ksize=3)
Laplacian = cv2.convertScaleAbs(dst)

#Scharr
x = cv2.Scharr(binary, cv2.CV_32F, 1, 0)
y = cv2.Scharr(binary, cv2.CV_32F, 0, 1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Scharr = cv2.addWeighted(absX, 0.5, absY, 0.5,0)

#Canny
gaussianBlur = cv2.GaussianBlur(binary, (3,3), 0)#高斯滤波
Canny = cv2.Canny(gaussianBlur, 50, 100)

#LOG
gaussianBlur = cv2.GaussianBlur(binary, (3,3), 0)#高斯滤波
dst = cv2.Laplacian(gaussianBlur, cv2.CV_16S, ksize=3)
LOG = cv2.convertScaleAbs(dst)

#效果图
titles=['(a)Source Image', '(b)Binary Image', '(c)Roberts', '(d)Prewitt'
    , '(e)Sobel', '(f)Laplacian', '(g)Scharr', '(h)Canny', '(i)LOG']
images = [rgb_img, binary, Roberts, Prewitt, Sobel, Laplacian, Scharr, Canny, LOG]

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
plt.xticks([])
plt.yticks([])
plt.show()