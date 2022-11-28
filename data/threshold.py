import cv2
import matplotlib.pyplot as plt

for i in range(40, 60):
    lena_BGR = cv2.imread("dataset/srcImage/{}.jpg".format(str(i)))
    lena_RGB = cv2.cvtColor(lena_BGR, cv2.COLOR_BGR2RGB)
    # display BGR lena
    # plt.subplot(1, 4, 1)
    # plt.imshow(lena_BGR)
    # plt.axis('off')
    # plt.title('img_BGR')

    # display RGB lena
    plt.subplot(1, 2, 1)
    plt.imshow(lena_RGB)
    plt.axis('off')
    plt.title('img_RGB')

    # 转换成灰度图像,并执行高斯模糊
    gray = cv2.cvtColor(lena_BGR, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # display RGB lena
    # blurred_RGB = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    # plt.subplot(1, 4, 3)
    # plt.imshow(blurred_RGB)
    # plt.axis('off')
    # plt.title('img_gray')

    # 将图像中小于60的置为0,大于100的置为255
    # 返回的temp是一个元组,temp[0]表示设置的阈值,也就是100; temp[1]是变换后的图像
    temp = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
    thresh = temp[0]
    lena_thresh = temp[1]

    # display lena_thresh image
    #plt.subplot(1, 2, 2)
    #plt.imshow(lena_thresh, cmap='gray')
    #plt.axis('off')
    #plt.title('img_thresh')

    #plt.show()
    cv2.imwrite("dataset/seg/{}.jpg".format(str(i)), lena_thresh)

