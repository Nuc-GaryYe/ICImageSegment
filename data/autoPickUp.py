import os
from pytesseract import image_to_string
from PIL import Image
from PIL import ImageGrab
import win32api
import win32con
import time
from ctypes import *
# 截图，获取需要识别的区域
os.chdir("d:\Python")

def get_color(x, y):
    gdi32 = windll.gdi32
    user32 = windll.user32
    hdc = user32.GetDC(None)  # 获取颜色值
    pixel = gdi32.GetPixel(hdc, x, y)  # 提取RGB值
    r = pixel & 0x0000ff
    g = (pixel & 0x00ff00) >> 8
    b = pixel >> 16
    return [r, g, b]

time.sleep(3)
x = 0
y = 69
m = 1773
n = 1000

# for i in range(1, 10):
#     box = (x, y, m, n)
#     img = ImageGrab.grab(box)
#     img.save("d:\Python1\img" + str(i) + ".jpeg")
#     win32api.keybd_event(0x23, 0, 0, 0)  # enter
#     time.sleep(1)
#     print(get_color(x,y))
#     i = 1
i = 1
j = 1
point_left_top = get_color(20, 100)
point_left_buttom = get_color(20, 950)
point_right_top = get_color(1750, 100)
point_right_buttom = get_color(1750, 950)
black = [0, 0, 0]
while(i < 80):
    box = (x, y, m, n)

    while((point_right_buttom != black and point_right_buttom != [0, 1, 0] ) or (point_right_top != black and point_right_top != [0, 1, 0])):
        img = ImageGrab.grab(box)
        time.sleep(0.2)
        img.save("d:\Python\\" + str(i) + "_" + str(j) +".jpg")
        win32api.keybd_event(0x27, 0, 0, 0)  # enter
        j = j + 1
        time.sleep(0.2)
        point_right_top = get_color(1750, 100)
        point_right_buttom = get_color(1750, 950)

    img = ImageGrab.grab(box)
    img.save("d:\Python\\" + str(i) + "_" + str(j) +".jpg")
    i = i + 1

    while(j != 1):
        win32api.keybd_event(0x25, 0, 0, 0)  # enter
        time.sleep(0.1)
        j = j - 1

    time.sleep(0.1)
    win32api.keybd_event(0x28, 0, 0, 0)  # enter
    time.sleep(0.3)
    point_left_top = get_color(20,100)
    point_left_buttom = get_color(20, 950)
    point_right_top = get_color(1750,100)
    point_right_buttom = get_color(1750,950)

    if(point_left_buttom == black and point_left_top == black and point_right_top == black and point_right_buttom ==black):
        break
