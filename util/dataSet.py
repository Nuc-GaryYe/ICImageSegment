import network.unet
import os
import shutil
import numpy as np
from paddle.io import Dataset,DataLoader
from paddle.vision import transforms as T
from paddle.nn import functional as F
import cv2
import paddle
import matplotlib.pyplot as plt
import paddle.nn as nn
from tqdm import tqdm

# 测试集的数量
eval_num = 10
image_size = (256, 256)
# 训练图片路径
train_images_path = "E:/yexy/ICImageSegment/data/dataset/test/train"
# 标签图像路径
label_images_path = "E:/yexy/ICImageSegment/data/dataset/test/train_mask"
# 测试图片路径
test_images_path = "E:/yexy/ICImageSegment/data/dataset/test/val"

class ImageDataset(Dataset):
    def __init__(self, path, transform, imageSize):
        super(ImageDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.imageSize = imageSize

    def _load_image(self, path):
        '''
        该方法作用为通过路径获取图像
        '''
        img = cv2.imread(path)
        img = cv2.resize(img, self.imageSize)
        return img

    def __getitem__(self, index):
        '''
        这里之所以不对label使用transform，因为观察数据集发现label的图像矩阵主要为0或1
        但偶尔也有0-255的值，所以要对label分情况处理
        而对data都进行transform是因为data都是彩色图片，图像矩阵皆为0-255，所以可以统一处理
        '''
        path = self.path[index]
        if len(path) == 2:
            data_path, label_path = path
            data, label = self._load_image(data_path), self._load_image(label_path)
            data, label = self.transform(data), label
            label = label.transpose((2, 0, 1))
            label = label[0, :, :]
            label = np.expand_dims(label, axis=0)
            if True in (label > 1):
                label = label / 255.
            label = label.astype("int64")
            return data, label

        if len(path) == 1:
            data = self._load_image(path[0])
            data = self.transform(data)
            return data

    def __len__(self):
        return len(self.path)

def get_path(image_path):
    files=[]
    for image_name in os.listdir(image_path):
        if image_name.endswith('.jpg') and not image_name.startswith('.'):
            files.append(os.path.join(image_path, image_name))

    return sorted(files)

def get_test_data(test_images_path):
    test_data=[]
    for name in os.listdir(test_images_path):
        img_path=os.path.join(test_images_path,name)
        test_data.append(img_path)
    test_data=np.expand_dims(np.array(test_data),axis=1)
    return test_data

def getDataset():
    images = np.expand_dims(np.array(get_path(train_images_path)), axis=1)
    labels = np.expand_dims(np.array(get_path(label_images_path)), axis=1)
    data = np.array(np.concatenate((images, labels), axis=1))
    np.random.shuffle(data)

    train_data = data[:-eval_num, :]
    eval_data = data[-eval_num:, :]

    eval_transform = T.Compose([
        T.Resize(image_size),
        T.Transpose(),
        T.Normalize(mean=0., std=255.)
    ])
    train_dataset = ImageDataset(train_data, eval_transform, image_size)
    eval_dataset = ImageDataset(eval_data, eval_transform, image_size)
    return train_dataset, eval_dataset

def getTestDataset():
    test_data = get_test_data(train_images_path)
    eval_transform = T.Compose([
        T.Resize(image_size),
        T.Transpose(),
        T.Normalize(mean=0., std=255.)
    ])
    test_dataset = ImageDataset(test_data, eval_transform, image_size)
    return test_dataset