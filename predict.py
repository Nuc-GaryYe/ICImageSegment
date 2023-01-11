import network.unet
import network.unet3p
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
import util.dataSet
work_path = "E:/yexy/ICImageSegment/result"
checkpoint_path="./net_params/15.pdparams"

if __name__ == "__main__":
    test_dataset = util.dataSet.getTestDataset()
    save_dir = work_path

    # 实例化，网络三选一，默认U-Net
    #model = paddle.Model(network.unet.UNet(2))  # U-Net
    model = paddle.Model(network.unet.UNet(2))
    model.load(checkpoint_path)

    for i, img in tqdm(enumerate(test_dataset)):
        img = paddle.to_tensor(img).unsqueeze(0)
        predict = np.array(model.predict_batch(img)).squeeze(0).squeeze(0)
        predict = predict.argmax(axis=0)
        image_path = test_dataset.path[i]
        path_lst = image_path[0].split("/")
        save_path = os.path.join(save_dir, path_lst[-1][:-4]) + ".jpg"
        cv2.imwrite(save_path, predict * 255)
