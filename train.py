import network.unet
import network.unet3p
import util.eval
import util.dataSet
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

def trainUnet(train_dataset, eval_dataset, batch_size):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = paddle.Model(network.unet.UNet(1))
    opt = paddle.optimizer.Momentum(learning_rate=1e-3, parameters=model.parameters(), weight_decay=1e-2)
    model.prepare(opt, paddle.nn.CrossEntropyLoss(axis=1))
    model.fit(train_dataloader, eval_dataloader, epochs=10, verbose=2, save_dir="./net_params")

def trainUnet2(train_dataset, eval_dataset, batch_size):
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
                                        use_shared_memory=False)
    valid_loader = paddle.io.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True,
                                        use_shared_memory=False)
    model = network.unet.UNet(2)
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.01, step_size=30, gamma=0.1, verbose=False)
    opt = paddle.optimizer.Momentum(scheduler, parameters=model.parameters())

    MAX_EPOCH = 20
    for epoch in range(MAX_EPOCH):
        model.train()
        for i, data in enumerate(train_loader()):
            images, labels = data
            predicts = model(images)

            ce_loss = paddle.nn.CrossEntropyLoss(axis=1)
            loss = ce_loss(predicts, labels)

            loss.backward()  # 反向传播
            opt.step()  # 最小化loss，更新参数
            opt.clear_grad()  # 清除梯度
        scheduler.step()

        model.eval()

        all_acc = 0
        all_acc_cls = 0
        all_mean_iu = 0
        all_fwavacc = 0
        count = 1
        for data, label in enumerate(valid_loader()):
            images, labels = data
            predicts = model(data)
            ce_loss = paddle.nn.CrossEntropyLoss(axis=1)
            evalLoss = ce_loss(predicts, labels)
            acc, acc_cls, mean_iu, fwavacc = util.eval.label_accuracy_score(label.numpy(), np.argmax(predicts.numpy(), axis=1), 2)
            all_acc = all_acc + acc
            all_acc_cls = all_acc_cls + acc_cls
            all_mean_iu = all_mean_iu + mean_iu
            all_fwavacc = all_fwavacc + fwavacc
            count += 1


        print("epoch: {}, loss is: {}, evalLoss is: {}".format(epoch, loss.numpy(), evalLoss.numpy()))
        print("acc:{}, acc_cls:{}, mean_iu:{}, fwavacc:{}".format(all_acc/count, all_acc_cls/count, all_mean_iu/count, all_fwavacc/count))
    # 保存模型参数，文件名为Unet_model.pdparams
    paddle.save(model.state_dict(), './net_params/Unet_model.pdparams')
    print("模型保存成功，模型参数保存在Unet_model.pdparams中")

if __name__ == "__main__":
    train_dataset, eval_dataset = util.dataSet.getDataset()
    # 批量大小
    batch_size = 2
    trainUnet(train_dataset, eval_dataset, batch_size)