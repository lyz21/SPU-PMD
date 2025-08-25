# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Liu Yanzhe
@file:pc_utils_lyz.py
@time:2024/01/26 17:37
@IDE:PyCharm 
@introduce：
"""
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), "../../")))

from glob import glob
import torch
import time
from network import operations
import numpy as np
# from utils import pc_utils
# from network.model_loss import ChamferLoss, CD_dist
# import trimesh
# from scipy.spatial import cKDTree
# from multiprocessing import Pool  # 多进程
from uniformLoss.loss import Loss as UniformLoss
import random


# lyz
def make_non_uniform_data(file_path='/home/tsmc/teamip/lyz/data/PU1K/test/input_512/gt_2048',
                          target_folder='/home/tsmc/teamip/lyz/data/PU1K_non_uniform/input_512', patch_num=16,
                          DEVICE='cuda:0'):

    test_files = glob(file_path + '/*.xyz', recursive=True)
    for point_path in test_files:
        print('=' * 20)
        print('point_path:', point_path)
        folder = os.path.basename(os.path.dirname(point_path))
        save_folder = os.path.join(target_folder, folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        out_path = os.path.join(save_folder, point_path.split('/')[-1][
                                             :-4])

        data = np.loadtxt(point_path).astype(np.float32)
        data = data[np.newaxis, ...]
        data = torch.from_numpy(data).transpose(2, 1).to(
            device=DEVICE).float()
        num_patches = int(data.shape[2] / 4)
        idx, seeds = operations.fps_subsample(data, num_patches, NCHW=True)
        patches, _, _ = operations.group_knn(patch_num, seeds, data,
                                             NCHW=True)
        for patch in patches:
            patch = patch.transpose(1, 0)

            point_cloud = np.empty((0, 3))
            for i, p in enumerate(patch):
                p = p.transpose(1, 0)  # [256,3]
                p = p[random.randint(0, patch_num - 1)].cpu().numpy()
                point_cloud = np.vstack((point_cloud, p[np.newaxis, ...]))

            print('point_cloud:', point_cloud.shape)  # [32,3]

            np.savetxt(out_path + '.xyz', point_cloud,
                       fmt='%.6f')
            print('save:', out_path + '.xyz')


def get_uniform_loss(file_path, DEVICE='cuda:0'):
    print('file_path:', file_path)
    loss1, loss2, loss3, loss4, loss5, loss6 = 0, 0, 0, 0, 0, 0
    UniformLoss_ = UniformLoss()
    test_files = glob(file_path + '/*.xyz', recursive=True)
    for point_path in test_files:
        data = np.loadtxt(point_path).astype(np.float32)
        data = data[np.newaxis, ...]
        data = torch.from_numpy(data).to(device=DEVICE).float()
        loss1 = loss1 + UniformLoss_.get_uniform_loss_one(data, p=0.004)
        loss2 = loss2 + UniformLoss_.get_uniform_loss_one(data, p=0.006)
        loss3 = loss3 + UniformLoss_.get_uniform_loss_one(data, p=0.008)
        loss4 = loss4 + UniformLoss_.get_uniform_loss_one(data, p=0.010)
        loss5 = loss5 + UniformLoss_.get_uniform_loss_one(data, p=0.012)
        loss6 = loss6 + UniformLoss_.get_uniform_loss_one(data, p=0.5)
    print('UniformLoss 0.004:', loss1 / len(test_files))
    print('UniformLoss 0.006:', loss2 / len(test_files))
    print('UniformLoss 0.008:', loss3 / len(test_files))
    print('UniformLoss 0.010:', loss4 / len(test_files))
    print('UniformLoss 0.012:', loss5 / len(test_files))
    print('UniformLoss 0.5:', loss6 / len(test_files))


def get_uniform_loss_2(file_path, DEVICE='cuda:0'):
    print('file_path:', file_path)
    loss = 0
    UniformLoss_ = UniformLoss()
    test_files = glob(file_path + '/*.xyz', recursive=True)
    for point_path in test_files:
        data = np.loadtxt(point_path).astype(np.float32)
        data = data[np.newaxis, ...]
        data = torch.from_numpy(data).to(device=DEVICE).float()
        loss = loss + UniformLoss_.get_uniform_loss_one(data, p=0.012)
    print('UniformLoss:', loss / len(test_files))


def get_uniform_loss_single(point_path, DEVICE='cuda:0'):
    print('point_path:', point_path)
    UniformLoss_ = UniformLoss()
    data = np.loadtxt(point_path).astype(np.float32)
    data = data[np.newaxis, ...]
    data = torch.from_numpy(data).to(device=DEVICE).float()
    loss1 = UniformLoss_.get_uniform_loss_one(data, p=0.004)
    loss2 = UniformLoss_.get_uniform_loss_one(data, p=0.006)
    loss3 = UniformLoss_.get_uniform_loss_one(data, p=0.008)
    loss4 = UniformLoss_.get_uniform_loss_one(data, p=0.010)
    loss5 = UniformLoss_.get_uniform_loss_one(data, p=0.012)
    print('UniformLoss 0.004:', round(loss1.item(), 6))
    print('UniformLoss 0.006:', round(loss2.item(), 6))
    print('UniformLoss 0.008:', round(loss3.item(), 6))
    print('UniformLoss 0.010:', round(loss4.item(), 6))
    print('UniformLoss 0.012:', round(loss5.item(), 6))


