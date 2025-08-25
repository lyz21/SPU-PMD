# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Liu Yanzhe
@file:up_mesh.py
@time:2023/04/07 14:29
@IDE:PyCharm 
@introduce：mesh interpolation
"""
import sys
import os
from torch import nn

root_path = os.path.abspath(__file__)
root_path_ = '/'.join(root_path.split('/')[:-1])
sys.path.append(root_path_)
root_path_ = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path_)

import torch
import open3d as o3d
from utils import MeshUtil
import operations
import numpy as np


# import h5py
# import time

def add_points_obj_2(points, r):
    n = points.shape[0] * r
    pcd = o3d.geometry.PointCloud()
    for i in range(5):
        if points.shape[0] >= n:
            break
        pcd.points = o3d.utility.Vector3dVector(points)
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist, 1.3 * avg_dist]  # paper parm 23.08.02
        pcd.estimate_normals()
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        mesh = mesh.subdivide_midpoint(number_of_iterations=1)


        points = np.array(mesh.vertices)
    if points.shape[0] < n:
        # '''
        print('nn.upsample')
        print(points.shape)
        xyz = torch.tensor(points).T
        xyz = torch.unsqueeze(xyz, 0)
        up = nn.Upsample(scale_factor=r)
        xyz = up(xyz)
        points = xyz.reshape([-1, 3]).numpy()
        # '''
    return points


def add_points_pyvista(points, up_ratio):
    '''
    :param points:input point cloud [N,3]
    :param up_ratio:
    :return:
    '''
    n = points.shape[0] * up_ratio
    while True:
        mesh = MeshUtil.get_mesh_pyvista(points)
        if mesh == -1:
            return -1
        center_list = MeshUtil.add_point_pyvista(mesh)
        points = np.concatenate((points, np.array(center_list)), axis=0)
        if points.shape[0] >= n:
            break
    return points


class Upsampling:
    def __init__(self, up_ratio=4, mesh_method_name='ball'):
        self.up_ratio = up_ratio
        self.mesh_method_name = mesh_method_name

    def up(self, datas, k=3):
        """
        :param datas:[B,3,N]
        :param k:create_mesh_Ball method's radius of ball/alpha_shape method's alpha
        :return:np.array，[B,3,N*r]
        """
        DEVICE = datas.get_device()
        datas = datas.transpose(2, 1).detach().cpu().contiguous() 
        datas = np.array(datas)
        list_input = []
        for (i, data) in enumerate(datas):  # [N,3]
            if self.mesh_method_name == 'pyvista_alpha':
                points = add_points_pyvista(points=data, up_ratio=self.up_ratio)
                if type(points) is int:
                    print('The', i, 'data insert wrong,use method：ball')
                    points = add_points_obj_2(points=data, r=self.up_ratio)
            else:
                # print('---------------------------insert method：', self.mesh_method_name)
                points = add_points_obj_2(points=data, r=self.up_ratio)
            points = torch.tensor(points).to(device=DEVICE).float()
            points_ = torch.unsqueeze(points, 0)  # to [1,N*k,3]
            idx, sampled_pc = operations.fps_subsample(points_, npoint=data.shape[0] * self.up_ratio,
                                                       NCHW=False)  # to [1,N*r,3]
            list_input.append(sampled_pc.reshape(-1, 3).cpu().numpy())  # to [N*r,3]

        return torch.tensor(np.array(list_input)).to(device=DEVICE).float().transpose(2,
                                                                                      1).contiguous()  # to [B,3,N]
