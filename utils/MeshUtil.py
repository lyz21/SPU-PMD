# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Liu Yanzhe
@file:MeshUtil.py
@time:2023/04/07 16:27
@IDE:PyCharm 
@introduce：
"""

import open3d as o3d
import numpy as np
import pyvista as pv


def create_mesh_poisson(pcd, depth=9):
    pcd.estimate_normals()
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    return mesh


def create_mesh_alpha(pcd, alpha=0.05):
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=alpha)
    mesh.compute_vertex_normals()
    return mesh


def create_mesh_Ball(pcd, k=3):
    pcd.estimate_normals()
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = k * avg_dist
    radii = [avg_dist, radius]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(
        radii))  # o3d.utility.DoubleVector()方法是转为open3d格式
    return mesh


def add_points(mesh, pcd_new):
    # Get mesh vertices and triangles
    vertices = mesh.vertices
    triangles = mesh.triangles

    # Calculate triangle centers
    for triangle in triangles:
        p1 = vertices[triangle[0]]
        p2 = vertices[triangle[1]]
        p3 = vertices[triangle[2]]
        center = (p1 + p2 + p3) / 3
        pcd_new.points.append(center)
    return pcd_new


def add_point_pyvista(mesh):
    center_list = []
    for i in range(mesh.number_of_cells):
        cell = mesh.get_cell(i)
        center = np.array(cell.points).mean(axis=0)
        center_list.append(center)
    return center_list


def get_mesh_pyvista(points):
    alpha = get_mean_dis(points)
    if alpha == 0:
        return -1
    points = pv.PolyData(points)
    mesh = points.delaunay_3d(alpha=alpha)
    return mesh


def get_mean_dis(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    distances = pcd.compute_nearest_neighbor_distance()
    mean_dis = np.mean(distances)
    return mean_dis
