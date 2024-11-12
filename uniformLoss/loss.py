import torch
import torch.nn as nn
import os, sys

sys.path.append('../')
# from auction_match import auction_match
# import pointnet2.pointnet2_utils as pn2_utils
from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils as pn2_utils
import math
from knn_cuda import KNN


class Loss(nn.Module):
    def __init__(self, radius=1.0):
        super(Loss, self).__init__()
        self.radius = radius
        self.knn_uniform = KNN(k=2, transpose_mode=True)
        self.knn_repulsion = KNN(k=20, transpose_mode=True)

    """
    def get_emd_loss(self, pred, gt, radius=1.0):
        '''
        pred and gt is B N 3
        '''
        idx, _ = auction_match(pred.contiguous(), gt.contiguous())
        # gather operation has to be B 3 N
        # print(gt.transpose(1,2).shape)
        matched_out = pn2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
        matched_out = matched_out.transpose(1, 2).contiguous()
        dist2 = (pred - matched_out) ** 2
        dist2 = dist2.view(dist2.shape[0], -1)  # <-- ???
        dist2 = torch.mean(dist2, dim=1, keepdims=True)  # B,
        dist2 /= radius
        return torch.mean(dist2)
    """

    def get_uniform_loss(self, pcd, percentage=[0.02, 0.04, 0.06, 0.08, 0.1], radius=1.0):
        B, N, C = pcd.shape[0], pcd.shape[1], pcd.shape[2]
        npoint = int(N * 0.05)
        loss = 0
        further_point_idx = pn2_utils.furthest_point_sample(pcd.contiguous(), npoint)
        new_xyz = pn2_utils.gather_operation(pcd.permute(0, 2, 1).contiguous(), further_point_idx)  # B,C,N
        for p in percentage:
            nsample = int(N * p)
            r = math.sqrt(p * radius)
            disk_area = math.pi * (radius ** 2) / N

            idx = pn2_utils.ball_query(r, nsample, pcd.contiguous(),
                                       new_xyz.permute(0, 2, 1).contiguous())  # b N nsample

            expect_len = math.sqrt(disk_area)

            grouped_pcd = pn2_utils.grouping_operation(pcd.permute(0, 2, 1).contiguous(), idx)  # B C N nsample
            grouped_pcd = grouped_pcd.permute(0, 2, 3, 1)  # B N nsample C

            grouped_pcd = torch.cat(torch.unbind(grouped_pcd, dim=1), dim=0)  # B*N nsample C

            dist, _ = self.knn_uniform(grouped_pcd, grouped_pcd)
            # print(dist.shape)
            uniform_dist = dist[:, :, 1:]  # B*N nsample 1
            # 此处改动：1e-8 --> 1e-1   ————lyz 2023.07.17
            e_ = 1e-1
            uniform_dist = torch.abs(uniform_dist + e_)
            uniform_dist = torch.mean(uniform_dist, dim=1)
            uniform_dist = (uniform_dist - expect_len) ** 2 / (expect_len + e_)
            mean_loss = torch.mean(uniform_dist)
            mean_loss = mean_loss * math.pow(p * 100, 2)  # math.pow(底数，指数)
            loss += mean_loss
        return loss / len(percentage)

    def get_uniform_loss_one(self, pcd, p=0.02, radius=1.0):
        B, N, C = pcd.shape[0], pcd.shape[1], pcd.shape[2]
        npoint = int(N * 0.05)
        loss = 0
        further_point_idx = pn2_utils.furthest_point_sample(pcd.contiguous(), npoint)
        new_xyz = pn2_utils.gather_operation(pcd.permute(0, 2, 1).contiguous(), further_point_idx)  # B,C,N
        nsample = int(N * p)
        r = math.sqrt(p * radius)
        disk_area = math.pi * (radius ** 2) / N

        idx = pn2_utils.ball_query(r, nsample, pcd.contiguous(),
                                   new_xyz.permute(0, 2, 1).contiguous())  # b N nsample

        expect_len = math.sqrt(disk_area)

        grouped_pcd = pn2_utils.grouping_operation(pcd.permute(0, 2, 1).contiguous(), idx)  # B C N nsample
        grouped_pcd = grouped_pcd.permute(0, 2, 3, 1)  # B N nsample C

        grouped_pcd = torch.cat(torch.unbind(grouped_pcd, dim=1), dim=0)  # B*N nsample C

        dist, _ = self.knn_uniform(grouped_pcd, grouped_pcd)
        # print(dist.shape)
        uniform_dist = dist[:, :, 1:]  # B*N nsample 1
        e_ = 1e-8
        uniform_dist = torch.abs(uniform_dist + e_)
        uniform_dist = torch.mean(uniform_dist, dim=1)
        uniform_dist = (uniform_dist - expect_len) ** 2 / (expect_len + e_)
        mean_loss = torch.mean(uniform_dist)
        mean_loss = mean_loss * math.pow(p * 100, 2)  # math.pow(底数，指数)
        loss += mean_loss

        return loss


def get_repulsion_loss(self, pcd, h=0.05):
    dist, idx = self.knn_repulsion(pcd, pcd)  # B N k   # 例：dist.shape [8,256,20]

    # dist = dist[:, :, 1:4] ** 2  # top 4 cloest neighbors   # 例：dist [8,256,3]
    # dist = dist[:, :, 1:] ** 2  # top 4 cloest neighbors --> 使用所有（20个）邻居 lyz-23.07.02 # 例：dist [8,256,19]
    dist = dist[:, :, 1:6] ** 2  # top 4 cloest neighbors --> top 6 cloest neighbors

    # 这里的思路是用h-dist，也就是dist越大的结果越小，dist大于h的直接舍弃不要。
    loss = torch.clamp(-dist + h, min=0)  # clamp是压缩数据到区间内。clamp(input,min,max),小于min的直接被截为0！
    loss = torch.mean(loss)
    return loss


def get_discriminator_loss(self, pred_fake, pred_real):
    real_loss = torch.mean((pred_real - 1) ** 2)
    fake_loss = torch.mean(pred_fake ** 2)
    loss = real_loss + fake_loss
    return loss


def get_generator_loss(self, pred_fake):
    fake_loss = torch.mean((pred_fake - 1) ** 2)
    return fake_loss


def get_discriminator_loss_single(self, pred, label=True):
    if label == True:
        loss = torch.mean((pred - 1) ** 2)
        return loss
    else:
        loss = torch.mean((pred) ** 2)
        return loss


if __name__ == "__main__":
    loss = Loss().cuda()
    point_cloud = torch.rand(2, 256, 3).cuda()
    uniform_loss = loss.get_uniform_loss(point_cloud)
    repulsion_loss = loss.get_repulsion_loss(point_cloud)
    print('uniform_loss:', uniform_loss)
    print('repulsion_loss:', repulsion_loss)
