import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from pointnet2_ops.pointnet2_utils import grouping_operation
import os
import sys

sys.path.append('../')
sys.path.append(os.getcwd())
from network import up_mesh
'''
The final model structure of the paper
'''

class SPUPMDNet(torch.nn.Module):
    """
    SPU-PMD: Self-Supervised Point Cloud Upsampling via Progressive Mesh Deformation
    Input:
        points: Input points, (B, 3, N_input)
    Output:
        up_point: upsampled results, (B, 3, up_ratio * N_input)
    """

    def __init__(self, up_ratio):
        super(SPUPMDNet, self).__init__()
        step_up_rate = int(np.sqrt(up_ratio))  # if up_ratio=4，step_up_rate=2
        self.stage_1_upsampling = UpsamplingStage(up_ratio=step_up_rate, is_step_1=True, rate=0.1)
        self.stage_2_refinement = RefinmentStage(rate=0.02)
        self.stage_3_upsampling = UpsamplingStage(up_ratio=step_up_rate, rate=0.08)
        self.stage_4_refinement = RefinmentStage(rate=0.01)

        print('Init SPUPMD')

    def forward(self, point_cloud, gt=None):
        point_cloud = point_cloud.float().contiguous()  # example: [16, 3, 256]

        p1_pre, h = self.stage_1_upsampling(point_cloud)  # example: [16, 3, 512]
        p2_pre, h = self.stage_2_refinement(p1_pre, h)  # example: [16, 3, 512]
        p3_pre, h = self.stage_3_upsampling(p2_pre, h)  # example: [16, 3, 1024]
        p4_pre, h = self.stage_4_refinement(p3_pre, h)  # example: [16, 3, 1024]

        P1 = p1_pre.permute(0, 2, 1).contiguous()  # example: [16, 512, 3]
        P2 = p2_pre.permute(0, 2, 1).contiguous()  # example: [16, 512, 3]
        P3 = p3_pre.permute(0, 2, 1).contiguous()  # example: [16, 1024, 3]
        P4 = p4_pre.permute(0, 2, 1).contiguous()  # example: [16, 1024, 3]

        if self.training:
            return [P1, P2, P3, P4], gt
        else:
            return P4

    def get_gs(self):
        return [self.stage_1_upsampling.get_g(), self.stage_2_refinement.get_g(), self.stage_3_upsampling.get_g(),
                self.stage_4_refinement.get_g()]

    def get_offsets(self):
        return [self.stage_1_upsampling.get_offset(), self.stage_2_refinement.get_offset(),
                self.stage_3_upsampling.get_offset(), self.stage_4_refinement.get_offset()]


# Upsampling Stage
class UpsamplingStage(nn.Module):
    """
    Upsampling or Refinement Subnetwork

    Input:
        points: Input points, (B, 3, N_input)
    Output:
        up_point: upsampled results, (B, 3, up_ratio * N_input)
    """

    def __init__(self, up_ratio=2, is_step_1=False, rate=1.0):
        super(UpsamplingStage, self).__init__()
        self.rate = rate
        self.feature_extractor = Transformer_extractor(128, 64)
        self.up_unit = Upsampling_unit(up_ratio=up_ratio)
        self.RFA = RFA(is_step_1=is_step_1, in_channel=128)

        # self.regressor = MLP_CONV(in_channel=128, layer_dims=[64, 3])
        self.regressor = GCR(in_channel=128)
        self.offset = None

    def forward(self, points, pre_h=None):
        point_feat = self.feature_extractor(points)  # point_feat; (B, 128, N)
        point_feat, _ = self.RFA(point_feat, pre_h)  # pre_h:[B,C,N]，example:[16,128,256]
        up_feat, duplicated_point = self.up_unit(point_feat, points)  # # up_feat:[B,C,r*N],example:[16,128,512]or[16,128,1024]

        # offset = self.regressor(up_feat)
        # offset = torch.tanh(offset)
        offset = self.regressor(up_feat) * self.rate
        self.offset = offset

        up_point = duplicated_point + offset

        return up_point, up_feat

    def get_g(self):
        return self.regressor.get_g()

    def get_offset(self):
        return self.offset


# Upsampling Stage
class RefinmentStage(nn.Module):
    """
    Upsampling or Refinement Subnetwork

    Input:
        points: Input points, (B, 3, N_input)
    Output:
        up_point: upsampled results, (B, 3, up_ratio * N_input)
    """

    def __init__(self, is_step_1=False, rate=1.0):
        super(RefinmentStage, self).__init__()
        self.rate = rate
        self.feature_extractor = Transformer_extractor(128, 64)
        self.RFA = RFA(is_step_1=is_step_1, in_channel=128)

        # self.regressor = MLP_CONV(in_channel=128, layer_dims=[64, 3])
        self.regressor = GCR(in_channel=128)
        self.offset = None

    def forward(self, points, pre_h=None):
        point_feat = self.feature_extractor(points)  # point_feat; (B, 128, N) 
        point_feat, pre_h = self.RFA(point_feat, pre_h)  # pre_h:[B,C,N]，example:[16,128,256]

        # offset = self.regressor(up_feat)
        # offset = torch.tanh(offset)
        offset = self.regressor(point_feat) * self.rate
        self.offset = offset

        up_point = points + offset

        return up_point, pre_h

    def get_g(self):
        return self.regressor.get_g()

    def get_offset(self):
        return self.offset


class SubNetwork(nn.Module):
    """
    Upsampling or Refinement Subnetwork

    Input:
        points: Input points, (B, 3, N_input)
    Output:
        up_point: upsampled results, (B, 3, up_ratio * N_input)
    """

    def __init__(self, up_ratio=2, is_step_1=False, rate=1.0):
        super(SubNetwork, self).__init__()

        self.up_mesh = up_mesh.Upsampling(up_ratio=up_ratio)
        self.feature_extractor = Transformer_extractor(128, 64)  # (dim,hidden_dim)
        self.RFA = RFA(is_step_1=is_step_1, up_ratio=up_ratio, in_channel=128)

        # self.regressor = MLP_CONV(in_channel=128, layer_dims=[64, 3])
        self.gcr = GCR(in_channel=128)

        self.rate = rate

    def forward(self, points, pre_h=None):
        points = self.up_mesh.up(points)  # points:[B，3,N*r]
        point_feat = self.feature_extractor(points)  # point_feat; (B, 128, N)
        up_feat, pre_h = self.RFA(point_feat, pre_h)  # pre_h:[B,C,N]，example:[16,128,256]
        # offset = self.regressor(up_feat)
        # offset = torch.tanh(offset)
        offset = self.gcr(up_feat) * self.rate
        self.offset = offset

        up_point = points + offset

        return up_point, pre_h

    def get_g(self):
        return self.gcr.get_g()

    def get_offset(self):
        return self.offset


class RFA(nn.Module):
    """
    RFA module, gated recycling unit (GRU)
    Input:
        cur_f: Tensor, (B, in_channel, N)
        prev_h: Tensor, (B, in_channel, N)
    Output:
        h: Tensor, (B, in_channel, N)
        h: Tensor, (B, in_channel, N)
    """

    def __init__(self, is_step_1, up_ratio=1, in_channel=256):
        super(RFA, self).__init__()
        self.is_step_1 = is_step_1
        self.up_ratio = up_ratio
        if is_step_1:
            return

        if up_ratio > 1:
            self.duplicated_branch = nn.Upsample(scale_factor=up_ratio)  # [N,C,W]->[N,C,scale*W]

        self.conv_z = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.sigmoid)
        self.conv_r = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.sigmoid)
        self.conv_h = Conv1d(in_channel * 2, in_channel, if_bn=True,
                             activation_fn=torch.relu)  # torch.relu--> torch.tanh -lyz 23.07.03

    def forward(self, cur_f, pre_h):
        """
        Args:
            cur_f: Tensor, (B, in_channel, N)
            prev_h: Tensor, (B, in_channel, N)
        Returns:
            h: Tensor, (B, in_channel, N)
            h: Tensor, (B, in_channel, N)
        """

        if self.is_step_1 or pre_h is None:
            return cur_f, cur_f

        if self.up_ratio > 1:
            pre_h = self.duplicated_branch(pre_h)
        z = self.conv_z(torch.cat([cur_f, pre_h], 1))  # z example:[16,128,512]or[16,128,1024]
        r = self.conv_r(torch.cat([cur_f, pre_h], 1))  # r example:[16,128,512]or[16,128,1024]
        h_hat = self.conv_h(torch.cat([r * cur_f, pre_h], 1))
        h = (1 - z) * cur_f + z * h_hat  # h example:[16,128,512]or[16,128,1024]

        return h, h




class GCR(nn.Module):
    """
    Used in the coordinate reconstruction module to control whether the point moves or not
    Input:
        feature: Tensor, (B, channel, N)。example:[16,128,8192]
    Output:
        offset: Tensor, (B, 3, N)。example:[16,3,8192]
    """

    def __init__(self, in_channel=128):
        super(GCR, self).__init__()
        self.mlp_g = MLP_CONV(in_channel=in_channel, layer_dims=[64, 16, 1])
        self.mlp_reg = MLP_CONV(in_channel=in_channel, layer_dims=[64, 3])
        self.g = None
        self.ftanh = FTanh()
        self.fsign = FSign()

    def forward(self, feature):
        # g = torch.sigmoid(self.mlp_g(feature))  # example: g.shape [16,1,8192]
        # g = torch.softmax(self.mlp_g(feature), dim=-1) * 10e3
        # g = torch.nn.functional.normalize(self.mlp_g(feature), dim=-1)
        g = self.fsign(self.mlp_g(feature))  # example: g.shape [16,1,8192]
        self.g = g
        # with torch.no_grad():
        #     g[g < 0.3] = 0
        #     g[g >= 0.3] = 1
        # offset = self.ftanh(self.mlp_reg(feature))  # example: offset [16,3,8192]
        offset = torch.tanh(self.mlp_reg(feature))  # example: offset [16,3,8192]
        offset = g * offset  # example: offset [16,3,8192]
        return offset

    def get_g(self):
        return self.g


####################################################### Add GCR -lyz-23.07.03

class Transformer_extractor(nn.Module):
    """
    Point-wise feature extractor.

    Input:
        points: input points, (B, 3, N_input)
    Output:
        point_feat: ouput feature, (B, dim_feat, N_input)
    """

    def __init__(self, dim_feat, hidden_dim):
        super(Transformer_extractor, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, dim_feat])
        self.mlp_2 = MLP_CONV(in_channel=dim_feat * 2, layer_dims=[dim_feat * 2, dim_feat])
        self.point_transformer = Transformer(dim_feat, dim=hidden_dim)

    def forward(self, points):
        feature_1 = self.mlp_1(points)
        global_feature = torch.max(feature_1, 2, keepdim=True)[0]
        feature_2 = torch.cat([feature_1, global_feature.repeat((1, 1, feature_1.size(2)))], 1)
        feature_3 = self.mlp_2(feature_2)
        point_feat = self.point_transformer(feature_3, points)
        return point_feat


class Upsampling_unit(nn.Module):
    """
    Point upsampling unit

    Input:
        point_feat: input feature, (B, dim_feat, N_input)
        points: input points, (B, 3, N_input)
    Output:
        up_feat: upsampled feature, (B, dim, up_ratio * N_input)
        duplicated_point: upsampled results, (B, 3, up_ratio * N_input)
    """

    def __init__(self, up_ratio=2):
        super(Upsampling_unit, self).__init__()
        # self.mlp_1 = MLP_CONV(in_channel=128, layer_dims=[64, 32])
        # self.mlp_2 = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)
        self.mlp = MLP_Res(in_dim=128, hidden_dim=128, out_dim=128)
        # self.deconv_branch = nn.ConvTranspose1d(32, 128, up_ratio, up_ratio,bias=False)  # (in_channels, in_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None）
        self.duplicated_branch = nn.Upsample(scale_factor=up_ratio)  # [N,C,W]->[N,C,scale*W]
        self.duplicated_branch_mesh = up_mesh.Upsampling(up_ratio=up_ratio)

    def forward(self, point_feat, points):  # point_feat:[B,C,N]
        # deconved_feat = self.deconv_branch(self.mlp_1(point_feat))  # duplicated_feat:[B,128,r*N]；self.mlp_1(point_feat):[B,32,N]
        duplicated_feat = self.duplicated_branch(point_feat)  # duplicated_feat:[N,C,r*N]
        # up_feat = self.mlp_2(torch.cat([deconved_feat, duplicated_feat], 1))  # up_feat:[B,C,r*N]
        up_feat = self.mlp(duplicated_feat)  # [B,C,r*N]
        up_feat = torch.relu(up_feat)
        duplicated_point = self.duplicated_branch_mesh.up(points)
        return up_feat, duplicated_point


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_knn(nsample, xyz, new_xyz, include_self=True):
    """
    Find k-NN of new_xyz in xyz

    """
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    idx = torch.argsort(sqrdists, dim=-1, descending=False)[:, :, pad: nsample + pad]
    return idx.int()


def query_knn_point(k, xyz, new_xyz):
    """
    Use knn to divide groups and return subscripts.
    :param k:int
    :param xyz:Tensor,[B,N,3]
    :param new_xyz:Tensor,[B,N,3]
    :return:
    group_idx Tensor,[B,N,k]
    """
    dist = square_distance(new_xyz, xyz)
    _, group_idx = dist.topk(k, largest=False)
    return group_idx.int()


class Conv1d(nn.Module):
    """
    Input:
        input: Tensor, [B,in_channel,N],example:[16,256,8192]
    Output:
        out: Tensor,[B,out_channel,N],example:[16,128,8192]
    """

    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, if_bn=True, activation_fn=torch.relu):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)
        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True,
                 activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


class MLP_CONV(nn.Module):
    """
    Input:
        input: Tensor, [B,in_channel,N],example: [16,256,8192]
    Output:
         Tensor,[B,layer_dims[-1],N],example: [16,128,8192]
    """

    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class MLP_Res(nn.Module):
    """
    MLP
    Input:
        x: Tensor, [B,in_dim,N],example: [16,256,8192]
    Output:
        out: Tensor,[B,out_dim,N],example: [16,128,8192]
    """

    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out


class Transformer(nn.Module):
    """
    [Point Transformer](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf)

    feed forward of transformer
    Args:
        x: Tensor of features, (B, in_channel, n)
        pos: Tensor of positions, (B, 3, n)

    Returns:
        y: Tensor of features with attention, (B, in_channel, n)

    """

    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn_point(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key
        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y + identity


class FTanh(nn.Module):
    def __init__(self):
        super(FTanh, self).__init__()

    def forward(self, x):
        a = - 2 / (1 + torch.exp(-2 * x))
        b = 2 - (2 / (1 + torch.exp(-2 * x)))
        res = torch.where(x < 0, a, b)
        return res


class FSign(torch.nn.Module):
    def __init__(self):
        super(FSign, self).__init__()

    def forward(self, x):
        return 0.5 * (torch.sign(x) + 1)

