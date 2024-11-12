from collections import OrderedDict
import os
import torch
from math import log
from collections import defaultdict

from network.model_loss import ChamferLoss, OffsetLoss, UniformLoss
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation



class Model(object):
    def __init__(self, net, phase, opt, writer_tensorboard=None):
        self.net = net
        self.phase = phase
        self.writer_tensorboard = writer_tensorboard

        if self.phase == 'train':
            self.error_log = defaultdict(int)
            self.chamfer_criteria = ChamferLoss()
            self.uniformloss = UniformLoss(loss_name='uniform', alpha=1)
            self.offsetLoss = OffsetLoss()

            self.old_lr = opt.lr_init
            self.lr = opt.lr_init
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=opt.lr_init,
                                              betas=(0.9, 0.999))
            self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.7)
            self.decay_step = opt.decay_iter
        self.step = 0

    def set_input(self, input_pc, radius, label_pc=None):
        """`
        :param
            input_pc       Bx3xN
            up_ratio       int
            label_pc       Bx3xN'
        """
        self.input = input_pc.detach()
        self.radius = radius

        # gt point cloud
        if label_pc is not None:
            self.gt = label_pc.detach()
        else:
            self.gt = None

    def forward(self):
        if self.gt is not None:
            self.predicted, self.gt = self.net(self.input, gt=self.gt)
        else:
            self.predicted = self.net(self.input)

    def get_lr(self, optimizer):
        """Get the current learning rate from optimizer.
        """
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def optimize(self, steps=None, epoch=None):
        """
        run forward and backward, apply gradients
        """
        self.optimizer.zero_grad()
        self.net.train()
        self.forward()


        P1, P2, P3, P4 = self.predicted
        gt = self.gt.permute(0, 2, 1).contiguous()
        gt_downsample_idx = furthest_point_sample(gt, int(P1.shape[1] / 2))
        gt_downsample = gather_operation(self.gt, gt_downsample_idx)
        gt_downsample = gt_downsample.permute(0, 2, 1).contiguous()

        cd_1 = self.compute_chamfer_loss(P1, self.gt)
        cd_2 = self.compute_chamfer_loss(P2, self.gt)
        cd_3 = self.compute_chamfer_loss(P3, self.gt)
        cd_4 = self.compute_chamfer_loss(P4, self.gt)

        if self.writer_tensorboard is not None:
            self.writer_tensorboard.add_scalars(main_tag="train/cd_loss",
                                                tag_scalar_dict={'cd1': cd_1, 'cd2': cd_2, 'cd3': cd_3, 'cd4': cd_4},
                                                global_step=steps
                                                )




        alpha = 0.1
        uniform_1 = self.uniformloss(P1) * alpha
        uniform_2 = self.uniformloss(P2) * alpha
        uniform_3 = self.uniformloss(P3) * alpha
        uniform_4 = self.uniformloss(P4) * alpha



        if self.writer_tensorboard is not None:
            self.writer_tensorboard.add_scalars(main_tag="train/uniform_loss",
                                                tag_scalar_dict={'uniform_1': uniform_1, 'uniform_2': uniform_2,
                                                                 'uniform_3': uniform_3, 'uniform_4': uniform_4},
                                                global_step=steps
                                                )
        loss1, loss2, loss3, loss4 = cd_1 + uniform_1, uniform_2 , cd_3 + uniform_3, uniform_4

        self.gs = self.net.module.get_gs()
        self.offsets = self.net.module.get_offsets()

        if self.writer_tensorboard is not None:
            if steps % 100 == 0:
                for i in range(len(self.gs)):
                    self.writer_tensorboard.add_histogram('data/g-' + str(i + 1), self.gs[i].cpu().detach().numpy(),
                                                          steps)
                    self.writer_tensorboard.add_histogram('data/offset-' + str(i + 1),
                                                          self.offsets[i].cpu().detach().numpy(), steps)

        loss = loss1 + loss2 + loss3 + loss4
        losses = [loss1.item(), loss2.item(), loss3.item(), loss4.item()]
        if self.writer_tensorboard is not None:

            self.writer_tensorboard.add_scalar(tag="train/total_loss",
                                               scalar_value=loss,
                                               global_step=steps
                                               )

        loss.backward()
        self.optimizer.step()
        if steps % self.decay_step == 0 and steps != 0:
            self.lr_scheduler.step()
        lr = self.get_lr(self.optimizer)
        return losses, lr

    def compute_chamfer_loss(self, pc, pc_label):

        loss_chamfer = self.chamfer_criteria(
            pc.transpose(1, 2).contiguous(),
            pc_label.transpose(1, 2).contiguous(), self.radius)

        return loss_chamfer
