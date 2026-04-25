

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from PIL import Image
from mmdet.models.builder import DEFORM, build_loss
from mmengine.visualization import Visualizer
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from mmcv.cnn import (constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
import scipy

foo = SummaryWriter(comment='test')


def makeGaussian(size, a=1, b=0.5, fwhm1=100, fwhm2=25):
    x1 = np.arange(0, size, 1, float)
    x2 = np.arange(0, size, 1, float)
    y1 = x1[:, np.newaxis]
    y2 = x2[:, np.newaxis]
    x10 = y10 = size // 2
    x20 = y20 = size // 2
    g1 = np.exp(-4 * np.log(2) * ((x1 - x10) ** 2 + (y1 - y10) ** 2) / fwhm1 ** 2)
    g2 = np.exp(-4 * np.log(2) * ((x2 - x20) ** 2 + (y2 - y20) ** 2) / fwhm2 ** 2)
    gaussian = a * g1 - b * g2
    return gaussian


class Conv3Density(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, bn_momentum=0.1):
        super(Conv3Density, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=7, stride=1, padding=3)
        self.BatchNorm1 = nn.BatchNorm2d(16)
        self.BatchNorm2 = nn.BatchNorm2d(32)
        self.BatchNorm3 = nn.BatchNorm2d(16)
        self.act = nn.ReLU(inplace=True)
        self.MaxPool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        xs = self.conv1(x)
        x1 = self.BatchNorm1(xs)
        xs = self.act(x1)
        xs = self.conv2(xs)
        xs = self.BatchNorm2(xs)
        xs = self.act(xs)
        xs = self.conv3(xs)
        xs = self.BatchNorm3(xs)
        xs = self.act(xs)
        xs = self.conv4(xs)
        return xs


class Conv3DensityGroup(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, bn_momentum=0.1):
        super(Conv3DensityGroup, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=7, stride=1, padding=3)
        self.GroupNorm1 = nn.GroupNorm(8, 16)
        self.GroupNorm2 = nn.GroupNorm(8, 32)
        self.GroupNorm3 = nn.GroupNorm(8, 16)
        self.act = nn.ReLU(inplace=True)
        self.MaxPool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        xs = self.conv1(x)
        x1 = self.GroupNorm1(xs)
        xs = self.act(x1)
        xs = self.conv2(xs)
        xs = self.GroupNorm2(xs)
        xs = self.act(xs)
        xs = self.conv3(xs)
        xs = self.GroupNorm3(xs)
        xs = self.act(xs)
        xs = self.conv4(xs)
        return xs


class Conv3DensitySKernel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, bn_momentum=0.1):
        super(Conv3DensitySKernel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.BatchNorm1 = nn.BatchNorm2d(16, eps=0.001, momentum=bn_momentum, affine=True)
        self.BatchNorm2 = nn.BatchNorm2d(16, eps=0.001, momentum=bn_momentum, affine=True)
        self.BatchNorm3 = nn.BatchNorm2d(16, eps=0.001, momentum=bn_momentum, affine=True)
        self.BatchNorm4 = nn.BatchNorm2d(16, eps=0.001, momentum=bn_momentum, affine=True)
        self.BatchNorm5 = nn.BatchNorm2d(16, eps=0.001, momentum=bn_momentum, affine=True)
        self.act = nn.ReLU(inplace=True)
        self.MaxPool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        xs = self.conv1(x)
        x1 = self.BatchNorm1(xs)
        xs = self.act(x1)
        xs = self.conv2(xs)
        xs = self.BatchNorm2(xs)
        xs = self.act(xs)
        xs = self.conv3(xs)
        xs = self.BatchNorm3(xs)
        xs = self.act(xs)
        xs = self.conv4(xs)
        xs = self.BatchNorm4(xs)
        xs = self.act(xs)
        xs = self.conv5(xs)
        xs = self.BatchNorm5(xs)
        xs = self.act(xs)
        xs = self.conv6(xs)
        return xs


class Conv3DensityDilation(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, bn_momentum=0.1):
        super(Conv3DensityDilation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=4, dilation=3)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=4, dilation=3)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=4, dilation=3)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=4, dilation=3)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=4, dilation=3)
        self.conv6 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=4, dilation=3)
        self.BatchNorm1 = nn.BatchNorm2d(16, eps=0.001, momentum=bn_momentum, affine=True)
        self.BatchNorm2 = nn.BatchNorm2d(16, eps=0.001, momentum=bn_momentum, affine=True)
        self.BatchNorm3 = nn.BatchNorm2d(16, eps=0.001, momentum=bn_momentum, affine=True)
        self.BatchNorm4 = nn.BatchNorm2d(16, eps=0.001, momentum=bn_momentum, affine=True)
        self.BatchNorm5 = nn.BatchNorm2d(16, eps=0.001, momentum=bn_momentum, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        xs = self.conv1(x)
        x1 = self.BatchNorm1(xs)
        xs = self.act(x1)
        xs = self.conv2(xs)
        xs = self.BatchNorm2(xs)
        xs = self.act(xs)
        xs = self.conv3(xs)
        xs = self.BatchNorm3(xs)
        xs = self.act(xs)
        xs = self.conv4(xs)
        xs = self.BatchNorm4(xs)
        xs = self.act(xs)
        xs = self.conv5(xs)
        xs = self.BatchNorm5(xs)
        xs = self.act(xs)
        xs = self.conv6(xs)
        return xs


class DeformSampler(nn.Module):
    def __init__(self, grid_size=256, padding_size=30, deform_input_size=512):
        super(DeformSampler, self).__init__()
        self.grid_size = grid_size
        self.padding_size = padding_size
        self.global_size = self.grid_size + 2 * self.padding_size
        self.input_size_net = deform_input_size
        gaussian_weights = torch.FloatTensor(makeGaussian(2 * self.padding_size + 1))
        # Spatial transformer localization-network
        self.filter = nn.Conv2d(1, 1, kernel_size=(2 * self.padding_size + 1, 2 * self.padding_size + 1), bias=False)
        self.filter.weight[0].data[:, :, :] = gaussian_weights

        self.P_basis = torch.zeros(2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size)
        for k in range(2):
            for i in range(self.global_size):
                for j in range(self.global_size):
                    self.P_basis[k, i, j] = k * (i - self.padding_size) / (self.grid_size - 1.0) + (1.0 - k) * (
                            j - self.padding_size) / (self.grid_size - 1.0)
        self.P = torch.autograd.Variable(
            torch.zeros(1, 2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size).cuda(),
            requires_grad=False)
        self.P[0, :, :, :] = self.P_basis
        h = torch.linspace(-1.0, 1.0, self.input_size_net).view(-1, 1).repeat(1, self.input_size_net)
        w = torch.linspace(-1.0, 1.0, self.input_size_net).repeat(self.input_size_net, 1)
        self.grid_o = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)

    def create_grid(self, x, p):
        P = p.expand(x.size(0), 2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size)
        x_cat = torch.cat((x, x), 1)
        p_filter = self.filter(x)
        x_mul = torch.mul(P, x_cat).view(-1, 1, self.global_size, self.global_size)
        all_filter = self.filter(x_mul).view(-1, 2, self.grid_size, self.grid_size)
        x_filter = all_filter[:, 0, :, :].contiguous().view(-1, 1, self.grid_size, self.grid_size)
        y_filter = all_filter[:, 1, :, :].contiguous().view(-1, 1, self.grid_size, self.grid_size)
        x_filter = x_filter / p_filter
        y_filter = y_filter / p_filter
        xgrids = x_filter * 2 - 1
        ygrids = y_filter * 2 - 1
        xgrids = torch.clamp(xgrids, min=-1, max=1)
        ygrids = torch.clamp(ygrids, min=-1, max=1)
        xgrids = xgrids.view(-1, 1, self.grid_size, self.grid_size)
        ygrids = ygrids.view(-1, 1, self.grid_size, self.grid_size)
        grid = torch.cat((xgrids, ygrids), 1)
        grid = nn.Upsample(size=(self.input_size_net, self.input_size_net), mode='bilinear')(grid)
        n, c, h, w = grid.size()
        grid_o = self.grid_o.repeat(n, 1, 1, 1).type_as(x).to(x.device)

        grid_l = torch.transpose(grid, 1, 2)
        grid_l = torch.transpose(grid_l, 2, 3)
        grid_r = (49 * grid_o + 1 * grid_l) / 50
        # grid_v = (9 * grid_o + 2 * grid_l) / 10
        #
        # grid_img = Image.open('/home/zx/CSRDETE/OBBDetection/grid_2448_2448.png').convert('RGB')
        # grid_img = Image.open('/home/zx/CSRDETE/OBBDetection/sample_points.png').convert('RGBA')
        # grid_resized = grid_img.resize((512, 512), Image.BILINEAR)
        # del grid_img
        # grid_resized = np.float32(np.array(grid_resized)) / 255.
        # grid_resized = grid_resized.transpose((2, 0, 1))
        # grid_resized = torch.from_numpy(grid_resized.copy())
        # grid_resized = torch.unsqueeze(grid_resized, 0).expand(1, grid_resized.shape[-3],
        #                                                        grid_resized.shape[-2], grid_resized.shape[-1])
        # grid_resized = grid_resized.cuda()
        # grid_output = F.grid_sample(grid_resized, grid_v, align_corners=True)
        # deformed_grid = vutils.make_grid(grid_output, normalize=True, scale_each=True)
        # foo.add_image('Deformed Grid', deformed_grid)

        return grid_r, grid

    def forward(self, x, xs):
        if not self.grid_size == xs.size()[2]:
            xs = nn.Upsample(size=(self.grid_size, self.grid_size), mode='bilinear')(xs)
        xs = xs.view(-1, self.grid_size * self.grid_size)
        xs = xs.view(-1, 1, self.grid_size, self.grid_size)
        xs_hm = nn.ReplicationPad2d(self.padding_size)(xs)
        grid_r, grid = self.create_grid(xs_hm, self.P)
        x_sampled = F.grid_sample(x, grid_r, align_corners=True)

        return x_sampled, grid


class RegistrationModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(RegistrationModule, self).__init__()
        self.flow_conv1 = nn.Conv2d(out_channels * 2, out_channels // 4, kernel_size=3, padding=1, bias=True)
        self.flow_conv2 = nn.Conv2d(out_channels // 4, 2, kernel_size=3, padding=1, bias=True)
        self.BatchNorm1 = nn.BatchNorm2d(out_channels // 4, eps=0.001, momentum=0.1, affine=True)
        self.BatchNorm2 = nn.BatchNorm2d(2, eps=0.001, momentum=0.1, affine=True)
        self.Relu = nn.ReLU(inplace=True)
        self.norm = torch.tensor([[[[128, 128]]]]).cuda()
        h = torch.linspace(-1.0, 1.0, 128).view(-1, 1).repeat(1, 128)
        w = torch.linspace(-1.0, 1.0, 128).repeat(128, 1)
        self.grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)

    def forward(self, x_u, x_d):
        H, W = x_d.shape[2:]
        x_u = F.interpolate(x_u, size=(H, W), mode='bilinear', align_corners=False)
        flow = self.Relu(self.BatchNorm1(self.flow_conv1(torch.cat([x_u, x_d], 1))))
        flow = self.Relu(self.BatchNorm2(self.flow_conv2(flow)))
        x_a = self.flow_warp(x_d, flow)
        return x_a, x_u

    def flow_warp(self, input, flow):
        n, c, h, w = input.size()
        grid = self.grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        flow = flow.permute(0, 2, 3, 1)
        flow = flow / self.norm
        grid = grid + flow
        output = F.grid_sample(input, grid, align_corners=True)
        return output


class RegistrationModulev2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(RegistrationModulev2, self).__init__()
        self.Conv_deform = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv_uni = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.flow_conv1 = nn.Conv2d(out_channels * 2, out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.flow_conv2 = nn.Conv2d(out_channels // 4, 2, kernel_size=3, padding=1, bias=False)
        self.GroupNorm1 = nn.GroupNorm(32, out_channels // 4)
        self.GroupNorm2 = nn.GroupNorm(2, 2)
        self.Relu = nn.ReLU(inplace=True)

        self.norm = torch.tensor([[[[128, 128]]]]).cuda()
        h = torch.linspace(-1.0, 1.0, 128).view(-1, 1).repeat(1, 128)
        w = torch.linspace(-1.0, 1.0, 128).repeat(128, 1)
        self.grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)

    def forward(self, x_u, x_d):
        H, W = x_d.shape[2:]
        x_u_inter = F.interpolate(x_u, size=(H, W), mode='bilinear', align_corners=False)
        x_u = self.Conv_uni(x_u_inter)
        x_d = self.Conv_deform(x_d)
        flow = self.Relu(self.GroupNorm1(self.flow_conv1(torch.cat([x_u, x_d], 1))))
        flow = self.Relu(self.GroupNorm2(self.flow_conv2(flow)))
        x_a = self.flow_warp(x_d, flow)
        return x_a, x_u_inter

    def flow_warp(self, input, flow):
        n, c, h, w = input.size()
        grid = self.grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        flow = flow.permute(0, 2, 3, 1)
        flow = flow / self.norm
        grid = grid + flow
        output = F.grid_sample(input, grid, align_corners=True)
        return output


class RegistrationModulev3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(RegistrationModulev3, self).__init__()
        self.Conv_deform = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv_uni = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.flow_conv1 = nn.Conv2d(out_channels * 2, out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.flow_conv2 = nn.Conv2d(out_channels // 4, 2, kernel_size=3, padding=1, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(out_channels // 4)
        self.BatchNorm2 = nn.BatchNorm2d(2)
        self.Relu = nn.ReLU(inplace=True)

        self.norm = torch.tensor([[[[128, 128]]]]).cuda()
        h = torch.linspace(-1.0, 1.0, 128).view(-1, 1).repeat(1, 128)
        w = torch.linspace(-1.0, 1.0, 128).repeat(128, 1)
        self.grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)

    def forward(self, x_u, x_d):
        H, W = x_d.shape[2:]
        x_u_inter = F.interpolate(x_u, size=(H, W), mode='bilinear', align_corners=False)
        x_u = self.Conv_uni(x_u_inter)
        x_d = self.Conv_deform(x_d)
        flow = self.Relu(self.BatchNorm1(self.flow_conv1(torch.cat([x_u, x_d], 1))))
        flow = self.Relu(self.BatchNorm2(self.flow_conv2(flow)))
        x_a = self.flow_warp(x_d, flow)
        return x_a, x_u_inter

    def flow_warp(self, input, flow):
        n, c, h, w = input.size()
        grid = self.grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        flow = flow.permute(0, 2, 3, 1)
        flow = flow / self.norm
        grid = grid + flow
        output = F.grid_sample(input, grid, align_corners=True)
        return output


class A2UFusion(nn.Module):
    def __init__(self, init_channels, r=4):
        super(A2UFusion, self).__init__()
        channels = init_channels
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_a, x_u):
        x = x_a + x_u
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        weight = self.sigmoid(xlg)

        xo = 2 * x_a * weight + 2 * x_u * (1 - weight)
        return xo


class Grad(nn.Module):
    """
    N-D gradient loss
    """

    def __init__(self, penalty='l1'):
        super(Grad, self).__init__()
        self.penalty = penalty

    def _diffs(self,
               y):  # y shape(bs, nfeat, vol_shape) # 定义一个私有方法_diffs，用于计算图像y在各个维度上的差分，y的形状为(bs, nfeat, vol_shape)，其中bs是批次大小，nfeat是特征数，vol_shape是图像尺寸
        ndims = y.ndimension() - 2  # 计算图像的维度数，减去前两个维度（bs和nfeat）
        df = [None] * ndims  # 创建一个空列表，长度为维度数，用于存储各个维度上的差分
        for i in range(ndims):  # 遍历每个维度
            d = i + 2  # y shape(bs, c, d, h, w) # 计算当前维度在y中的索引，加上前两个维度
            # permute dimensions to put the ith dimension first
            if i == 0:
                # 创建一个列表r，存储重排后的维度顺序，将当前维度放在第一位，其他维度按原顺序排列
                y = y.permute(d, 0, 1, 3)  # 对y进行维度重排，按照r中的顺序
                dfi = y[1:, ...] - y[:-1, ...]  # 计算当前维度上相邻两个体素/像素之间的差值
                df[i] = dfi.permute(1, d, 0, 3)
            elif i == 1:
                y = y.permute(d, 0, 1, 2)
                dfi = y[1:, ...] - y[:-1, ...]  # 计算当前维度上相邻两个体素/像素之间的差值
                df[i] = dfi.permute(1, 2, d, 0)
            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            # 创建一个列表r，存储恢复后的维度顺序，将当前维度放回原来的位置
            # 对差值进行维度重排，并存储到列表中

        return df  # 返回列表

    def forward(self, pred):  # 定义类的前向传播方法，接受一个参数pred，表示预测出的形变场
        ndims = pred.ndimension() - 2  # 计算形变场的维度数
        if pred.is_cuda:  # 判断形变场是否在GPU上
            df = torch.zeros(1).cuda()  # 如果是，则创建一个零值张量，并转换为Variable类型，用于存储梯度损失
            df.requires_grad_(True)
        else:
            df = torch.zeros(1)  # 如果不是，则创建一个零值张量，并转换为Variable类型，用于存储梯度损失
            df.requires_grad_(True)
        for f in self._diffs(pred):  # 遍历形变场在各个维度上的差分
            if self.penalty == 'l1':  # 判断梯度损失的类型是否为'l1'
                df.data += f.abs().mean() / ndims  # 如果是，则计算差分的绝对值的均值，并除以维度数，累加到梯度损失上
            else:
                assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty  # 如果不是，则断言梯度损失的类型为'l2'，否则报错
                df.data += f.pow(2).mean() / ndims  # 计算差分的平方的均值，并除以维度数，累加到梯度损失上
        return df  # 返回梯度损失


@DEFORM.register_module()
class Deform(nn.Module):
    def __init__(self,
                 density_use_layer=0,
                 density_channels=None,
                 loss_mask=None,
                 loss_re=None,
                 grid_size=256,
                 padding_size=15,
                 deform_input_size=512,
                 sigma=25,
                 depth=50,
                 pretrained=None,
                 pretrained_deform=None):
        super(Deform, self).__init__()
        self.density_use_layer = density_use_layer
        self.density_channels = density_channels
        self.loss_mask = build_loss(loss_mask)
        self.loss_re = build_loss(loss_re)
        self.grid_size = grid_size
        self.padding_size = padding_size
        self.deform_input_size = deform_input_size
        self.sigma = sigma
        density_channel = self.density_channels[self.density_use_layer]
        self.density_net = Conv3DensityGroup()
        self.loss_grad = Grad()
        from mmdet.models.backbones import ResNet
        self.deform_branch = ResNet(depth=depth, num_stages=4, out_indices=(0, 1, 2, 3), frozen_stages=1,
                                    norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True)
        self.deform_sampler = DeformSampler(self.grid_size, self.padding_size, self.deform_input_size)
        self.registration = RegistrationModulev2(density_channel, density_channel)
        self.A2UFusion = A2UFusion(density_channel)
        self.init_weights(pretrained=pretrained, pretrained_res=pretrained_deform)

    def init_weights(self, pretrained=None, pretrained_res=None):

        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
            pass
        else:
            raise TypeError('pretrained must be a str or None.'
                            f' But received {type(pretrained)}.')
        if isinstance(pretrained_res, str):
            logger = logging.getLogger()
            load_checkpoint(self.deform_branch, pretrained_res, strict=False, logger=logger)
        elif pretrained_res is None:
            # use default initializer or customized initializer in subclasses
            pass

    def get_gt_mask_gaussian(self, x, gt_obboxes):
        # x:输入特征，shape为[N,C,H,W]
        # gt_obboxes:真值box坐标，shape为[B,N,5],其中N为每张图片box的数量
        N, C, H, W = x.shape
        density_map = torch.zeros(N, 1, H, W).to(x.device)
        kernel = self.gaussian_kernel()
        for n in range(N):
            B = gt_obboxes[n].shape[0]
            for b in range(B):
                box = gt_obboxes[n][b]  # 获取每个box的坐标，[x, y, w, h, theta]
                cx = box[0]
                cy = box[1]
                cx = cx * W / 1024
                cy = cy * H / 1024
                point_x = cx.long()
                point_y = cy.long()
                full_kernel_size = kernel.shape[0]
                kernel_size = full_kernel_size // 2

                min_img_x = max(0, point_x - kernel_size)
                min_img_y = max(0, point_y - kernel_size)
                max_img_x = min(point_x + kernel_size + 1, W - 1)
                max_img_y = min(point_y + kernel_size + 1, H - 1)
                assert max_img_x > min_img_x
                assert max_img_y > min_img_y

                kernel_x_min = kernel_size - point_x if point_x <= kernel_size else 0
                kernel_y_min = kernel_size - point_y if point_y <= kernel_size else 0
                kernel_x_max = kernel_x_min + max_img_x - min_img_x
                kernel_y_max = kernel_y_min + max_img_y - min_img_y
                assert kernel_x_max > kernel_x_min
                assert kernel_y_max > kernel_y_min

                density_map[n, 0, min_img_y:max_img_y, min_img_x:max_img_x] += kernel[kernel_y_min:kernel_y_max,
                                                                               kernel_x_min:kernel_x_max]

        return density_map

    def gaussian_kernel(self):
        sigma = self.sigma
        kernel_size = sigma * 4
        img_shape = (kernel_size * 2 + 1, kernel_size * 2 + 1)
        img_center = (img_shape[0] // 2, img_shape[1] // 2)
        arr = np.zeros(img_shape)
        arr[img_center] = 1
        arr = scipy.ndimage.filters.gaussian_filter(arr, sigma, mode='constant')
        kernel = arr / arr.max()
        kernel = torch.from_numpy(kernel).cuda()
        return kernel

    def forward_train(self, x_uni, img, gt_obboxes):
        use_layer = self.density_use_layer
        x_uni = x_uni[use_layer]
        xs = self.density_net(img)
        target_label = self.get_gt_mask_gaussian(xs, gt_obboxes)
        loss_mask = self.loss_mask(xs, target_label)
        x_deformed, grid = self.deform_sampler(img, xs)
        loss_smooth = self.loss_grad(grid)
        # x_deformed = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)
        x_deformed = self.deform_branch.conv1(x_deformed)
        x_deformed = self.deform_branch.bn1(x_deformed)
        x_deformed = self.deform_branch.relu(x_deformed)
        x_deformed = self.deform_branch.maxpool(x_deformed)
        outs = []
        for i, layer_name in enumerate(self.deform_branch.res_layers):
            res_layer = getattr(self.deform_branch, layer_name)
            if i == 0:
                x_deformed = res_layer(x_deformed)
                # x_u = F.interpolate(x_uni, size=(128, 128), mode='bilinear', align_corners=False)
                x_align, x_u = self.registration(x_uni, x_deformed)
                # loss_re = self.loss_re(x_align, x_u)
                x = self.A2UFusion(x_align, x_u)
                # x = x_align+x_u
            else:
                x = res_layer(x)
            outs.append(x)
            outs[0] = x_uni
        return loss_mask, loss_smooth, tuple(outs)
        # return tuple(outs)

    def simple_test(self, x_uni, img):
        use_layer = self.density_use_layer
        x_uni = x_uni[use_layer]
        xs = self.density_net(img)
        x_deformed, _ = self.deform_sampler(img, xs)
        # x_deformed = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)

        xs = vutils.make_grid(xs, normalize=True, scale_each=True)
        foo.add_image('Density Map', xs)
        x_deformed_v = vutils.make_grid(x_deformed, normalize=True, scale_each=True)
        foo.add_image('Deformed Image', x_deformed_v)

        x_deformed = self.deform_branch.conv1(x_deformed)
        x_deformed = self.deform_branch.bn1(x_deformed)
        x_deformed = self.deform_branch.relu(x_deformed)
        x_deformed = self.deform_branch.maxpool(x_deformed)
        outs = []
        for i, layer_name in enumerate(self.deform_branch.res_layers):
            res_layer = getattr(self.deform_branch, layer_name)
            if i == 0:
                x_deformed = res_layer(x_deformed)
                # x_u = F.interpolate(x_uni, size=(128, 128), mode='bilinear', align_corners=False)
                x_align, x_u = self.registration(x_uni, x_deformed)
                x = self.A2UFusion(x_align, x_u)
                # x = x_align+x_u
                # visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')], save_dir='temp_dir')
                # drawn_img0 = visualizer.draw_featmap(x_deformed[0], channel_reduction='squeeze_mean', )
                # visualizer.add_image('x_deformed', drawn_img0)
                # drawn_img1 = visualizer.draw_featmap(x_u[0], channel_reduction='squeeze_mean', )
                # visualizer.add_image('x_u', drawn_img1)
                # drawn_img2 = visualizer.draw_featmap(x_align[0], channel_reduction='squeeze_mean', )
                # visualizer.add_image('x_align', drawn_img2)
                # drawn_img3 = visualizer.draw_featmap(x[0], channel_reduction='squeeze_mean', )
                # visualizer.add_image('x_fusion', drawn_img3)
            else:
                x = res_layer(x)
            outs.append(x)
            outs[0] = x_uni
        return tuple(outs)
